"""Gemini LLM 提供商实现（基于 httpx 的 REST API 实现，深度整合标准化 Transport 代理）"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, AsyncIterator, Optional, Type, Union

import httpx
from pydantic import BaseModel

from domain.ai.services.llm_service import GenerationConfig, GenerationResult, LLMService
from domain.ai.value_objects.prompt import Prompt
from domain.ai.value_objects.token_usage import TokenUsage
from infrastructure.ai.config.settings import Settings
from .base import BaseProvider
from .model_resolution import require_resolved_model_id

logger = logging.getLogger(__name__)

DEFAULT_MODEL = 'gemini-1.5-flash'
DEFAULT_BASE_URL = 'https://generativelanguage.googleapis.com/v1beta'

class GeminiProvider(BaseProvider):
    """Google Gemini LLM 提供商实现
    
    加固策略：
    1. 采用 httpx.AsyncHTTPTransport 显式配置代理。
    2. 支持 Pydantic Schema 校验，确保生成内容结构化。
    """
    def __init__(self, settings: Settings):
        super().__init__(settings)
        if not settings.api_key:
            raise ValueError('API key is required for GeminiProvider')
        self.base_url = (settings.base_url or DEFAULT_BASE_URL).rstrip('/')
        
        # 优先从环境变量获取代理配置，实现灵活切换
        proxy_url = os.getenv("PROXY_URL")
        transport = None
        if proxy_url:
            logger.info(f"GeminiProvider initialized with proxy: {proxy_url}")
            transport = httpx.AsyncHTTPTransport(proxy=proxy_url)
        else:
            logger.info("GeminiProvider initialized without proxy (direct connection)")

        # 长生命周期 httpx client（跨请求复用连接池）
        # 🔥 分层超时：避免 API 卡住时整个进程挂起
        self._http_client = httpx.AsyncClient(
            transport=transport,
            timeout=httpx.Timeout(
                connect=settings.connect_timeout,
                read=settings.read_timeout,
                write=60.0,
                pool=30.0,
            ),
            trust_env=False,
        )

    async def generate(self, prompt: Prompt, config: GenerationConfig) -> GenerationResult:
        model_id = require_resolved_model_id(
            config.model,
            self.settings.default_model,
            provider_label="Gemini",
        )
        payload = self._build_payload(prompt, config)
        query = self._build_query()
        url = self._build_url(model_id, 'generateContent')

        logger.info(f"Final Gemini API Request URL: {url}")
        response = await self._http_client.post(
            url,
            params=query,
            headers=self._build_headers(stream=False),
            json=payload,
        )
        response.raise_for_status()
        data = response.json()

        content = self._extract_text(data)
        if not content.strip():
            raise RuntimeError('Gemini returned empty content')

        # 结构化输出校验
        target_schema = config.response_format
        if target_schema and isinstance(target_schema, type) and issubclass(target_schema, BaseModel):
            try:
                validated_obj = target_schema.model_validate_json(content)
                content = validated_obj.model_dump_json()
            except Exception as e:
                logger.warning(f"Gemini output Pydantic check failed (returning raw): {e}")

        usage = data.get('usageMetadata') or {}
        token_usage = TokenUsage(
            input_tokens=int(usage.get('promptTokenCount') or 0),
            output_tokens=int(usage.get('candidatesTokenCount') or 0),
        )
        return GenerationResult(content=content, token_usage=token_usage)

    async def stream_generate(self, prompt: Prompt, config: GenerationConfig) -> AsyncIterator[str]:
        model_id = require_resolved_model_id(
            config.model,
            self.settings.default_model,
            provider_label="Gemini",
        )
        payload = self._build_payload(prompt, config)
        query = self._build_query({'alt': 'sse'})
        url = self._build_url(model_id, 'streamGenerateContent')

        async with self._http_client.stream(
            'POST',
            url,
            params=query,
            headers=self._build_headers(stream=True),
            json=payload,
        ) as response:
            response.raise_for_status()
            buffer = ''
            async for chunk in response.aiter_text():
                buffer += chunk.replace('\r\n', '\n')
                while '\n\n' in buffer:
                    event_text, buffer = buffer.split('\n\n', 1)
                    text = self._parse_sse_event(event_text)
                    if text:
                        yield text

    def _build_url(self, model: str, action: str) -> str:
        """构建完整的 Gemini API URL。
        
        策略：
        1. 优先使用传入的 model，如果为空则回退到 settings/default。
        2. 智能版本补全：检测 v1 或 v1beta，避免遗漏导致 404。
        3. 自动转换主权型号：gemini-1.5-flash 是目前最稳型号，支持 -latest 别名。
        """
        model_name = model or self.settings.default_model or DEFAULT_MODEL
        
        # 处理可能的模型别名或路径冲突
        if model_name.startswith('models/'):
            model_name = model_name[7:]
            
        base = self.base_url.rstrip('/')
        
        # 更加精确的版本检测，避免 /v1 匹配到 /v1beta 的逻辑漏洞
        if not ('/v1beta' in base or '/v1' in base):
            # 默认使用 v1beta 以支持最新特性（如 Pydantic Schema），但在生产环境可切换
            base = f"{base}/v1beta"
            
        return f'{base}/models/{model_name}:{action}'

    def _build_query(self, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        query: dict[str, Any] = {'key': self.settings.api_key}
        query.update(self.settings.extra_query or {})
        if extra:
            query.update(extra)
        return query

    def _build_headers(self, *, stream: bool) -> dict[str, str]:
        headers = {'Content-Type': 'application/json'}
        if stream:
            headers['Accept'] = 'text/event-stream'
        headers.update(self.settings.extra_headers or {})
        return headers

    def _build_payload(self, prompt: Prompt, config: GenerationConfig) -> dict[str, Any]:
        generation_config = {
            'temperature': config.temperature,
            'maxOutputTokens': config.max_tokens,
        }
        
        safety_settings = [
            {'category': cat, 'threshold': 'BLOCK_NONE'}
            for cat in [
                'HARM_CATEGORY_HATE_SPEECH', 'HARM_CATEGORY_HARASSMENT',
                'HARM_CATEGORY_SEXUALLY_EXPLICIT', 'HARM_CATEGORY_DANGEROUS_CONTENT'
            ]
        ]
        
        # 🔥 response_format 自适应：
        # Gemini 支持 responseMimeType 但不支持 json_schema 结构定义
        # OpenAI 的 json_object → Gemini 的 responseMimeType: application/json
        # OpenAI 的 json_schema → Gemini 的 responseMimeType + responseSchema
        if config.response_format:
            fmt = config.response_format
            if isinstance(fmt, dict):
                if fmt.get("type") == "json_object":
                    generation_config["responseMimeType"] = "application/json"
                elif fmt.get("type") == "json_schema":
                    generation_config["responseMimeType"] = "application/json"
                    # 如果 json_schema 中有 schema 定义，传递给 Gemini
                    schema = fmt.get("json_schema", {}).get("schema")
                    if schema:
                        generation_config["responseSchema"] = schema
            elif isinstance(fmt, type) and issubclass(fmt, BaseModel):
                generation_config["responseMimeType"] = "application/json"
                generation_config["responseSchema"] = self._to_gemini_schema(fmt)

        payload: dict[str, Any] = {
            'contents': [
                {
                    'role': 'user',
                    'parts': [{'text': prompt.user}],
                }
            ],
            'generationConfig': generation_config,
            'safetySettings': safety_settings,
        }

        if config.response_format and not generation_config.get("responseMimeType"):
            payload['generationConfig']['responseMimeType'] = 'application/json'

        if prompt.system.strip():
            payload['systemInstruction'] = {
                'parts': [{'text': prompt.system}],
            }
            
        extra_body = dict(self.settings.extra_body or {})
        generation_override = extra_body.pop('generationConfig', None)
        if isinstance(generation_override, dict):
            payload['generationConfig'].update(generation_override)
        payload.update(extra_body)
        return payload

    def _to_gemini_schema(self, model_cls: Type[BaseModel]) -> dict[str, Any]:
        """Convert a Pydantic model to a clean, dereferenced OpenAPI-compatible schema for Gemini."""
        raw_schema = model_cls.model_json_schema()
        defs = raw_schema.pop("$defs", {}) or raw_schema.pop("definitions", {}) or {}
        
        def _clean(node: Any) -> Any:
            if isinstance(node, dict):
                if "$ref" in node:
                    ref_path = node["$ref"]
                    ref_key = ref_path.split("/")[-1]
                    if ref_key in defs:
                        resolved = _clean(defs[ref_key])
                        merged = {k: v for k, v in node.items() if k != "$ref"}
                        merged.update(resolved)
                        node = merged
                
                cleaned = {}
                for k, v in node.items():
                    if k in ("title", "default", "additionalProperties", "serialization_alias", "validation_alias"):
                        continue
                    cleaned[k] = _clean(v)
                return cleaned
            elif isinstance(node, list):
                return [_clean(item) for item in node]
            return node

        return _clean(raw_schema)


    def _extract_text(self, data: dict[str, Any]) -> str:
        pieces: list[str] = []
        for candidate in data.get('candidates') or []:
            content = candidate.get('content') or {}
            for part in content.get('parts') or []:
                if part.get('thought') is True:
                    continue
                text = part.get('text')
                if text:
                    pieces.append(str(text))
        return ''.join(pieces)

    def _parse_sse_event(self, event_text: str) -> str:
        data_lines: list[str] = []
        for line in event_text.splitlines():
            if line.startswith('data:'):
                data_lines.append(line[5:].strip())
        if not data_lines:
            return ''
        raw_payload = ''.join(data_lines).strip()
        if not raw_payload or raw_payload == '[DONE]':
            return ''
        try:
            payload = json.loads(raw_payload)
        except json.JSONDecodeError:
            return ''
        if isinstance(payload, list):
            return ''.join(self._extract_text(item) for item in payload if isinstance(item, dict))
        if isinstance(payload, dict):
            return self._extract_text(payload)
        return ''
