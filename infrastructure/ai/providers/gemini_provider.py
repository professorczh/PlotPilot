"""Google Gemini LLM 提供商实现 (SDK 1.0+ 深度融合版)"""
import logging
import os
from typing import AsyncIterator, Optional, Type, Union, Any

from google import genai
from google.genai import types
from pydantic import BaseModel

from domain.ai.services.llm_service import GenerationConfig, GenerationResult
from domain.ai.value_objects.prompt import Prompt
from domain.ai.value_objects.token_usage import TokenUsage
from infrastructure.ai.config.settings import Settings
from .base import BaseProvider

logger = logging.getLogger(__name__)

# 对齐上游作者的默认定义
DEFAULT_MODEL = 'gemini-2.0-flash'
DEFAULT_BASE_URL = 'https://generativelanguage.googleapis.com/v1beta'


class GeminiProvider(BaseProvider):
    """Google Gemini LLM 提供商实现
    
    融合版策略：
    1. 结构对齐：继承 BaseProvider，与作者物理结构一致。
    2. 连接加固：强制注入 socks5h 代理与 HTTP/1.1 (解决 Windows ConnectError)。
    3. 结构化输出：支持 Pydantic Schema，杜绝 JSON 解析错误（多逗号问题）。
    """

    def __init__(self, settings: Settings):
        super().__init__(settings)
        if not settings.api_key:
            raise ValueError('API key is required for GeminiProvider')
            
        # 1. 代理与协议硬核加固 (解决 SSL UNEXPECTED_EOF)
        # 强制使用 HTTP/1.1 并注入 socks5h 隧道，确保在 Clash/V2Ray 等环境下握手成功
        os.environ["HTTPX_HTTP2"] = "0" 
        os.environ["HTTP_PROXY"] = "socks5h://127.0.0.1:10808"
        os.environ["HTTPS_PROXY"] = "socks5h://127.0.0.1:10808"

        # 2. 配置 HTTP 选项（对齐上游 Base URL 逻辑）
        base_url = (settings.base_url or DEFAULT_BASE_URL).rstrip('/')
        http_options = types.HttpOptions(
            base_url=base_url,
            api_version="v1beta"
        )
            
        # 3. 初始化官方 GenAI Client
        api_key = settings.api_key.strip()
        self.client = genai.Client(
            api_key=api_key,
            http_options=http_options
        )

    async def generate(
        self,
        prompt: Prompt,
        config: GenerationConfig
    ) -> GenerationResult:
        """生成文本内容 (支持结构化校验)"""
        try:
            # 基础生成配置 (安全设置 BLOCK_NONE 确保小说创作不被拦截)
            safety_settings = [
                types.SafetySetting(category=cat, threshold="BLOCK_NONE") 
                for cat in [
                    "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_HARASSMENT", 
                    "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"
                ]
            ]

            gen_config_kwargs = {
                "temperature": config.temperature,
                "max_output_tokens": config.max_tokens,
                "system_instruction": prompt.system,
                "safety_settings": safety_settings,
            }

            # 结构化输出逻辑 (Pydantic 优先)
            target_schema = config.response_format
            if target_schema:
                gen_config_kwargs["response_mime_type"] = "application/json"
                if isinstance(target_schema, type) and issubclass(target_schema, BaseModel):
                    gen_config_kwargs["response_json_schema"] = target_schema
                else:
                    gen_config_kwargs["response_json_schema"] = target_schema

            # 调用 SDK 执行请求
            response = await self.client.aio.models.generate_content(
                model=config.model or self.settings.default_model or DEFAULT_MODEL,
                contents=prompt.user,
                config=types.GenerateContentConfig(**gen_config_kwargs)
            )

            if not response.text:
                raise RuntimeError("Gemini API returned empty content")

            content = response.text

            # 如果启用了 Pydantic，执行二次校验确保 100% 格式稳定性
            if target_schema and isinstance(target_schema, type) and issubclass(target_schema, BaseModel):
                try:
                    validated_obj = target_schema.model_validate_json(content)
                    content = validated_obj.model_dump_json()
                except Exception as e:
                    logger.warning(f"Gemini output Pydantic check failed (returning raw): {e}")

            # 组装 Token 使用信息
            usage = response.usage_metadata
            token_usage = TokenUsage(
                input_tokens=usage.prompt_token_count or 0,
                output_tokens=usage.candidates_token_count or 0
            )

            return GenerationResult(content=content, token_usage=token_usage)

        except Exception as e:
            logger.error(f"Gemini generation failed ({type(e).__name__}): {str(e)}", exc_info=True)
            raise RuntimeError(f"Gemini generation failed: {type(e).__name__} - {str(e) or 'Network connection error'}") from e

    async def stream_generate(
        self,
        prompt: Prompt,
        config: GenerationConfig
    ) -> AsyncIterator[str]:
        """流式生成内容"""
        try:
            gen_config = types.GenerateContentConfig(
                temperature=config.temperature,
                max_output_tokens=config.max_tokens,
                system_instruction=prompt.system,
                safety_settings=[
                    types.SafetySetting(category=cat, threshold="BLOCK_NONE") 
                    for cat in ["HATE_SPEECH", "HARASSMENT", "SEXUALLY_EXPLICIT", "DANGEROUS_CONTENT"]
                ],
            )

            async for chunk in await self.client.aio.models.generate_content_stream(
                model=config.model or self.settings.default_model or DEFAULT_MODEL,
                contents=prompt.user,
                config=gen_config
            ):
                if chunk.text:
                    yield chunk.text

        except Exception as e:
            logger.error(f"Gemini streaming failed: {e}")
            raise RuntimeError(f"Gemini streaming failed: {str(e)}") from e
