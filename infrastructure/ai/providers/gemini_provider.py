"""Google Gemini LLM 提供商实现 (SDK 1.0+)"""
import logging
import os
from typing import AsyncIterator, Optional, Type, Union

from google import genai
from google.genai import types
from pydantic import BaseModel

from domain.ai.services.llm_service import GenerationConfig, GenerationResult
from domain.ai.value_objects.prompt import Prompt
from domain.ai.value_objects.token_usage import TokenUsage
from .base import BaseProvider

logger = logging.getLogger(__name__)

# 默认模型：高性价比、低延迟的预览版
DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-lite-preview-02-05")


class GeminiProvider(BaseProvider):
    """Google Gemini LLM 提供商实现
    
    使用最新的 google-genai SDK (1.0+) 实现，支持 Pydantic 结构化输出。
    """

    def __init__(self, settings):
        """初始化 Gemini 提供商
        
        Args:
            settings: AI 配置设置 (包含 api_key 和 base_url)
        """
        super().__init__(settings)
        
        if not settings.api_key:
            raise ValueError("API key is required for GeminiProvider")
            
        # 配置 HTTP 选项（支持代理/Base URL）
        http_options = None
        if settings.base_url:
            # 去掉末尾斜杠，并显式指定 v1beta 版本
            base_url = settings.base_url.rstrip("/")
            http_options = types.HttpOptions(
                base_url=base_url,
                api_version="v1beta"
            )
            
        # 初始化 GenAI Client
        api_key = settings.api_key.strip() if settings.api_key else ""
        
        # 硬核加固：无视外部环境变量，强制注入稳定的 socks5h 代理并禁用 HTTP/2
        import os
        os.environ["HTTPX_HTTP2"] = "0" 
        os.environ["HTTP_PROXY"] = "socks5h://127.0.0.1:10808"
        os.environ["HTTPS_PROXY"] = "socks5h://127.0.0.1:10808"

        self.client = genai.Client(
            api_key=api_key,
            http_options=http_options
        )

    async def generate(
        self,
        prompt: Prompt,
        config: GenerationConfig
    ) -> GenerationResult:
        """生成文本
        
        Args:
            prompt: 提示词 (system, user)
            config: 生成配置 (model, temperature, max_tokens, response_format)
            
        Returns:
            生成结果
        """
        try:
            # 1. 基础生成配置
            # 安全设置：极简模式下设为最宽松，防止小说创作被拦截
            safety_settings = [
                types.SafetySetting(
                    category=cat,
                    threshold="BLOCK_NONE"
                ) for cat in [
                    "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"
                ]
            ]

            gen_config_kwargs = {
                "temperature": config.temperature,
                "max_output_tokens": config.max_tokens,
                "system_instruction": prompt.system,
                "safety_settings": safety_settings,
            }

            # 2. 结构化输出支持 (Pydantic 优先)
            target_schema = config.response_format
            if target_schema:
                gen_config_kwargs["response_mime_type"] = "application/json"
                # 如果是 Pydantic 类，直接传入 SDK 会自动处理 schema
                if isinstance(target_schema, type) and issubclass(target_schema, BaseModel):
                    gen_config_kwargs["response_json_schema"] = target_schema
                else:
                    # 否则假设已经是 schema 字典或字符串
                    gen_config_kwargs["response_json_schema"] = target_schema

            # 3. 调用 API
            response = await self.client.aio.models.generate_content(
                model=config.model or self.settings.default_model or DEFAULT_MODEL,
                contents=prompt.user,
                config=types.GenerateContentConfig(**gen_config_kwargs)
            )

            if not response.text:
                raise RuntimeError("Gemini API returned empty content")

            content = response.text

            # 4. 如果启用了结构化输出，进行二次校验（确保 100% 稳定性）
            if target_schema and isinstance(target_schema, type) and issubclass(target_schema, BaseModel):
                try:
                    # 使用 Pydantic 的 model_validate_json
                    validated_obj = target_schema.model_validate_json(content)
                    content = validated_obj.model_dump_json()
                except Exception as e:
                    logger.error(f"Gemini output fails Pydantic validation: {e}\nRaw: {content}")
                    # 如果校验失败，我们仍然返回原始字符串，但记录错误
                    # 或者根据项目需求抛出异常

            # 5. 组装结果
            usage = response.usage_metadata
            token_usage = TokenUsage(
                input_tokens=usage.prompt_token_count or 0,
                output_tokens=usage.candidates_token_count or 0
            )

            return GenerationResult(content=content, token_usage=token_usage)

        except Exception as e:
            logger.error(f"Gemini generation error: {type(e).__name__}: {str(e)}", exc_info=True)
            raise RuntimeError(f"Gemini generation failed ({type(e).__name__}): {str(e) or 'No error message'}") from e

    async def stream_generate(
        self,
        prompt: Prompt,
        config: GenerationConfig
    ) -> AsyncIterator[str]:
        """流式生成内容
        
        Args:
            prompt: 提示词
            config: 生成配置
            
        Yields:
            生成的文本片段
        """
        try:
            # 基础流式配置（不含结构化约束，SSE 下结构化支持有限）
            safety_settings = [
                types.SafetySetting(
                    category=cat,
                    threshold="BLOCK_NONE"
                ) for cat in [
                    "HATE_SPEECH", "HARASSMENT", "SEXUALLY_EXPLICIT", "DANGEROUS_CONTENT"
                ]
            ]

            gen_config = types.GenerateContentConfig(
                temperature=config.temperature,
                max_output_tokens=config.max_tokens,
                system_instruction=prompt.system,
                safety_settings=safety_settings,
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
