"""LLM 客户端包装器"""
import os
from typing import Optional, AsyncIterator
from infrastructure.ai.providers.anthropic_provider import AnthropicProvider
from infrastructure.ai.providers.openai_provider import OpenAIProvider
from infrastructure.ai.providers.mock_provider import MockProvider
from infrastructure.ai.config.settings import Settings
from domain.ai.value_objects.prompt import Prompt
from domain.ai.services.llm_service import GenerationConfig


class LLMClient:
    """LLM 客户端包装器，自动选择 Anthropic 或 Mock 提供者"""

    def __init__(self, provider=None):
        """初始化 LLM 客户端

        Args:
            provider: 可选的 LLM 提供者实例。如果未提供，将自动创建。
        """
        if provider:
            self.provider = provider
        else:
            provider_name = os.getenv("LLM_PROVIDER", "").lower()
            
            if provider_name == "openai":
                settings = Settings(
                    api_key=os.getenv("OPENAI_API_KEY", "").strip(),
                    base_url=os.getenv("OPENAI_BASE_URL", "").strip() or None,
                    default_model=os.getenv("ARK_MODEL", ""),
                )
                self.provider = OpenAIProvider(settings)
            elif provider_name == "gemini":
                from infrastructure.ai.providers.gemini_provider import GeminiProvider
                settings = Settings(
                    api_key=os.getenv("GEMINI_API_KEY", "").strip(),
                    base_url=os.getenv("GEMINI_BASE_URL", "").strip() or None,
                    default_model=os.getenv("WRITING_MODEL", ""), # 使用 Manager 注入的环境变量
                )
                self.provider = GeminiProvider(settings)
            elif provider_name == "anthropic":
                settings = Settings(
                    api_key=os.getenv("ANTHROPIC_API_KEY", "").strip(),
                    base_url=os.getenv("ANTHROPIC_BASE_URL", "").strip() or None
                )
                self.provider = AnthropicProvider(settings)
            else:
                # 兼容旧逻辑/环境变量
                self.provider = self._auto_detect_provider()

    def _auto_detect_provider(self):
        """兼容旧逻辑：根据环境变量自动探测提供商"""
        ark_key = os.getenv("ARK_API_KEY", "").strip()
        anthropic_key = (os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN") or "").strip()

        if ark_key:
            settings = Settings(
                api_key=ark_key,
                base_url=os.getenv("ARK_BASE_URL", "").strip() or None,
                default_model=os.getenv("ARK_MODEL", ""),
            )
            return OpenAIProvider(settings)
        elif anthropic_key:
            settings = Settings(
                api_key=anthropic_key,
                base_url=os.getenv("ANTHROPIC_BASE_URL", "").strip() or None
            )
            return AnthropicProvider(settings)
        return MockProvider()

    def _get_api_key(self) -> Optional[str]:
        """获取 API key"""
        raw = os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN")
        if raw is None:
            return None
        key = raw.strip()
        return key or None

    def _get_base_url(self) -> Optional[str]:
        """获取 base URL"""
        u = os.getenv("ANTHROPIC_BASE_URL")
        return u.strip() if u and u.strip() else None

    async def generate(self, prompt: str, **kwargs) -> str:
        """生成文本

        Args:
            prompt: 提示词字符串
            **kwargs: 其他参数（model, max_tokens, temperature等）

        Returns:
            生成的文本
        """
        # 创建 Prompt 对象
        prompt_obj = Prompt(
            system="你是一个专业的小说创作助手。",
            user=prompt
        )

        # 创建 GenerationConfig 对象
        config = GenerationConfig(
            model=kwargs.get("model") or os.getenv("WRITING_MODEL", ""),
            max_tokens=kwargs.get("max_tokens", 4096),
            temperature=kwargs.get("temperature", 1.0)
        )

        # 调用 provider
        result = await self.provider.generate(prompt_obj, config)
        return result.content

    async def stream_generate(
        self,
        prompt,          # Prompt 对象或 str
        config=None,
        **kwargs
    ) -> AsyncIterator[str]:
        """流式生成，代理到底层 provider"""
        # 如果是字符串，转换为 Prompt 对象
        if isinstance(prompt, str):
            prompt_obj = Prompt(
                system="你是一个专业的小说创作助手。",
                user=prompt
            )
        else:
            prompt_obj = prompt

        # 如果没有提供 config，创建默认配置
        if config is None:
            config = GenerationConfig(
                max_tokens=kwargs.get("max_tokens", 3000),
                temperature=kwargs.get("temperature", 0.85)
            )

        # 流式生成
        async for chunk in self.provider.stream_generate(prompt_obj, config):
            yield chunk
