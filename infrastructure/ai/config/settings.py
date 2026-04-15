"""AI 配置设置"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class Settings:
    """AI 配置设置

    管理 LLM 提供商的配置参数。
    """

    default_model: str = ""
    default_temperature: float = 0.7
    default_max_tokens: int = 4096
    api_key: Optional[str] = None
    #: 兼容自建/转发网关，与官方 ANTHROPIC_BASE_URL 一致；未设则走官方默认
    base_url: Optional[str] = None
    timeout_seconds: float = 60.0
    extra_headers: Optional[dict] = None
    extra_query: Optional[dict] = None
    extra_body: Optional[dict] = None
    provider_name: str = "custom"
    protocol: str = "openai"

    def __post_init__(self):
        """验证配置参数"""
        if self.default_temperature is not None and not (0.0 <= self.default_temperature <= 2.0):
            raise ValueError("Temperature must be between 0.0 and 2.0")

        if self.default_max_tokens is not None and self.default_max_tokens <= 0:
            raise ValueError("Max tokens must be positive")
