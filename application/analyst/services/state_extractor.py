import logging
import os
from domain.ai.services.llm_service import LLMService, GenerationConfig
from domain.ai.value_objects.prompt import Prompt
from domain.novel.value_objects.chapter_state import ChapterState
from application.ai.chapter_state_llm_contract import (
    build_chapter_state_extraction_system_prompt,
    chapter_state_payload_to_domain,
    empty_chapter_state,
    parse_chapter_state_llm_response,
    ChapterStateLlmPayload,
)
from application.ai.structured_json_pipeline import structured_json_generate

logger = logging.getLogger(__name__)


class StateExtractor:
    """状态提取应用服务

    使用 LLM 从章节内容中提取结构化信息
    """

    def __init__(self, llm_service: LLMService):
        self.llm_service = llm_service

    async def extract_chapter_state(self, content: str) -> ChapterState:
        """从章节内容中提取状态

        Args:
            content: 章节内容

        Returns:
            提取的章节状态
        """
        logger.info(f"StateExtractor.extract_chapter_state: content_length={len(content)}")

        # 构建提取提示词
        system_prompt, user_prompt = self._build_extraction_prompt(content)
        prompt = Prompt(system=system_prompt, user=user_prompt)

        # 配置 LLM 并强制指定 JSON schema
        config = GenerationConfig(
            model=os.getenv("WRITING_MODEL", ""),
            max_tokens=4096,
            temperature=0.3,
            response_format=ChapterStateLlmPayload,
        )

        payload = await structured_json_generate(
            llm=self.llm_service,
            prompt=prompt,
            config=config,
            schema_model=ChapterStateLlmPayload,
        )

        if payload is None:
            logger.warning(
                "StateExtractor: LLM 输出结构化生成失败，返回安全空值。"
            )
            chapter_state = empty_chapter_state()
        else:
            chapter_state = chapter_state_payload_to_domain(payload)
        logger.info(
            f"StateExtractor result: "
            f"new_characters={len(chapter_state.new_characters)}, "
            f"character_actions={len(chapter_state.character_actions)}, "
            f"relationship_changes={len(chapter_state.relationship_changes)}, "
            f"foreshadowing_planted={len(chapter_state.foreshadowing_planted)}, "
            f"foreshadowing_resolved={len(chapter_state.foreshadowing_resolved)}, "
            f"events={len(chapter_state.events)}"
        )
        return chapter_state

    def _build_extraction_prompt(self, content: str) -> tuple[str, str]:
        """构建提取提示词

        Args:
            content: 章节内容

        Returns:
            (system_prompt, user_prompt) 元组
        """
        system_prompt = build_chapter_state_extraction_system_prompt()

        user_prompt = f"""请从以下章节内容中提取结构化信息：

{content}"""

        return system_prompt, user_prompt
