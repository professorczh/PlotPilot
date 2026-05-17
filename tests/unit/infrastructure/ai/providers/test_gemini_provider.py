from typing import List
from pydantic import BaseModel, Field
from infrastructure.ai.config.settings import Settings
from infrastructure.ai.providers.gemini_provider import GeminiProvider

class SubModel(BaseModel):
    name: str = Field(..., max_length=10)

class DummyPayload(BaseModel):
    title: str = Field(default="", max_length=100)
    items: List[SubModel] = Field(default_factory=list)

def test_to_gemini_schema():
    settings = Settings(api_key="dummy_key")
    provider = GeminiProvider(settings)
    
    schema = provider._to_gemini_schema(DummyPayload)
    
    # Assert output is clean and has resolved definitions/refs
    assert "$defs" not in schema
    assert "$ref" not in schema
    assert "properties" in schema
    assert "title" not in schema.get("properties", {}).get("title", {})
    
    # Assert array items sub-model is correctly inlined
    facts_schema = schema["properties"]["items"]
    assert facts_schema["type"] == "array"
    
    sub_model_schema = facts_schema["items"]
    assert sub_model_schema["type"] == "object"
    assert "name" in sub_model_schema["properties"]
    assert sub_model_schema["properties"]["name"]["maxLength"] == 10
