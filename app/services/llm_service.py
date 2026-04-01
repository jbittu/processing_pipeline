from langchain_nvidia_ai_endpoints import ChatNVIDIA
from app.config import get_settings


def get_llm(temperature: float = 0.0) -> ChatNVIDIA:
    """Get a configured ChatNVIDIA instance pointing to NVIDIA NIM.

    Uses the free API at build.nvidia.com. No cost.
    Supports vision/image input out of the box.
    """
    settings = get_settings()
    return ChatNVIDIA(
        model=settings.nvidia_model,
        api_key=settings.nvidia_api_key,
        temperature=temperature,
        max_tokens=4096,
    )
