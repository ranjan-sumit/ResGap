"""
LLM Client — NVIDIA NIM, Master Model: openai/gpt-oss-120b
Used for all 7 main pipeline stages.
LLM Council uses separate per-agent models (see llm_council.py).

All models share the same NVIDIA base URL and API key.
Streaming is used throughout — handles reasoning_content from thinking models.
"""
import base64
from openai import OpenAI

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
MASTER_MODEL    = "openai/gpt-oss-120b"


def _collect_stream(stream) -> str:
    """
    Collect streaming response from NVIDIA NIM.
    Skips reasoning_content (chain-of-thought) and returns only final content.
    """
    parts = []
    for chunk in stream:
        if not getattr(chunk, "choices", None):
            continue
        delta = chunk.choices[0].delta
        content = getattr(delta, "content", None)
        if content:
            parts.append(content)
    return "".join(parts).strip()


def make_nvidia_client(api_key: str) -> OpenAI:
    return OpenAI(base_url=NVIDIA_BASE_URL, api_key=api_key)


class AzureOpenAIClient:
    """
    Master pipeline client using openai/gpt-oss-120b via NVIDIA NIM.
    Name kept for backward compatibility — all pipeline imports unchanged.
    """

    def __init__(self, api_key: str, **kwargs):
        self.client  = make_nvidia_client(api_key)
        self.api_key = api_key
        self.model   = MASTER_MODEL

    def chat(self, system: str, user: str,
             temperature: float = 0.3, max_tokens: int = 4096,
             _retry: int = 0) -> str:
        """Streaming chat with timeout and one automatic retry on failure.
        B10 fix: previously no timeout — one hung call blocked Streamlit indefinitely."""
        import time
        MAX_RETRIES = 2
        TIMEOUT_SECS = 120   # 2 minutes max per call

        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                temperature=temperature,
                top_p=1,
                max_tokens=max_tokens,
                stream=True,
                timeout=TIMEOUT_SECS,
            )
            return _collect_stream(stream)
        except Exception as e:
            err = str(e)
            if _retry < MAX_RETRIES and ("timeout" in err.lower() or "rate" in err.lower() or "429" in err):
                wait = 2 ** _retry   # exponential back-off: 1s, 2s
                time.sleep(wait)
                return self.chat(system, user, temperature, max_tokens, _retry=_retry + 1)
            # Non-retryable or exhausted retries — return empty so callers
            # hit their own fallback logic rather than crashing
            return f"[API error after {_retry+1} attempt(s): {err[:200]}]"

    def chat_json(self, system: str, user: str, max_tokens: int = 4096) -> str:
        """Chat that instructs the model to return only valid JSON."""
        system_json = (
            system
            + "\n\nCRITICAL: Your final response must be ONLY valid JSON. "
            "No markdown fences, no preamble, no explanation. "
            "Start directly with { or [ and end with } or ]."
        )
        return self.chat(system_json, user, temperature=0.1, max_tokens=max_tokens)

    def vision(self, system: str, user_text: str,
               image_bytes: bytes, max_tokens: int = 2048) -> str:
        """Vision call — gpt-oss-120b supports image input."""
        b64 = base64.b64encode(image_bytes).decode("utf-8")
        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text",      "text": user_text},
                            {"type": "image_url", "image_url": {
                                "url": f"data:image/png;base64,{b64}"
                            }},
                        ],
                    },
                ],
                temperature=0.1,
                top_p=1,
                max_tokens=max_tokens,
                stream=True,
            )
            return _collect_stream(stream)
        except Exception as e:
            return f"[Vision unavailable: {e}]"
