import asyncio

from openai import OpenAI


class LMClient:
    def __init__(
        self,
        model: str,
        system_prompt: str = "",
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "none",
    ):
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.system_prompt = system_prompt
        self.messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": self.system_prompt}],
            }
        ]

    def reset(self) -> None:
        self.messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": self.system_prompt}],
            }
        ]

    async def query(
        self, text: str, max_tokens: int = 256, temperature: float = 0.7, top_p: float = 0.95
    ) -> str:
        self.messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                ],
            }
        )

        response = await asyncio.to_thread(
            self.client.chat.completions.create,
            model=self.model,
            messages=self.messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=42,
        )

        content = response.choices[0].message.content
        self.messages.append(
            {"role": "assistant", "content": [{"type": "text", "text": content}]}
        )

        return content
