from openai import OpenAI


class Agent():
    def __init__(self, system_prompt: str, model='gpt-4', temperature=1.0):
        self.system_prompt = system_prompt
        self.model = model
        self.temperature = temperature
        self.client = OpenAI()

    def complete(self, user_prompt: str) -> str:
        completion = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )
        return completion.choices[0].message.content
