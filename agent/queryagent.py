from agent.agent import Agent
from typing import List

SYSTEM_PROMPT = '''You are a vocabulary assistant. Given a user's input, identify whether their goal is exam preparation (e.g., IELTS, GRE) or learning a specific area (e.g., travel, academic research). Generate a list of {k} relevant words for their context.'''

USER_PROMPT = '''User's input: "{user_input}". Analyze the goal and generate a list of {k} words. Format it as one line per word (all lowercase, without any other characters, JUST PURE WORDS) and DO NOT OUTPUT ANYTHING OTHER THAN THE WORDS.'''


class QueryAgent(Agent):
    def __init__(self):
        super().__init__(SYSTEM_PROMPT, temperature=0.7)

    def query(self, user_input: str, k=5) -> List[str]:
        complete = self.complete(USER_PROMPT.format(user_input=user_input, k=k))
        return [word.strip() for word in complete.splitlines() if word.strip()]
