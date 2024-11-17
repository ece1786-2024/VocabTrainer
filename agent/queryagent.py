from agent.agent import Agent


SYSTEM_PROMPT = '''You are a vocabulary assistant. Given a user's input, identify whether their goal is exam preparation (e.g., IELTS, GRE) or learning a specific area (e.g., travel, academic research). Generate a list of 10 relevant words for their context. Include each word's difficulty level (A1 to C2).'''

USER_PROMPT = '''User's input: "{user_input}". Analyze the goal and generate a list of words with difficulty levels.'''


class QueryAgent(Agent):
    def __init__(self):
        super().__init__(SYSTEM_PROMPT, temperature=0.7)

    def query(self, user_input):
        return self.complete(USER_PROMPT.format(user_input=user_input)).strip()
