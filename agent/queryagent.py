from agent.agent import Agent
from typing import List
import json
from string import Template


SYSTEM_PROMPT = '''You are a vocabulary assistant. Given a user's input, identify whether their goal is exam preparation (e.g., IELTS, GRE) or learning a specific area (e.g., travel, academic research, mathematics). Based on the user's intent, generate a JSON object containing the exam name (all uppercase) and a list of {k} relevant keywords (all lowercase). Ensure the keywords are highly relevant to the user's desired learning area or exam, incorporating terms that align closely with their focus. For example, if the user expresses interest in learning mathematics, include keywords like "math", "theory", "integration", "calculus", etc.'''

USER_PROMPT_TEMPLATE = Template('''User's input: "$user_input". Analyze the goal and generate a JSON object in the following format: 
{
    "exam": "EXAM_NAME",
    "keywords": ["word1", "word2", ..., "word$k"]
}
Replace EXAM_NAME with the appropriate exam name in uppercase, or empty string if it cannot be determined. Replace "word1", "word2", ..., "word$k" with the relevant keywords in lowercase. Tailor the keywords to reflect the user's specific intent or learning focus. DO NOT OUTPUT ANYTHING OTHER THAN THE JSON OBJECT.''')




class QueryAgent(Agent):
    def __init__(self):
        super().__init__(SYSTEM_PROMPT, temperature=0.7)

    def query(self, user_input: str, k=5) -> List[str]:
        complete = self.complete(USER_PROMPT_TEMPLATE.substitute(
            user_input=user_input, k=k
        ))
        return json.loads(complete)
