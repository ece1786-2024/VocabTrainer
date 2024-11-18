from agent.agent import Agent
import json

SYSTEM_PROMPT = """
You are an AI language assistant tasked with analyzing a list of questions and user responses to determine understanding levels for each word. 
Each word may have one or more associated questions. For each word:
1. Analyze the user's responses to the corresponding question(s).
2. Assign an understanding level between 0 (no understanding) to 1 (perfect understanding).
3. If there are multiple questions for a word, calculate the average score for that word.

Return the output as a JSON object where the keys are words, and the values are their understanding levels.
"""

USER_PROMPT = """
**Word List:**
{word_list}

**Question List:**
{question_list}

For each word, calculate the average understanding level based on the responses and return a JSON object where:
- Each key is a word.
- Each value is the understanding level (0 to 1).
"""

class AnalyzerAgent(Agent):
    def __init__(self):
        super().__init__(SYSTEM_PROMPT)

    def query(self, word_list, question_list):
        word_list_str = ', '.join(word_list)
        question_list_str = '\n'.join(
            [
                f"- Word: {q['word']}, Type: {q['type']}, Correct Answer: {q['correct_answer']}, User Response: {q['user_response']}"
                for q in question_list
            ]
        )

        prompt = USER_PROMPT.format(word_list=word_list_str, question_list=question_list_str)
        
        response = self.complete(prompt)

        try:
            results = json.loads(response)
        except json.JSONDecodeError:
            raise ValueError(f"Failed to parse response as JSON: {response}")

        return results

