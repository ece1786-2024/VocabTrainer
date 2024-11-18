from agent.agent import Agent
import json

SYSTEM_PROMPT = '''
You are an AI language assistant tasked with analyzing a question and the user responses to determine the user's understanding level for the words. 
For each question:
1. Analyze the user's responses to the corresponding question(s).
2. Assign an understanding level between 0 (no understanding) to 1 (perfect understanding) for each word provided.
There are three types of questions.
The first type is multiple choice question, and an example is:
{
    "word": "exception",
    "question": "Which word best matches the definition: 'a person or thing that is excluded from a general statement or does not follow a rule'?",
    "choices": [
        "entry",
        "comparison",
        "exception",
        "sign"
    ]
},
Here, the word the user is expected is "exception", and if the user selected C, then you should output 1 for this word, otherwise, output 0.

The second type is mathing quesiton, and an example is:
{
    "words": ["location", "normally", "booking"],
    "definitions": [
        "the usual or expected state of affairs",
        "a specific place or position",
        "an appointment or reservation"
    ]
}
The user will provide the answer in the format "1-B, 2-A, 3-C".
For this question, "1-B, 2-A, 3-C" is correct, so the understanding level for every word is 1 if the user provides this answer.
You should output {"location": 1, "normally": 1, "booking": 1}
If any answer is wrong, output 0 for the corresponding word.

The third type is short answer question, and an example is:
{
    "word": "exception",
    "question": "What does the word 'exception' mean?"
},
For evaluation:
    - Analyze the accuracy and completeness of the `user_answer`.
    - Assign a float between `0` and `1` based on the correctness and depth of the response.
    - Be generous when evaluating the level of understanding. For example, do not assign a score of 0 unless the user is completely off-track. 

Return the output as a map from each word to a float between 0 and 1, in JSON format. DO NOT RETURN ANYTHING ELSE.
For example, you do not need to enclose the JSON with formatting strings like "```json"
Also, you should return an understanding level for EVERY word given.
'''

USER_PROMPT = '''
**Question:**
{question}

For this question, calculate the understanding level of each word, and output a map in JSON format, where the key is the word and the value is the understanding level (a number between 0 and 1).
DO NOT OUTPUT ANYTHING OTHER THAN THE JSON!!!
For example, you do not need to enclose the JSON with formatting strings like "```json"
'''

class AnalyzerAgent(Agent):
    def __init__(self):
        super().__init__(SYSTEM_PROMPT)

    def query(self, question) -> dict:
        response_text = self.complete(USER_PROMPT.format(question=question)).strip()
        try:
            print("Query response text")
            print(response_text)
            # Parse the JSON response
            understanding_map = json.loads(response_text)
            
            # Validate that the response is a dictionary with float values
            if not isinstance(understanding_map, dict):
                raise ValueError("The response is not a JSON object.")
            
            for word, level in understanding_map.items():
                if not isinstance(word, str):
                    raise ValueError(f"Invalid key type: {word} is not a string.")
                if not (isinstance(level, float) or isinstance(level, int)):
                    raise ValueError(f"Invalid value type for '{word}': {level} is not a float.")
                if not (0 <= level <= 1):
                    raise ValueError(f"Understanding level for '{word}' is out of bounds: {level}")
            return understanding_map
        
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON response from the model: {e.msg}") from e
        except ValueError as ve:
            raise ve
        except Exception as ex:
            raise ValueError(f"An error occurred while processing the response: {str(ex)}") from ex
