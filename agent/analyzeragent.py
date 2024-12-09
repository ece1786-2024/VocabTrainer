from agent.agent import Agent
import json

SYSTEM_PROMPT = '''
You are an AI language assistant tasked with analyzing questions and user responses to determine the user's understanding level for each word.
For each question:
1. **Analyze** the user's response in relation to the corresponding question.
2. **Assign** an understanding level between **0** (no understanding) and **1** (perfect understanding) for each provided word.

There are **four types of questions**, each with specific evaluation criteria:

1. **Multiple-choice questions**:
   - Example:
   {
       "word": "exception",
       "question": "Which word best matches the definition: 'a person or thing that is excluded from a general statement or does not follow a rule'?",
       "choices": [
           "entry",
           "comparison",
           "exception",
           "sign"
       ]
   }
   - If the user selects the correct answer ("exception"), assign `1` for this word; otherwise, assign `0`.

2. **Matching questions**:
   - Example:
   {
       "words": ["location", "normally", "booking"],
       "definitions": [
           "the usual or expected state of affairs",
           "a specific place or position",
           "an appointment or reservation"
       ]
   }
   - The user will provide the answer in the format "1-B, 2-C, 3-A".
   - For this question, "1-B, 2-A, 3-C" is correct. You should assign:
     {"location": 1, "normally": 0, "booking": 0}.

3. **Short answer questions**:
   - Example:
   {
       "word": "exception",
       "question": "What does the word 'exception' mean?"
   }
   - Analyze the accuracy and completeness of the `user_answer`.
   - Assign a float between `0` and `1` based on the correctness and depth of the response.
   - Be generous when evaluating the level of understanding. For example, do not assign a score of `0` unless the user is entirely off track. If the user conveys the correct meaning but is too brief, you may still assign full marks.

4. **Scenario-based questions**:
   - Example:
   {
       "word": "apple",
       "question": "I ate an _.",
       "choices": [
           "apple",
           "car",
           "charger",
           "crate"
       ]
   }
   - If the user selects the correct answer ("apple"), assign `1` for this word; otherwise, assign `0`.

**Evaluation Notes**:
- Each word provided in the question must have a corresponding score in the output.
- Return the output as a map from each word to a float between `0` and `1`, in JSON format.
- DO NOT RETURN ANYTHING ELSE.
- For example, do not enclose the JSON with formatting strings like "```json".

Example Output:
{
    "exception": 1,
    "location": 1,
    "normally": 0,
    "booking": 0,
    "apple": 1
}
'''

USER_PROMPT = '''
**Question:**
{question}

**User Answer:**
{user_answer}

For this question, calculate the understanding level of each word, and output a map in JSON format, where the key is the word and the value is the understanding level (a number between 0 and 1).

DO NOT OUTPUT ANYTHING OTHER THAN THE JSON!!!
For example, you do not need to enclose the JSON with formatting strings like "```json".
'''

class AnalyzerAgent(Agent):
    def __init__(self):
        super().__init__(SYSTEM_PROMPT)

    def query(self, question, user_answer) -> dict:
        response_text = self.complete(USER_PROMPT.format(question=question, user_answer=user_answer)).strip()
        try:
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
