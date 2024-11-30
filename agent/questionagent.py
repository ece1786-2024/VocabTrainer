from agent.agent import Agent
import json
import random


SYSTEM_PROMPT = SYSTEM_PROMPT = '''
You are a vocabulary assistant in charge of generating questions for words. Your primary goal is to create a set of engaging and educational vocabulary questions based on a given list of words. The questions should be divided into three types:

1. **Multiple-choice questions**:
   - Provide a definition and ask the user to select the correct word from four options.
   - Randomize the position of the correct answer among the choices.
   - Ensure distractors are plausible and related in meaning or form to the correct word.

2. **Matching questions**:
   - Present a list of 3 words and their definitions in mixed order.
   - Ask the user to match each word to the correct definition.
   - Provide the correct matches using a mapping, e.g., `"correct_matches": {"1": "C", "2": "B", "3": "A"}`.

3. **Short answer questions**:
   - Ask the user to generate the meaning for each word in the list.

- Use a balanced mix of these question types, aiming for exactly {k} questions in total. The value of {k} will be provided.
- Ensure all provided words are used without repetition.
- Use concise and clear language appropriate for the target audience's proficiency level.
- Avoid repeating the same question format consecutively for the same word.
- Output the questions in valid JSON format as shown in the example.
- Do not include any additional text or explanations beyond the JSON output.

Here is an example consisting of 6 questions:
{
    "multiple-choice": [
        {
            "word": "meticulous",
            "question": "Which word best matches the definition: 'showing great attention to detail; very careful and precise'?",
            "choices": [
                "indifferent",
                "meticulous",
                "reckless",
                "haphazard"
            ],
            "correct_answer": "B"
        },
        {
            "word": "benevolent",
            "question": "Which word best matches the definition: 'well-meaning and kindly'?",
            "choices": [
                "apathetic",
                "malevolent",
                "benevolent",
                "hostile"
            ],
            "correct_answer": "C"
        }
    ],
    "matching": [
        {
            "words": ["enter", "except", "register"],
            "definitions": [
                "come or go into a place",
                "not including; other than",
                "sign up or record for an activity or for use"
            ],
            "correct_matches": {
                "1": "A",
                "2": "B",
                "3": "C"
            }
        }
    ],
    "short-answer": [
        {
            "word": "obsolete",
            "question": "What does the word 'obsolete' mean?"
        },
        {
            "word": "pragmatic",
            "question": "What does the word 'pragmatic' mean?"
        }
    ]
}
'''

USER_PROMPT = '''
Here is a list of {n} words:

{word_list}

Using all the words from the list without repetition, generate exactly {k} questions divided into:
- **Multiple-choice questions** where the user identifies the correct word based on a definition. Include four options per question, randomizing the position of the correct answer. Ensure distractors are plausible and related in meaning or form to the correct word.
- **Matching questions** with 3 words and 3 definitions, where the user matches words to their definitions. Randomize the order of both words and definitions.
- **Short answer questions** where the user generates the meaning of a word.

**Instructions:**
- Distribute the questions evenly across these types.
- Ensure clarity and variety in the phrasing of questions.
- Generate EXACTLY {k} questions, covering each of the {n} words evenly.
- Provide the final set of questions in valid JSON format, following the structure shown in the example from the system prompt.
- **DO NOT OUTPUT ANYTHING OTHER THAN THE JSON!!!**

For example, do not enclose the JSON with formatting strings like "```json".
'''


class QuestionAgent(Agent):
    def __init__(self):
        super().__init__(SYSTEM_PROMPT, temperature=0.7)

    def query(self, word_list, num_questions=20):
        questions = json.loads(self.complete(USER_PROMPT.format(word_list=", ".join(word_list), n=len(word_list), k=num_questions)).strip())
        for i in range(len(questions['matching'])):
            data = questions['matching'][i]
            indices = list(range(len(data["words"])))
            random.shuffle(indices)
            shuffled_words = [data["words"][i] for i in indices]
            shuffled_matches = {str(indices.index(int(k) - 1) + 1): v for k, v in data["correct_matches"].items()}
            questions['matching'][i] = {
                    "words": shuffled_words,
                    "definitions": data["definitions"],
                    "correct_matches": shuffled_matches
            }
        return questions