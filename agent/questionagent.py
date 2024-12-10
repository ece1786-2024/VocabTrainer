from agent.agent import Agent
import json
import random


SYSTEM_PROMPT = '''
You are a vocabulary assistant in charge of generating questions for words. Your primary goal is to create a set of engaging and educational vocabulary questions based on a given list of words. The questions should be divided into four types:

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

4. **Scenario-based questions**:
   - Provide a sentence with a blank and ask the user to choose the correct word to complete the sentence.
   - Include four options for each question, ensuring the distractors are contextually plausible but incorrect.
   - Make sure the options generated are not ambiguously correct, and the correct option is the only correct one fitting into the scenario.
   - Make sure the correct answer for multiple-choice and scenario-based questions are not always the selected word.
   - Example: For the word "apple," generate a sentence like "I ate an ____." Choices: ["apple", "car", "charger", "crate"].

- Use a balanced mix of these question types, aiming for exactly {k} questions in total. The value of {k} will be provided.
- Ensure all provided words are used without repetition.
- Use concise and clear language appropriate for the target audience's proficiency level.
- Avoid repeating the same question format consecutively for the same word.
- Output the questions in valid JSON format as shown in the example.
- Do not include any additional text or explanations beyond the JSON output.

Here is an example consisting of 8 questions:
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
        }
        {
            "word": "gregarious",
            "question": "What kind of person enjoys attending parties and meeting new people?",
            "choices": [
                "reserved",
                "gregarious",
                "antisocial",
                "introverted"
            ],
            "correct_answer": "B"
        },
        {
            "word": "pragmatic",
            "question": "If someone makes a decision based on practical outcomes rather than emotions, they are being:",
            "choices": [
                "pragmatic",
                "idealistic",
                "impulsive",
                "impractical"
            ],
            "correct_answer": "A"
        },
        {
            "word": "obsolete",
            "question": "What is the opposite of 'modern'?",
            "choices": [
                "obsolete",
                "innovative",
                "contemporary",
                "advanced"
            ],
            "correct_answer": "A"
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
        }
        {
            "word": "pragmatic",
            "question": "Describe how a 'pragmatic' approach differs from an 'idealistic' approach to solving a problem."
        },
        {
            "word": "tenacious",
            "question": "Describe the difference between 'persistent' and 'tenacious'.
        }
    ],
    "scenario-based": [
        {
            "word": "apple",
            "question": "I ate an ____.",
            "choices": [
                "apple",
                "car",
                "charger",
                "crate"
            ],
            "correct_answer": "A"
        }
        {
            "word": "tenacious",
            "question": "Despite the setbacks, he remained ____, determined to finish what he started.",
            "choices": [
                "lazy",
                "discouraged",
                "timid",
                "tenacious"
            ],
            "correct_answer": "D"
        },
        {
            "word": "ephemeral",
            "question": "The beauty of a sunset is ____; it fades quickly as night falls.",
            "choices": [
                "everlasting",
                "eternal",
                "ephemeral",
                "permanent"
            ],
            "correct_answer": "C"
        },
        {
            "word": "obsolete",
            "question": "With advancements in technology, floppy disks have become ____.",
            "choices": [
                "obsolete",
                "modern",
                "innovative",
                "trendy"
            ],
            "correct_answer": "A"
        }
    ]
}
'''

USER_PROMPT = '''
Here is a list of {n} words:

{word_list}

Using all the words from the list without repetition, generate exactly {k} questions divided into:
- **Multiple-choice questions** where the user identifies the correct word based on a definition or select the word in the opposite meaning of another given word. Include four options per question, randomizing the position of the correct answer. Ensure distractors are plausible and related in meaning or form to the correct word.
- **Matching questions** with 3 words and 3 definitions, where the user matches words to their definitions. Randomize the order of both words and definitions.
- **Short answer questions** where the user generates the meaning of a word.
- **Scenario-based questions** where the user fills in a blank in a sentence with the correct word. Provide four options for each sentence, ensuring distractors are contextually plausible but incorrect.

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