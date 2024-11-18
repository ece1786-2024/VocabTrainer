from agent.agent import Agent


SYSTEM_PROMPT = '''You are a vocabulary assistant in charge of generating questions for words. Your primary goal is to create a set of engaging and educational vocabulary questions based on a given list of words. The questions should be divided into three types:

1. **Multiple-choice questions**: Provide a definition, and ask the user to select the correct word from four options.
2. **Matching questions**: Present a list of 3 words and their definitions in mixed order, and ask the user to match each word to the correct definition.
3. **Short answer questions**: Ask the user to generate the meaning for each word in the list.

- Use a balanced mix of these question types, aiming for {k} questions in total.
- Ensure the multiple-choice options are challenging by including plausible distractors.
- Use concise and clear language for all questions.
- Avoid repeating the same question format consecutively for the same word.

Here is an example consisting of 6 questions:
{
    "multiple-choice": [
        {
            "word": "meticulous",
            "question": "Which word best matches the definition: 'showing great attention to detail; very careful and precise'?",
            "choices": [
                "meticulous",
                "haphazard",
                "reckless",
                "indifferent"
            ],
            "correct_option": "A"
        },
        {
            "word": "benevolent",
            "question": "Which word best matches the definition: 'well-meaning and kindly'?",
            "choices": [
                "malevolent",
                "benevolent",
                "hostile",
                "apathetic"
            ],
            "correct_option": "B"
        }
    ],
    "matching": [
        {
            "words": ["gregarious", "ambiguous"],
            "definitions": [
                "fond of company; sociable",
                "open to more than one interpretation; not having one obvious meaning"
            ]
        },
        {
            "words": ["tenacious", "ephemeral"],
            "definitions": [
                "tending to keep a firm hold of something; persistent",
                "lasting for a very short time"
            ]
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

USER_PROMPT = '''Here is a list of {n} words:

{word_list}

Generate exactly {k} questions based on this list, dividing them into:
- Multiple-choice questions where the user identifies the correct word based on a definition.
- Matching questions with 3 words and 3 definitions, where the user matches words to their definitions.
- Short answer questions where the user generates the meaning of a word.

Distribute the questions evenly across these types and ensure clarity and variety in the phrasing. Provide the final set of questions in JSON format. 
DO NOT OUTPUT ANYTHING OTHER THAN THE JSON!!!
For example, you do not need to enclose the JSON with formatting strings like "```json"
'''


class QuestionAgent(Agent):
    def __init__(self):
        super().__init__(SYSTEM_PROMPT, temperature=0.7)

    def query(self, word_list, k=20):
        return self.complete(USER_PROMPT.format(word_list=", ".join(word_list), n=len(word_list), k=k)).strip()
