from agent.agent import Agent
from typing import List, Tuple
from string import Template
import json


SYSTEM_PROMPT = '''You are a vocabulary quiz generator in a word memorization software. Your task is to intelligently create a word list and generate quizzes to improve the user's efficiency in learning new vocabulary based on their requirements and historical performance data.'''

USER_PROMPT_TEMPLATE = Template('''The table below contains five columns:  
1. **Word**: the vocabulary word,  
2. **CEFR Level**: the word's difficulty rating based on the CEFR scale,  
3. **User Memory Score**: a score from 0 to 1, where a higher score indicates better retention,  
4. **IELTS Vocabulary**: whether the word belongs to IELTS vocabulary (True/False),  
5. **GRE Vocabulary**: whether the word belongs to GRE vocabulary (True/False).

$vocab_table

Now, based on the user's requirements provided below, select $n words that match the conditions and ensure they have moderate difficulty and low memory scores:  

$user_input

Once you have selected these words, generate exactly $k questions based on the selected word list. The questions should be of the following types:  

1. **Multiple-choice questions**: Provide a definition, and the user selects the correct word from four options.  
2. **Matching questions**: Include three words and three definitions, where the user matches each word to its correct definition.  
3. **Short-answer questions**: Ask the user to provide the meaning of a given word.  

Your final output (including the selected words and the questions) must be in a complete JSON Object. Below is an example of the desired output format:
{
    "words": ["meticulous", "benevolent", "gregarious", "ambiguous", "tenacious", "ephemeral", "obsolete", "pragmatic"],
    "questions": {
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
}

Remember you need to select $n questions and use them to generate $k questions!

Do not output anything other than the JSON Object!!! Also, do not wrap the JSON with a "```json" code block.''')

class BaseLineAgent(Agent):
    def __init__(self):
        super().__init__(SYSTEM_PROMPT, model='gpt-4o-mini')

    def query(self, vocab_table: List[Tuple[str, str, float, bool, bool]], user_input: str, n=10, k=20):
        vocab_table_str = '\n'.join([f'{word}, {cefr}, {mem}, {ielts}, {gre}' for word, cefr, mem, ielts, gre in vocab_table])
        prompt = USER_PROMPT_TEMPLATE.substitute(
            vocab_table=vocab_table_str, n=n, k=k, user_input=user_input
        )
        complete = self.complete(prompt)
        return json.loads(complete)
