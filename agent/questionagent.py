from agent.agent import Agent


SYSTEM_PROMPT = '''You are a vocabulary assistant in charge of generating questions for words. Your primary goal is to create a set of engaging and educational vocabulary questions based on a given list of words. The questions should be divided into three types:

1. **Multiple-choice questions**: Provide a definition, and ask the user to select the correct word from four options.
2. **Matching questions**: Present a list of words and their definitions in mixed order, and ask the user to match each word to the correct definition.
3. **Short answer questions**: Ask the user to generate the meaning for each word in the list.

- Use a balanced mix of these question types, aiming for {k} questions in total.
- Ensure the multiple-choice options are challenging by including plausible distractors.
- Use concise and clear language for all questions.
- Avoid repeating the same question format consecutively for the same word.'''

USER_PROMPT = '''Here is a list of {n} words:

{word_list}

Generate exactly {k} questions based on this list, dividing them into:
- Multiple-choice questions where the user identifies the correct word based on a definition.
- Matching questions where the user matches words to their definitions.
- Short answer questions where the user generates the meaning of a word.

Distribute the questions evenly across these types and ensure clarity and variety in the phrasing. Provide the final set of questions in JSON format. DO NOT OUTPUT ANYTHING OTHER THAN THE JSON!!!'''


class QuestionAgent(Agent):
    def __init__(self):
        super().__init__(SYSTEM_PROMPT, temperature=0.7)

    def query(self, word_list, k=20):
        return self.complete(USER_PROMPT.format(word_list=", ".join(word_list), n=len(word_list), k=k)).strip()
