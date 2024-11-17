from agent.agent import Agent
from typing import List


SYSTEM_PROMPT = '''You are an AI language assistant tasked with recommending {k} words for me to memorize today from the following vocabulary table. Each entry in the table includes:

- **Word**
- **CEFR Level**: One of A1, A2, B1, B2, C1, or C2 (higher levels are more advanced words).
- **Memorization Level**: A real number between 0 and 1, where 0 means I did not understand the word at all, and 1 means I have a good memory of the word'''

USER_PROMPT = '''**Instructions:**

1. **Balance CEFR Levels:**
   - Select words that are not too easy or too hard. Aim for a mix of words primarily from A2 to B2 levels.

2. **Consider Memorization Levels:**
   - Prioritize words with lower memorization levels (closer to 0), as these are words I need to focus on learning.
   - Include a few words with higher memorization levels (closer to 1) for review purposes.

3. **Overall Balance:**
   - Ensure the list of {k} words is varied and balanced in terms of difficulty and familiarity.

4. **Output Format:**
   - Provide the list of {k} words along with their CEFR level and memorization level in a clear and organized manner.

**Vocabulary Table:**

{vocab_table}

**Your Task:**

Using the above instructions and vocabulary table, generate a list of {k} words for me to memorize today. Format it as one line per word and DO NOT OUTPUT ANYTHING OTHER THAN THE WORDS.'''


class RankingAgent(Agent):
    def __init__(self):
        super().__init__(SYSTEM_PROMPT)

    def query(self, vocab_table: List[tuple[str, str, float]], k=20) -> List[str]:
        vocab_table_str = '\n'.join([f'{word}, {cefr}, {mem}' for word, cefr, mem in vocab_table])
        complete = self.complete(USER_PROMPT.format(k=k, vocab_table=vocab_table_str))
        return [word.strip() for word in complete.splitlines() if word.strip()]
