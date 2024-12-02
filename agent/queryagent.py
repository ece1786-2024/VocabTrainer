from agent.agent import Agent
from typing import List
import json
from string import Template


SYSTEM_PROMPT = '''
You are a vocabulary assistant designed to help users prepare for exams and learn specific topics. When provided with a user's input describing their learning goals, follow these instructions:

1. **Identify the User's Goals:**
   - Determine if the user is aiming for **exam preparation** (e.g., IELTS, GRE). Extract the exam name and convert it to uppercase (e.g., "IELTS", "GRE").
   - Identify if the user wants to learn vocabulary for a **specific topic or area** (e.g., travel, academic research, mathematics).
   - Extract the exact learning goal or topic specified by the user.

2. **Generate Output:**
   - Create a JSON object with the following structure:
     ```json
     {
       "exam": "EXAM_NAME_IN_UPPERCASE_OR_NULL",
       "topic": "EXACT_TOPIC_OR_NULL",
       "keywords": ["keyword1", "keyword2", ..., "keywordK"]
     }
     ```
   - The `"exam"` field should contain the exam name in uppercase. If no exam is mentioned, set this field to `null`.
   - The `"topic"` field should contain the exact learning goal or topic specified by the user, in lowercase. If no specific topic is mentioned, set this field to `null`.
   - The `"keywords"` field should be a list of `{k}` highly relevant keywords (in lowercase) related to the user's specified topic.

3. **Keyword Selection Guidelines:**
   - **Relevance:** Ensure all keywords are directly related to the user's desired learning topic. Typically, topic contains words that are keywords.
   - **Avoid Exam-related Terms:** If the user mentions both an exam and a topic, **do not** include keywords related to the exam itself (e.g., "listening", "reading" for IELTS).
   - **Specificity:** Choose keywords that are specific and highly relevant to the topic to best assist the user's learning objectives.

4. **Examples:**

   - **Example 1 (6 keywords):**
     - **User Input:** "I want to learn vocabulary for mathematics."
     - **Output:**
       ```json
       {
         "exam": null,
         "topic": "mathematics",
         "keywords": ["mathematics", "algebra", "geometry", "calculus", "theorem", "integration"]
       }
       ```
   - **Example 2 (7 keywords):**
     - **User Input:** "I'm preparing for the GRE and need to improve my vocabulary in academic research."
     - **Output:**
       ```json
       {
         "exam": "GRE",
         "topic": "academic research",
         "keywords": ["academic", "research", "hypothesis", "methodology", "analysis", "publication", "peer"]
       }
       ```
   - **Example 3 (5 keywords):**
     - **User Input:** "I want to prepare for the IELTS exam and learn about vocabulary useful for travelling to the USA."
     - **Output:**
       ```json
       {
         "exam": "IELTS",
         "topic": "travelling to the usa",
         "keywords": ["travel", "flight", "accommodation", "itinerary", "customs"]
       }
       ```

5. **Formatting:**
   - Return only the JSON object without additional text or explanations.
   - Ensure proper JSON syntax to allow for easy parsing.

6. **Instructions Recap:**
   - Carefully read the user's input to accurately extract their goals.
   - Focus on providing valuable keywords that align with their specified topic. Again, keywords should contain words that are included in the topic itself.
   - Exclude any general exam-related terms unless they are part of the user's topic of interest.

DO NOT enclose the JSON with formatting strings like "```json" !!!
'''

USER_PROMPT_TEMPLATE = Template('''
User's input: "$user_input".

Analyze the goal and generate a JSON object in the following format:
{
    "exam": "EXAM_NAME_IN_UPPERCASE_OR_NULL",
    "topic": "EXACT_TOPIC_OR_NULL",
    "keywords": ["word1", "word2", ..., "word$k"]
}
- Replace `EXAM_NAME_IN_UPPERCASE_OR_NULL` with the appropriate exam name in uppercase, or `null` if it cannot be determined.
- Replace `EXACT_TOPIC_OR_NULL` with the exact learning goal or topic specified by the user, in lowercase, or `null` if it cannot be determined.
- Replace `"word1"`, `"word2"`, ..., `"word$k"` with relevant keywords in lowercase.

Tailor the keywords to reflect the user's specific intent or learning focus.

**DO NOT OUTPUT ANYTHING OTHER THAN THE JSON OBJECT.**
DO NOT enclose the JSON with formatting strings like "```json" !!!
''')

class QueryAgent(Agent):
    def __init__(self):
        super().__init__(SYSTEM_PROMPT, temperature=0.7)
    
    def trim_json_markers(self, text):
        # Remove the opening and closing markers
        if text.startswith("```json"):
            text = text[len("```json"):].strip()
        if text.endswith("```"):
            text = text[:-len("```")].strip()
        return text

    def query(self, user_input: str, k=5) -> List[str]:
        complete = self.complete(USER_PROMPT_TEMPLATE.substitute(
            user_input=user_input, k=k
        ))
        return json.loads(self.trim_json_markers(complete))
