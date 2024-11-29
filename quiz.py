import json
import random

class Quiz:
    def __init__(self, questions_json):
        self.data = json.loads(questions_json)
        self.questions_list = self._prepare_questions()
        self.current_index = 0
    
    def _prepare_questions(self):
        questions_list = []
        for q in self.data.get('multiple-choice', []):
            questions_list.append({'type': 'multiple-choice', 'content': q})
        for q in self.data.get('matching', []):
            questions_list.append({'type': 'matching', 'content': q})
        for q in self.data.get('short-answer', []):
            questions_list.append({'type': 'short-answer', 'content': q})
        
        # Shuffle the questions
        random.shuffle(questions_list)
        return questions_list

    def run_quiz(self):
        if self.current_index >= len(self.questions_list):
            print("Thank you for completing today's training.")
            return None

        item = self.questions_list[self.current_index]
        self.current_index += 1  # Increment the index for the next call

        q_type = item['type']
        content = item['content']
        print(f"Question {self.current_index}:")
        user_answer = None

        if q_type == 'multiple-choice':
            # Present multiple-choice question
            print(content['question'])
            choices = content['choices']
            options = ['A', 'B', 'C', 'D']
            for letter, choice in zip(options, choices):
                print(f"  {letter}. {choice}")
            user_answer = input("Your answer (A/B/C/D): ")

        elif q_type == 'matching':
            # Present matching question
            words = content['words']
            definitions = content['definitions']

            print("Match the following words to their correct definitions:")
            print("Words:")
            for i, word in enumerate(words, 1):
                print(f"  {i}. {word}")

            print("Definitions:")
            letters = [chr(65 + i) for i in range(len(definitions))]  # Generate letters A, B, C...
            for letter, definition in zip(letters, definitions):
                print(f"  {letter}. {definition}")
            print("Enter your answers in the format '1-A, 2-B, 3-C'")
            user_answer = input("Your answers: ")
        elif q_type == 'short-answer':
            # Present short-answer question
            print(content['question'])
            user_answer = input("Your answer: ")
        else:
            print("Unknown question type.")

        print()  # Add a newline for better readability
        return item, user_answer
