from agent.baselineagent import BaseLineAgent
from vectordb import VectorDB
from quiz import Quiz
import json


class BaseLine:
    def __init__(self):
        self.agent = BaseLineAgent()
        self.db = VectorDB()

    def run(self):
        user_input = input("Enter your learning goal: ").strip()
        vocab = self.db.query_all()
        vocab_table = []
        for row in vocab:
            vocab_table.append((row['word'], row['CEFR'], row['understanding_rating'], row['IELTS'], row['GRE']))
        result = self.agent.query(vocab_table, user_input)
        print('Selected words:', result['words'])
        quiz = Quiz(questions_json=json.dumps(result['questions']))
        while True:
            run_result = quiz.run_quiz()
            if run_result is None:
                # No more questions available
                break


if __name__ == '__main__':
    trainer = BaseLine()
    trainer.run()
