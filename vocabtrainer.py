from agent.analyzeragent import AnalyzerAgent
from agent.queryagent import QueryAgent
from agent.rankingagent import RankingAgent
from agent.questionagent import QuestionAgent
from embedding.glove import GloveEmbedding
from quiz import Quiz
from vectordb import VectorDB
import numpy as np


class VocabTrainer:
    def __init__(self):
        self.query_agent = QueryAgent()
        self.ranking_agent = RankingAgent()
        self.question_agent = QuestionAgent()
        self.embedding = GloveEmbedding()
        self.db = VectorDB()

    def run(self):
        print("Welcome to the Query Agent!")
        print("Describe your learning goal in a few sentences.")
        print("For example: 'I want to prepare for the IELTS exam' or 'I need travel vocabulary.'")

        user_input = input("\nEnter your learning goal: ").strip()
        query_words = self.query_agent.query(user_input)
        print('query_words:', query_words)
        query_vector = np.zeros(self.embedding.dim)
        query_actual_len = 0
        for word in query_words:
            if self.embedding.contains(word):
                query_actual_len += 1
                query_vector += self.embedding.encode(word)
        query_vector /= query_actual_len
        
        candidate_table = self.db.query_by_similarity(query_vector, n_results=100)
        candidate_vocab = []
        for row in candidate_table:
            candidate_vocab.append((row['word'], row['CEFR'], row['understanding_rating']))
        selected_words = self.ranking_agent.query(candidate_vocab)
        print('selected_words:', selected_words)
        
        print("Generating questions...")
        questions_json = self.question_agent.query(selected_words)

        #TODO: use GUI to prompt questions
        quiz = Quiz(questions_json=questions_json)
        while True:
            question, user_answer = quiz.run_quiz()
            if user_answer is None:
                # No more questions available
                break
            else:
                # Use the analyzer to evaluate the respnse
                analyzer = AnalyzerAgent()
                understanding_map = analyzer.query(question, user_answer)
                print(understanding_map)

        print("Thank you for completing today's training.")


if __name__ == '__main__':
    trainer = VocabTrainer()
    trainer.run()
