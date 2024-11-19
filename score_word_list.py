from typing import List, Optional
from vectordb import VectorDB
from embedding.glove import GloveEmbedding
import numpy as np


CEFR_LEVEL = {
    'a1': 0.0,
    'a2': 0.2,
    'b1': 0.4,
    'b2': 0.6,
    'c1': 0.8,
    'c2': 1.0
}


def load_word_dict():
    db = VectorDB()
    vocab = db.query_all()
    result = {}
    for row in vocab:
        result[row['word']] = {
            'CEFR': row['CEFR'],
            'mem': row['understanding_rating'],
            'IELTS': row['IELTS'],
            'GRE': row['GRE'],
        }
    return result


def get_word_similarity(word1: str, word2: str, embedding):
    vec1 = np.array(embedding.encode(word1))
    vec2 = np.array(embedding.encode(word2))
    return np.dot(vec1, vec2) / np.linalg.norm(vec1) / np.linalg.norm(vec2)


def score(words: List[str], keywords: List[str], exam: Optional[str]):
    word_dict = load_word_dict()
    embedding = GloveEmbedding()

    # Check for hallucination
    checked_words = []
    for word in words:
        if word in word_dict:
            checked_words.append(word)
    hallucination_scalar = len(checked_words) / len(words)
    words = checked_words

    average_similarities = []
    for word in words:
        similarities = []
        for key in keywords:
            similarities.append(get_word_similarity(word, key, embedding))
        average_similarities.append(np.mean(similarities))
    score_similarity = np.mean(average_similarities)

    cefr_levels = []
    for word in words:
        cefr_levels.append(CEFR_LEVEL[word_dict[word]['CEFR']])
    cefr_score = (0.5 - np.abs(np.mean(cefr_levels) - 0.5)) * 2.0

    exam_hit_rate = 1.0
    if exam is not None:
        exam_hits = []
        for word in words:
            exam_hits.append(1.0 if word_dict[word][exam] else 0.0)
        exam_hit_rate = np.mean(exam_hits)
    
    mem_levels = []
    for word in words:
        mem_levels.append(word_dict[word]['mem'])
    mem_score = 1.0 - np.mean(mem_levels)

    scores = np.array([score_similarity, cefr_score, exam_hit_rate, mem_score])
    weights = np.array([1.0, 0.5, 2.0, 1.0])

    print('Hallucination scalar:', hallucination_scalar)
    print('[Similarity, CEFR, Exam, Mem]', scores)

    return np.dot(scores, weights) / np.sum(weights) * hallucination_scalar


if __name__ == '__main__':
    words = ['suppose', 'ideal', 'relation', 'definite', 'element', 'dimension', 'corresponding', 'constraint', 'hypothesis', 'complexity']
    # words = ['diagram', 'measure', 'calculate', 'ratio', 'decimal', 'fraction', 'algebra', 'geometry', 'tally', 'equation']
    keywords = ['equation', 'integral', 'derivative', 'theorem', 'hypotenuse']
    exam = None
    print(score(words, keywords, exam))
