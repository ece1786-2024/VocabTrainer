from vectordb import VectorDB
from embedding.glove import GloveEmbedding
from tqdm import tqdm
import pandas as pd


DATA_PATH = 'dataset/data.csv'


def setup():
    emb = GloveEmbedding()
    db = VectorDB()
    df = pd.read_csv(DATA_PATH)

    # Insert to database, only needs to be executed once
    for row in tqdm(df.itertuples(index=False), total=len(df), desc='Processing'):
        word, cefr, in_ielts, in_gre = row.word, row.level, row.ielts, row.gre
        vec = emb.encode(word)
        db.add_word(word, vec, cefr, in_ielts==1, in_gre==1)

if __name__ == '__main__':
    setup()
