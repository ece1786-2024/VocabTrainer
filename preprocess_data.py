import pandas as pd


base = pd.read_csv('dataset/base/base.csv')
ielts = pd.read_csv('dataset/ielts/ielts.csv')
gre = pd.read_csv('dataset/gre/gre.csv')

ielts_set = set(ielts['word'])
gre_set = set(gre['word'])

base['ielts'] = base['word'].apply(lambda x: 1 if x in ielts_set else 0)
base['gre'] = base['word'].apply(lambda x: 1 if x in gre_set else 0)

base.to_csv('dataset/data.csv', index=False)