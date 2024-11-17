import pandas as pd

df = pd.read_csv('oxford-5k.csv')
df = df[['word', 'level']]
df.to_csv('base.csv', index=False)
