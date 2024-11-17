import pandas as pd
import re


with open("IELTS-4000.txt", "r") as file:
    lines = file.readlines()

words = []
for line in lines:
    match = re.match(r"^[A-Za-z]+:", line)
    if match:
        word = match.group(0).rstrip(':')
        words.append(word)

df = pd.DataFrame(words, columns=['word'])
df.to_csv("ielts.csv", index=False)
