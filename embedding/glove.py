import torch

class GloveEmbedding():
    def __init__(self):
        glove_file = 'glove.6B.50d.txt'
        self.embeddings = {}
        with open(glove_file, 'r', encoding='utf8') as f:
            for line in f:
                values = line.strip().split()
                word = values[0]
                vector = torch.tensor([float(x) for x in values[1:]], dtype=torch.float)
                self.embeddings[word] = vector
        self.dim = len(next(iter(self.embeddings.values())))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def contains(self, word):
        return word in self.embeddings

    def encode(self, word):
        word = word.lower()
        if self.contains(word):
            embedding = self.embeddings[word].to(self.device)
        else:
            embedding = torch.zeros(self.dim).to(self.device)
        return embedding.cpu().numpy().tolist()
