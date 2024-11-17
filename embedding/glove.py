import torch
import nltk

class GloveEmbedding():
    def __init__(self):
        nltk.download('punkt', quiet=True)
        glove_file = 'glove.6B.50d.txt'
        self.embeddings = {}
        with open(glove_file, 'r', encoding='utf8') as f:
            for line in f:
                values = line.strip().split()
                word = values[0]
                vector = torch.tensor([float(x) for x in values[1:]], dtype=torch.float)
                self.embeddings[word] = vector
        self.embedding_dim = len(next(iter(self.embeddings.values())))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = nltk.word_tokenize
        
    def encode(self, text):
        tokens = self.tokenizer(text)
        embeddings = []
        for token in tokens:
            token = token.lower()
            if token in self.embeddings:
                embeddings.append(self.embeddings[token])
            else:
                embeddings.append(torch.zeros(self.embedding_dim))
        if embeddings:
            embeddings = torch.stack(embeddings).to(self.device)
            cls_embedding = torch.mean(embeddings, dim=0)
        else:
            cls_embedding = torch.zeros(self.embedding_dim).to(self.device)
        return cls_embedding.cpu().numpy().tolist()
