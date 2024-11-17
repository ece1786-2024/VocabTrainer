import chromadb
from chromadb.config import Settings

# export CHROMA_PERSIST_DIRECTORY=".chromadb"


# Initialize Chroma client with specified persist directory
client = chromadb.Client()


# Create a collection
collection = client.create_collection(name="words_collection")

# Function to load GloVe embeddings
def load_glove_embeddings(file_path):
    embeddings = {}
    with open(file_path, 'r', encoding='utf8') as f:
        for line in f:
            values = line.strip().split()
            word = values[0]
            vector = [float(x) for x in values[1:]]
            embeddings[word] = vector
    return embeddings

# Load embeddings
glove_file = 'glove.6B.50d.txt'  # Ensure this file is in your working directory
embeddings = load_glove_embeddings(glove_file)

# Words and their categories
words = ['hello', 'world', 'in', 'python']
categories = ['greeting', 'noun', 'preposition', 'programming_language']

# Prepare lists for adding
documents = []
metadatas = []
embeddings_list = []
ids = []

for word, category in zip(words, categories):
    if word in embeddings:
        embedding = embeddings[word]
        metadata = {'category': category}
        documents.append(word)
        metadatas.append(metadata)
        embeddings_list.append(embedding)
        ids.append(word)
    else:
        print(f"Embedding for '{word}' not found.")

# Add words to the collection
collection.add(
    documents=documents,
    metadatas=metadatas,
    embeddings=embeddings_list,
    ids=ids
)

# Query by embedding similarity
query_word = 'hi'
if query_word in embeddings:
    query_embedding = embeddings[query_word]
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=2
    )
    print("Results for embedding similarity query:")
    print(results)
else:
    print(f"Embedding for '{query_word}' not found.")

# Query by metadata
category_query = {'category': 'noun'}
results = collection.get(
    where=category_query
)
print("Results for metadata query by category:")
print(results)
