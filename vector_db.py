import chromadb
from chromadb.config import Settings


class WordEmbeddingDatabase:
    def __init__(self, glove_file, persist_directory=".chromadb"):
        """
        Initialize the WordEmbeddingDatabase.

        :param glove_file: Path to the GloVe embeddings file.
        :param persist_directory: Directory to persist the ChromaDB database.
        """
        self.client = chromadb.Client(Settings(persist_directory=persist_directory))
        self.collection = self.client.create_collection(name="words_collection")
        self.embeddings = self._load_glove_embeddings(glove_file)

    def _load_glove_embeddings(self, file_path):
        """
        Load GloVe embeddings from a file.

        :param file_path: Path to the GloVe file.
        :return: Dictionary of embeddings.
        """
        embeddings = {}
        with open(file_path, 'r', encoding='utf8') as f:
            for line in f:
                values = line.strip().split()
                word = values[0]
                vector = [float(x) for x in values[1:]]
                embeddings[word] = vector
        return embeddings

    def add_words(self, words, categories, exams):
        """
        Add words with categories and exam metadata to the database.

        :param words: List of words to add.
        :param categories: Corresponding categories for the words.
        :param exams: Corresponding exam names for the words.
        """
        documents = []
        metadatas = []
        embeddings_list = []
        ids = []

        for word, category, exam in zip(words, categories, exams):
            if word in self.embeddings:
                embedding = self.embeddings[word]
                metadata = {'category': category, 'exam': exam}
                documents.append(word)
                metadatas.append(metadata)
                embeddings_list.append(embedding)
                ids.append(word)
            else:
                print(f"Embedding for '{word}' not found.")

        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings_list,
            ids=ids
        )

    def query_by_similarity(self, query_word, n_results=2):
        """
        Query the database for similar words by embedding similarity.

        :param query_word: Word to query for.
        :param n_results: Number of results to return.
        :return: Query results.
        """
        if query_word in self.embeddings:
            query_embedding = self.embeddings[query_word]
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )
            return results
        else:
            raise ValueError(f"Embedding for '{query_word}' not found.")

    def query_by_exam(self, exam_name):
        """
        Query the database by exam name.

        :param exam_name: Exam name to filter by.
        :return: Query results.
        """
        exam_query = {'exam': exam_name}
        results = self.collection.get(where=exam_query)
        return results


# Example Usage
if __name__ == "__main__":
    glove_file = 'glove.6B.50d.txt'  # Path to your GloVe embeddings file
    db = WordEmbeddingDatabase(glove_file)

    # Words, categories, and exams
    words = ['hello', 'world', 'in', 'python']
    categories = ['greeting', 'noun', 'preposition', 'programming_language']
    exams = ['general', 'general', 'general', 'coding']

    # Add words to the database
    db.add_words(words, categories, exams)

    # Query by similarity
    try:
        similar_results = db.query_by_similarity('hello', n_results=3)
        print("Results for embedding similarity query:")
        print(similar_results)
    except ValueError as e:
        print(e)

    # Query by exam name
    exam_results = db.query_by_exam('general')
    print("Results for query by exam name:")
    print(exam_results)
