import chromadb
from chromadb.config import Settings


class VectorDB:
    def __init__(self, glove_file, persist_directory=".chromadb"):
        """
        Initialize the WordEmbeddingDatabase.

        :param glove_file: Path to the GloVe embeddings file.
        :param persist_directory: Directory to persist the ChromaDB database.
        """
        self.client = chromadb.Client(Settings(persist_directory=persist_directory))
        self.collection = self.client.create_collection(name="words_collection")

    def add_words(self, words, embeddings, categories, exams):
        """
        Add words with their embeddings, categories, and exam metadata to the database.

        :param words: List of words to add.
        :param embeddings: List of corresponding embeddings for the words.
        :param categories: Corresponding categories for the words.
        :param exams: Corresponding exam names for the words.
        """
        if len(words) != len(embeddings) or len(words) != len(categories) or len(words) != len(exams):
            raise ValueError("Input lists (words, embeddings, categories, exams) must have the same length.")

        documents = []
        metadatas = []
        embeddings_list = []
        ids = []

        for word, embedding, category, exam in zip(words, embeddings, categories, exams):
            if embedding is not None:
                metadata = {'category': category, 'exam': exam}
                documents.append(word)
                metadatas.append(metadata)
                embeddings_list.append(embedding)
                ids.append(word)
            else:
                print(f"Embedding for '{word}' is None and will be skipped.")

        self.collection.add(
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings_list,
            ids=ids
        )

    def query_by_similarity(self, query_embedding, n_results=2):
        """
        Query the database for similar words by embedding similarity.

        :param query_embedding: Embedding to query for.
        :param n_results: Number of results to return.
        :return: Query results.
        """
        if query_embedding is not None:  # Check if the embedding is valid
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )
            return results
        else:
            raise ValueError("Query embedding is None. Please provide a valid embedding.")


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
    # Initialize the WordEmbeddingDatabase
    glove_file = 'glove.6B.50d.txt'  # Can change to use Bert
    db = VectorDB(glove_file)

    # Define input data
    words = ['hello', 'world', 'in', 'python']
    categories = ['greeting', 'noun', 'preposition', 'programming_language']
    exams = ['general', 'general', 'general', 'coding']

    # Retrieve GloVe embeddings for the words
    embeddings = [db.embeddings.get(word, None) for word in words]

    # Add words with their metadata and embeddings to the database
    db.add_words(words, embeddings, categories, exams)

    # Query by similarity
    query_word = 'hi'
    query_embedding = db.embeddings.get(query_word, None)
    if query_embedding is not None:
        try:
            similar_results = db.query_by_similarity(query_embedding, n_results=3)
            print("Results for embedding similarity query:")
            print(similar_results)
        except ValueError as e:
            print(e)
    else:
        print(f"Embedding for '{query_word}' not found in the GloVe embeddings.")

    # Query by exam name
    exam_name = 'general'
    exam_results = db.query_by_exam(exam_name)
    print(f"Results for query by exam name '{exam_name}':")
    print(exam_results)
