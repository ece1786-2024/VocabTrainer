import chromadb
from chromadb.config import Settings


class WordEmbeddingDatabase:
    def __init__(self, persist_directory="./chromadb"):
        """
        Initialize the WordEmbeddingDatabase.

        :param persist_directory: Directory to persist the ChromaDB database.
        """
        self.client = chromadb.Client(Settings(persist_directory=persist_directory))
        
        # Check if collection exists; if so, reuse it
        existing_collections = [col.name for col in self.client.list_collections()]
        if "words_collection" in existing_collections:
            print("Found collection")
            self.collection = self.client.get_collection(name="words_collection")
        else:
            print("Collection is not found")
            self.collection = self.client.create_collection(name="words_collection")

    def add_word(self, word, embedding, category, exam):
        """
        Add a single word with its embedding, category, and exam metadata to the database.

        :param word: Word to add.
        :param embedding: Corresponding embedding for the word (vector).
        :param category: Category for the word.
        :param exam: Exam name associated with the word.
        """
        if embedding is None:
            raise ValueError(f"Embedding for '{word}' is None. Cannot add to the database.")
        
        metadata = {'category': category, 'exam': exam}
        self.collection.add(
            documents=[word],
            metadatas=[metadata],
            embeddings=[embedding],
            ids=[word]
        )

    def query_by_similarity(self, query_embedding, n_results=2):
        """
        Query the database for similar words by embedding similarity.

        :param query_embedding: Embedding to query for.
        :param n_results: Number of results to return.
        :return: Query results.
        """
        if query_embedding is not None:
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


if __name__ == "__main__":
    # Initialize the WordEmbeddingDatabase
    db = WordEmbeddingDatabase()

    # Define input data
    words = ['hello', 'world', 'in', 'python']
    categories = ['greeting', 'noun', 'preposition', 'programming_language']
    exams = ['general', 'general', 'general', 'coding']
    
    # Example embeddings (replace with actual embeddings)
    embeddings = [
        [0.1, 0.2, 0.3],  # Dummy embedding for 'hello'
        [0.4, 0.5, 0.6],  # Dummy embedding for 'world'
        [0.7, 0.8, 0.9],  # Dummy embedding for 'in'
        [1.0, 1.1, 1.2]   # Dummy embedding for 'python'
    ]

    # Uncomment to add words to the database
    # for word, embedding, category, exam in zip(words, embeddings, categories, exams):
    #     db.add_word(word, embedding, category, exam)

    # Query by similarity
    query_embedding = [0.15, 0.25, 0.35]  # Example query embedding
    try:
        similar_results = db.query_by_similarity(query_embedding, n_results=3)
        print("Results for embedding similarity query:")
        print(similar_results)
    except ValueError as e:
        print(e)

    # Query by exam name
    exam_name = 'general'
    exam_results = db.query_by_exam(exam_name)
    print(f"Results for query by exam name '{exam_name}':")
    print(exam_results)
