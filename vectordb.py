import chromadb
# from chromadb.config import Settings


class VectorDB:
    # word, CEFR, embedding, understanding_rating, IELTS, GRE
    def __init__(self, persist_directory="./chromadb"):
        """
        Initialize the WordEmbeddingDatabase.

        :param persist_directory: Directory to persist the ChromaDB database.
        """
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Check if collection exists; if so, reuse it
        existing_collections = [col.name for col in self.client.list_collections()]
        if "words_collection" in existing_collections:
            print("Found collection")
            self.collection = self.client.get_collection(name="words_collection")
        else:
            print("Collection is not found")
            self.collection = self.client.create_collection(name="words_collection")

    def add_word(self, word, embedding, CEFR, IELTS, GRE):
        """
        Add a single word with its embedding, CEFR, and exam metadata to the database.

        :param word: Word to add.
        :param embedding: Corresponding embedding for the word (vector).
        :param CEFR: CEFR level of the word.
        :param IELTS: Boolean indicating if the word is in the IELTS exam.
        :param GRE: Boolean indicating if the word is in the GRE exam.
        """
        if embedding is None:
            raise ValueError(f"Embedding for '{word}' is None. Cannot add to the database.")
        
        # Metadata for the word
        metadata = {
            'CEFR': CEFR,
            'understanding_rating': 0,  # Initialize to 0
            'IELTS': IELTS,
            'GRE': GRE
        }
        
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
        :return: List of dictionaries with word, CEFR, and understanding_rating.
        """
        if query_embedding is None:
            raise ValueError("Query embedding is None. Please provide a valid embedding.")
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas"]
        )
        
        # Extract required fields from the results
        output = []
        for document, metadata in zip(results["documents"][0], results["metadatas"][0]):
            output.append({
                "word": document,
                "CEFR": metadata["CEFR"],
                "understanding_rating": metadata["understanding_rating"]
            })
        
        return output


    def query_by_exam(self, exam_name):
        """
        Query the database by exam name.

        :param exam_name: Exam name to filter by ('IELTS' or 'GRE').
        :return: List of dictionaries with word, CEFR, and understanding_rating.
        """
        if exam_name not in ["IELTS", "GRE"]:
            raise ValueError("Exam name must be 'IELTS' or 'GRE'.")
        
        exam_query = {exam_name: True}
        results = self.collection.get(where=exam_query, include=["documents", "metadatas"])
        
        # Extract required fields from the results
        output = []
        for document, metadata in zip(results["documents"], results["metadatas"]):
            output.append({
                "word": document,
                "CEFR": metadata["CEFR"],
                "understanding_rating": metadata["understanding_rating"]
            })
        
        return output


if __name__ == "__main__":
    # Initialize the WordEmbeddingDatabase
    db = VectorDB()

    # Define input data
    words = ['hello', 'world', 'in', 'python']
    difficulties = ['easy', 'medium', 'easy', 'hard']
    in_IELTS = [True, False, False, True]
    in_GRE = [False, True, True, False]
    
    # Example embeddings (replace with actual embeddings)
    embeddings = [
        [0.1, 0.2, 0.3],  # Dummy embedding for 'hello'
        [0.4, 0.5, 0.6],  # Dummy embedding for 'world'
        [0.7, 0.8, 0.9],  # Dummy embedding for 'in'
        [1.0, 1.1, 1.2]   # Dummy embedding for 'python'
    ]

    # Uncomment to add words to the database
    for word, embedding, CEFR, ielts, gre in zip(words, embeddings, difficulties, in_IELTS, in_GRE):
        db.add_word(word, embedding, CEFR, ielts, gre)

    # Query by similarity
    query_embedding = [0.15, 0.25, 0.35]  # Example query embedding
    try:
        similar_results = db.query_by_similarity(query_embedding, n_results=3)
        print("Results for embedding similarity query:")
        print(similar_results)
    except ValueError as e:
        print(e)

    # Query by exam name
    exam_name = 'IELTS'  # Can also use 'GRE'
    exam_results = db.query_by_exam(exam_name)
    print(f"Results for query by exam name '{exam_name}':")
    print(exam_results)
