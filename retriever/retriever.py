from vector_store.vector_store_manager import VectorStoreManager
from typing import List, Dict, Tuple
import logging

# Get a logger instance
logger = logging.getLogger(__name__)

class DocumentRetriever:
    def __init__(self, vector_store_manager: VectorStoreManager, all_documents: List[Dict[str, str]]):
        """
        Initializes the DocumentRetriever.

        Args:
            vector_store_manager: An instance of VectorStoreManager.
            all_documents: A list of dictionaries, where each dictionary
                           represents a document and has 'id' and 'content' keys.
                           This list should contain all documents indexed in the vector store.
        """
        self.vector_store_manager = vector_store_manager
        if all_documents is None: # Defensive check
            logger.warning("all_documents is None during DocumentRetriever initialization. Document map will be empty.")
            self.documents_map = {}
        else:
            self.documents_map = {doc['id']: doc['content'] for doc in all_documents}
        logger.info(f"DocumentRetriever initialized with {len(self.documents_map)} documents in map.")

    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, any]]:
        """
        Retrieves relevant documents for a given query.

        Args:
            query: The user query string.
            k: The number of documents to retrieve.

        Returns:
            A list of dictionaries, where each dictionary contains 'id', 'content',
            and 'score' of a retrieved document.
        """
        logger.debug(f"Retrieving top {k} documents for query: '{query}'")
        retrieved_docs_with_content = []
        
        if not self.vector_store_manager:
            logger.error("VectorStoreManager is not initialized in DocumentRetriever.")
            return [] # Cannot proceed without vector store manager

        try:
            # Use the vector_store_manager to find relevant document IDs and scores
            # search_results is a list of tuples: (document_id, score)
            search_results: List[Tuple[str, float]] = self.vector_store_manager.search(query_text=query, k=k)
            logger.debug(f"Vector search returned {len(search_results)} results.")

            for doc_id, score in search_results:
                content = self.documents_map.get(doc_id)
                if content:
                    retrieved_docs_with_content.append({
                        'id': doc_id,
                        'content': content,
                        'score': score  # score is distance; lower is better
                    })
                else:
                    # This case should ideally not happen if all_documents is comprehensive
                    logger.warning(f"Document ID '{doc_id}' found by vector search but not present in the document map.")
            
            logger.debug(f"Retrieved {len(retrieved_docs_with_content)} documents with content.")
        
        except Exception as e:
            logger.error(f"Error during document retrieval for query '{query}': {e}", exc_info=True)
            # Depending on policy, you might want to return an empty list or re-raise
            return [] # Return empty list on error

        return retrieved_docs_with_content

if __name__ == "__main__":
    # Setup basic logging for standalone execution
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[logging.StreamHandler()])
    logger.info("--- DocumentRetriever Demo ---")

    # 1. Setup: Define sample documents for the demo
    sample_documents = [
        {'id': 'doc_apple_info', 'content': 'Apples are a popular fruit, known for their crisp texture and sweet or tart taste. They are rich in fiber.'},
        {'id': 'doc_orange_info', 'content': 'Oranges are citrus fruits, famous for their high vitamin C content and juicy flesh.'},
        {'id': 'doc_banana_info', 'content': 'Bananas are elongated, curved fruits, that grow in clusters. They are a good source of potassium.'},
        {'id': 'doc_generic_fruit_facts', 'content': 'Fruits are an essential part of a healthy diet, providing vitamins, minerals, and fiber. Apples and oranges are common examples.'}
    ]
    
    # 2. Instantiate VectorStoreManager
    logger.info("\n--- Initializing VectorStoreManager (Demo) ---")
    # Ensure the default directory exists for the demo
    from pathlib import Path
    from vector_store.vector_store_manager import DEFAULT_VECTOR_STORE_DIR # Assuming this is still exposed or use relative path
    # Ensure DEFAULT_VECTOR_STORE_DIR is robustly defined in vector_store_manager or redefine here
    demo_vector_store_path = Path(__file__).resolve().parent.parent / "data" / "vector_store_data"
    demo_vector_store_path.mkdir(parents=True, exist_ok=True)
    
    vector_manager = VectorStoreManager(store_dir=str(demo_vector_store_path))

    # 3. Build/update the vector store with sample documents
    logger.info("\n--- Building/Updating Vector Store with Sample Documents (Demo) ---")
    vector_manager.build_store(sample_documents) # Overwrites/creates if new

    # 4. Instantiate DocumentRetriever
    logger.info("\n--- Initializing DocumentRetriever (Demo) ---")
    retriever_instance = DocumentRetriever(vector_store_manager=vector_manager, all_documents=sample_documents)

    # 5. Perform a sample retrieval
    logger.info("\n--- Performing Sample Retrieval (Demo) ---")
    query1 = "Tell me about apples"
    k_results = 2
    retrieved_documents = retriever_instance.retrieve(query=query1, k=k_results)

    logger.info(f"\nQuery: \"{query1}\" (Top {k_results} results)")
    if retrieved_documents:
        for doc in retrieved_documents:
            logger.info(f"  ID: {doc['id']}")
            logger.info(f"  Score (Distance): {doc['score']:.4f}") # Lower is better
            logger.info(f"  Content: \"{doc['content'][:100]}...\"") # Print snippet
            logger.info("-" * 20)
    else:
        logger.info("  No documents retrieved.")

    query2 = "Which fruits are healthy?"
    retrieved_documents_2 = retriever_instance.retrieve(query=query2, k=k_results) # using k_results for consistency
    logger.info(f"\nQuery: \"{query2}\" (Top {k_results} results)")
    if retrieved_documents_2:
        for doc in retrieved_documents_2:
            logger.info(f"  ID: {doc['id']}")
            logger.info(f"  Score (Distance): {doc['score']:.4f}")
            logger.info(f"  Content: \"{doc['content'][:100]}...\"")
            logger.info("-" * 20)
    else:
        logger.info("  No documents retrieved.")

    logger.info("\n--- DocumentRetriever Demo Finished ---")
