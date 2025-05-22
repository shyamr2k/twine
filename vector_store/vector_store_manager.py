import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from pathlib import Path
import logging

# Get a logger instance
logger = logging.getLogger(__name__)

# Constants
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
# Adjusted to be relative to this file's location for robust pathing
DEFAULT_VECTOR_STORE_DIR = str(Path(__file__).resolve().parent.parent / "data" / "vector_store_data")
FAISS_INDEX_FILE = 'index.faiss'
DOC_ID_MAP_FILE = 'doc_id_map.pkl'

class VectorStoreManager:
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME, store_dir: str = DEFAULT_VECTOR_STORE_DIR):
        """
        Initializes the VectorStoreManager.

        Args:
            model_name: The name of the sentence transformer model to use.
            store_dir: The directory to store/load the FAISS index and document ID map.
        """
        self.model = SentenceTransformer(model_name)
        self.store_dir = Path(store_dir)
        logger.info(f"Vector store directory set to: {self.store_dir}")
        try:
            self.store_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Ensured vector store directory exists: {self.store_dir}")
        except Exception as e:
            logger.error(f"Error creating vector store directory {self.store_dir}: {e}", exc_info=True)
            # Depending on desired behavior, might re-raise or handle gracefully
            raise # Re-raise for now, as this is critical for store operation

        self.index = None
        self.doc_id_map = []  # Stores original document IDs

        if not self.load_store():
            logger.info("No existing vector store found or failed to load. A new one will be created upon 'build_store'.")

    def build_store(self, documents: list[dict], batch_size: int = 32):
        """
        Builds the FAISS index from a list of documents.

        Args:
            documents: A list of dictionaries, e.g., [{'id': 'doc1', 'content': 'text1'}, ...].
            batch_size: Batch size for encoding documents.
        """
        if not documents:
            logger.warning("No documents provided to build the store.")
            return

        logger.info(f"Building new vector store with {len(documents)} documents.")
        contents = [doc['content'] for doc in documents]
        doc_ids = [doc['id'] for doc in documents]

        # Encode documents in batches
        all_embeddings = []
        for i in range(0, len(contents), batch_size):
            batch_contents = contents[i:i + batch_size]
            try:
                embeddings = self.model.encode(batch_contents, convert_to_tensor=False)
                all_embeddings.extend(embeddings)
                logger.debug(f"Encoded batch {i // batch_size + 1}/{(len(contents) - 1) // batch_size + 1}")
            except Exception as e:
                logger.error(f"Error encoding batch {i // batch_size + 1}: {e}", exc_info=True)
                # Decide if to continue with partial data or stop
                # For now, let's stop if a batch fails, as it might indicate a bigger issue.
                logger.error("Stopping build_store due to encoding error.")
                return


        if not all_embeddings:
            logger.warning("No embeddings were generated from the documents. Cannot build store.")
            return
            
        embeddings_np = np.array(all_embeddings).astype('float32')
        logger.info(f"Generated {embeddings_np.shape[0]} embeddings with dimension {embeddings_np.shape[1]}.")


        if embeddings_np.ndim == 1: # Handle case of single document
            embeddings_np = np.expand_dims(embeddings_np, axis=0)
        
        if embeddings_np.shape[0] == 0: # Should be caught by `if not all_embeddings` already
            logger.warning("No embeddings generated (shape[0] is 0). Cannot build store.")
            return

        try:
            embedding_dimension = embeddings_np.shape[1]
            self.index = faiss.IndexFlatL2(embedding_dimension)
            self.index.add(embeddings_np)
            self.doc_id_map = doc_ids

            self.save_store() # save_store already has logging
            logger.info(f"Vector store built and saved successfully with {self.index.ntotal} vectors.")
        except Exception as e:
            logger.error(f"Failed to build or save FAISS index: {e}", exc_info=True)
            # Reset to a clean state
            self.index = None
            self.doc_id_map = []

    def add_documents(self, documents: list[dict], batch_size: int = 32):
        """
        (Placeholder) Adds new documents to an existing store.
        This is a more complex operation that might involve re-training or updating the index.
        For IndexFlatL2, it means re-building if you want optimal search, or simply adding
        and hoping for the best (which can degrade performance over many additions).
        """
        if not self.index:
            logger.warning("Index not initialized. Build the store first or load an existing one.")
            logger.info("Falling back to 'build_store' as no index exists for 'add_documents'.")
            self.build_store(documents, batch_size)
            return

        if not documents:
            logger.warning("No documents provided to add_documents method.")
            return

        logger.info(f"Adding {len(documents)} new documents to the existing vector store.")
        contents = [doc['content'] for doc in documents]
        new_doc_ids = [doc['id'] for doc in documents]

        all_new_embeddings = []
        for i in range(0, len(contents), batch_size):
            batch_contents = contents[i:i + batch_size]
            try:
                embeddings = self.model.encode(batch_contents, convert_to_tensor=False)
                all_new_embeddings.extend(embeddings)
                logger.debug(f"Encoded batch {i // batch_size + 1}/{(len(contents) - 1) // batch_size + 1} for new documents")
            except Exception as e:
                logger.error(f"Error encoding batch {i // batch_size + 1} for new documents: {e}", exc_info=True)
                logger.error("Stopping add_documents due to encoding error.")
                return
        
        if not all_new_embeddings:
            logger.warning("No embeddings were generated for the new documents.")
            return

        embeddings_np = np.array(all_new_embeddings).astype('float32')

        if embeddings_np.ndim == 1: # Handle case of single document
            embeddings_np = np.expand_dims(embeddings_np, axis=0)

        if embeddings_np.shape[0] > 0:
            try:
                self.index.add(embeddings_np)
                self.doc_id_map.extend(new_doc_ids)
                self.save_store() # save_store already has logging
                logger.info(f"Successfully added {embeddings_np.shape[0]} new documents. Index now has {self.index.ntotal} vectors.")
            except Exception as e:
                logger.error(f"Error adding new documents to FAISS index or saving: {e}", exc_info=True)
                # Note: State might be inconsistent here (some added to index in memory, but not saved)
                # A more robust solution might involve trying to revert the add or marking store as dirty
        else:
            logger.warning("No valid embeddings generated for new documents, so no documents were added.")


    def search(self, query_text: str, k: int = 5) -> list[tuple[str, float]]:
        """
        Searches the FAISS index for the k nearest neighbors to the query text.

        Args:
            query_text: The query string.
            k: The number of top results to retrieve.

        Returns:
            A list of tuples (document_id, score/distance).
        """
        logger.debug(f"Searching for query: '{query_text}' with k={k}")
        if self.index is None or self.index.ntotal == 0:
            logger.warning("Search attempted on an uninitialized or empty vector store.")
            return []

        try:
            query_embedding = self.model.encode([query_text], convert_to_tensor=False).astype('float32')
            if query_embedding.ndim == 1:
                 query_embedding = np.expand_dims(query_embedding, axis=0)

            # Ensure k is not greater than the number of items in the index
            actual_k = min(k, self.index.ntotal)
            if actual_k == 0:
                logger.debug("No items in index to search.") # Debug as it's a valid state
                return []
            
            distances, indices = self.index.search(query_embedding, actual_k)
            logger.debug(f"Found {len(distances[0])} results for query '{query_text}'.")

            results = []
            for i in range(len(indices[0])):
                idx = indices[0][i]
                dist = distances[0][i]
                if idx < len(self.doc_id_map):
                    results.append((self.doc_id_map[idx], float(dist)))
                else:
                    logger.warning(f"Index {idx} from FAISS search out of bounds for doc_id_map (size {len(self.doc_id_map)}). This may indicate inconsistency.")
            return results
        except Exception as e:
            logger.error(f"Error during FAISS search for query '{query_text}': {e}", exc_info=True)
            return []

    def save_store(self):
        """Saves the FAISS index and document ID map to disk."""
        if self.index is None:
            logger.warning("Index is not initialized. Nothing to save.")
            return

        index_path = self.store_dir / FAISS_INDEX_FILE
        map_path = self.store_dir / DOC_ID_MAP_FILE
        logger.info(f"Attempting to save vector store to directory: {self.store_dir}")

        try:
            logger.info(f"Saving FAISS index to {index_path}")
            faiss.write_index(self.index, str(index_path))
            logger.info(f"FAISS index saved successfully to {index_path}")

            logger.info(f"Saving document ID map to {map_path}")
            with open(map_path, 'wb') as f:
                pickle.dump(self.doc_id_map, f)
            logger.info(f"Document ID map saved successfully to {map_path}")
            
            logger.info(f"Vector store (index and map) saved successfully to {self.store_dir}")
        except Exception as e:
            logger.error(f"Error saving vector store to {self.store_dir}: {e}", exc_info=True)
            # Consider if we should attempt to clean up partially saved files

    def load_store(self) -> bool:
        """
        Loads the FAISS index and document ID map from disk.

        Returns:
            True if loading was successful, False otherwise.
        """
        index_path = self.store_dir / FAISS_INDEX_FILE
        map_path = self.store_dir / DOC_ID_MAP_FILE
        logger.info(f"Attempting to load vector store from: {self.store_dir}")

        if index_path.exists() and map_path.exists():
            try:
                logger.info(f"Loading FAISS index from {index_path}")
                self.index = faiss.read_index(str(index_path))
                logger.info(f"FAISS index loaded successfully. Index has {self.index.ntotal} vectors.")

                logger.info(f"Loading document ID map from {map_path}")
                with open(map_path, 'rb') as f:
                    self.doc_id_map = pickle.load(f)
                logger.info(f"Document ID map loaded successfully. Map contains {len(self.doc_id_map)} entries.")
                
                if self.index.ntotal != len(self.doc_id_map):
                    logger.warning(f"Mismatch between FAISS index size ({self.index.ntotal}) and doc_id_map size ({len(self.doc_id_map)}). Store may be inconsistent.")
                    # Decide on handling: could raise error, try to reconcile, or just warn.
                    # For now, warn and proceed. A more robust system might invalidate the store.

                logger.info(f"Vector store loaded successfully from {self.store_dir}.")
                return True
            except FileNotFoundError: # Should be caught by index_path.exists() but as defense
                logger.error(f"File not found during load (index: {index_path}, map: {map_path}). This shouldn't happen if exists() check passed.", exc_info=True)
                self.index = None
                self.doc_id_map = []
                return False
            except pickle.UnpicklingError as e:
                logger.error(f"Error unpickling doc_id_map from {map_path}: {e}", exc_info=True)
                self.index = None # If map fails, index is useless
                self.doc_id_map = []
                return False
            except Exception as e: # Catch other FAISS errors or unexpected issues
                logger.error(f"Error loading vector store from {self.store_dir}: {e}", exc_info=True)
                self.index = None
                self.doc_id_map = []
                return False
        else:
            logger.info(f"Vector store not found at {self.store_dir} (index or map file missing).")
            return False

if __name__ == "__main__":
    # --- Basic Logging Setup for Standalone Script Execution ---
    # This is helpful if you run this script directly.
    # If it's imported, the importing module (e.g., api/main.py) should set up logging.
    if not logging.getLogger().hasHandlers(): # Check if root logger has handlers
        logging.basicConfig(level=logging.INFO, 
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                            handlers=[logging.StreamHandler()]) # Ensure logs go to console
        logger.info("Basic logging configured for standalone execution of vector_store_manager.py.")


    logger.info(f"VectorStoreManager script started (standalone execution). Default store directory: {DEFAULT_VECTOR_STORE_DIR}")
    
    # Ensure the store directory exists or can be created by the class constructor
    # Path(DEFAULT_VECTOR_STORE_DIR).mkdir(parents=True, exist_ok=True) # Constructor handles this now

    # 1. Instantiate VectorStoreManager
    logger.info("\n--- Initializing VectorStoreManager (Demo) ---")
    vector_manager = VectorStoreManager() # Uses DEFAULT_VECTOR_STORE_DIR

    # 2. Define dummy documents for initial build
    dummy_documents_initial = [
        {'id': 'report1_chunk1', 'content': 'Patient John Doe shows signs of recovery after treatment.'},
        {'id': 'report1_chunk2', 'content': 'Follow up recommended in 6 months for John Doe.'},
        {'id': 'research_paper_alpha', 'content': 'Study Alpha indicates new methods for early detection.'},
        {'id': 'guideline_beta', 'content': 'Guideline Beta suggests yearly checkups for specific conditions.'}
    ]

    # 3. Build the store if it wasn't loaded or to overwrite
    logger.info("\n--- Building Vector Store (Initial Documents - Demo) ---")
    # In a real demo, you might want to check if the store needs building or use --force logic.
    # For simplicity here, we just build. VectorStoreManager itself will log details.
    vector_manager.build_store(dummy_documents_initial)

    # 4. Perform a sample search
    logger.info("\n--- Performing Sample Search (Demo) ---")
    if vector_manager.index and vector_manager.index.ntotal > 0:
        search_query = "John Doe recovery"
        search_results = vector_manager.search(search_query, k=2)
        logger.info(f"Search results for '{search_query}':")
        if search_results:
            for doc_id, score in search_results:
                logger.info(f"  ID: {doc_id}, Score (Distance): {score:.4f}")
        else:
            logger.info("  No results found or store is empty.")
    else:
        logger.warning("Skipping search as vector store is not available or empty.")

    # 5. Demonstrate adding more documents
    logger.info("\n--- Adding More Documents (Demo) ---")
    dummy_documents_additional = [
        {'id': 'report2_chunk1', 'content': 'Patient Jane Smith is responding well to therapy.'},
        {'id': 'research_paper_gamma', 'content': 'Study Gamma explores long-term effects of certain medications.'}
    ]
    vector_manager.add_documents(dummy_documents_additional) # Method itself logs details

    # 6. Perform another search to see if new documents are found
    logger.info("\n--- Performing Sample Search After Adding Documents (Demo) ---")
    if vector_manager.index and vector_manager.index.ntotal > 0:
        search_query_2 = "Jane Smith therapy"
        search_results_2 = vector_manager.search(search_query_2, k=2)
        logger.info(f"Search results for '{search_query_2}':")
        if search_results_2:
            for doc_id, score in search_results_2:
                logger.info(f"  ID: {doc_id}, Score (Distance): {score:.4f}")
        else:
            logger.info("  No results found.")
            
        search_query_3 = "John Doe" # Check if original docs are still searchable
        search_results_3 = vector_manager.search(search_query_3, k=3)
        logger.info(f"\nSearch results for '{search_query_3}':")
        if search_results_3:
            for doc_id, score in search_results_3:
                logger.info(f"  ID: {doc_id}, Score (Distance): {score:.4f}")
        else:
            logger.info("  No results found for John Doe.")
    else:
        logger.warning("Skipping search as vector store is not available or empty after additions.")
    
    logger.info("\nVectorStoreManager script (standalone demo) finished.")
