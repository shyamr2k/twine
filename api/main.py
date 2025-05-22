import os
from pathlib import Path
from typing import List, Dict
import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Adjusting import paths based on typical project structure where 'api' is a module
# and the script is run from the root directory (e.g. using uvicorn api.main:app)
# If running this script directly, these imports might need adjustment
# or the PYTHONPATH needs to be set.

# Assuming 'scripts' or root is in PYTHONPATH or we are running from root
from data_ingestion.ingest import load_documents_from_directory
from vector_store.vector_store_manager import VectorStoreManager
from retriever.retriever import DocumentRetriever
from llm_integration.llm_handler import LLMHandler

# --- Global Variables & Initialization ---

# Define paths relative to the root of the project where this api/main.py is located.
# When running with 'uvicorn api.main:app --reload' from the project root,
# Path.cwd() is the project root. So, these relative paths should work.
# If main.py is in /app/api/, then ../data is /app/data/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "reports"
VECTOR_STORE_DIR = PROJECT_ROOT / "data" / "vector_store_data"

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

logger.info(f"Attempting to load data from: {DATA_DIR}")
logger.info(f"Vector store directory: {VECTOR_STORE_DIR}")

app = FastAPI()

# --- Global Initializations ---
all_docs: List[Dict[str, str]] = []
vector_manager: VectorStoreManager = None
doc_retriever: DocumentRetriever = None
llm_handler: LLMHandler = None

try:
    # Ensure DATA_DIR exists
    if not DATA_DIR.exists():
        logger.warning(f"Data directory {DATA_DIR} does not exist. Creating it.")
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        # Optionally, create a dummy file if you expect at least one file
        # with open(DATA_DIR / "dummy_initial_report.txt", "w") as f:
        #     f.write("This is a dummy report to ensure the directory is not empty on first run.")

    all_docs = load_documents_from_directory(str(DATA_DIR))
    if not all_docs:
        logger.warning(f"No documents found in {DATA_DIR}. The RAG system might not provide meaningful answers if vector store is also empty or not built.")
    else:
        logger.info(f"Successfully loaded {len(all_docs)} documents from {DATA_DIR}.")

    # Ensure VECTOR_STORE_DIR exists for VectorStoreManager
    VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
    vector_manager = VectorStoreManager(store_dir=str(VECTOR_STORE_DIR))

    if not vector_manager.load_store(): # load_store returns True if successful
        if all_docs:
            logger.info("Building vector store for the first time as it was not found or failed to load...")
            vector_manager.build_store(all_docs)
            logger.info("Vector store built and saved.")
        else:
            logger.warning("No documents loaded, and no existing vector store found. Vector store remains empty.")
    elif not all_docs and vector_manager.index is not None and vector_manager.index.ntotal == 0 :
         logger.info("Vector store loaded, but it's empty and no new documents found to add.")
    elif not all_docs and vector_manager.index is not None and vector_manager.index.ntotal > 0 :
         logger.info(f"Vector store loaded successfully with {vector_manager.index.ntotal} items. No new documents found in {DATA_DIR}.")
    else: # Store loaded
        logger.info(f"Vector store loaded successfully with {vector_manager.index.ntotal if vector_manager.index else 'N/A'} items.")

    doc_retriever = DocumentRetriever(vector_store_manager=vector_manager, all_documents=all_docs)
    llm_handler = LLMHandler() # Using default placeholder LLM

except Exception as e:
    logger.critical(f"Fatal error during application initialization: {e}", exc_info=True)
    # Depending on the desired behavior, you might want to exit or prevent the app from starting
    # For FastAPI, if this happens before app.run, it will likely prevent startup.
    # If this needs to be handled by FastAPI, then parts of this init need to be in a startup event.
    raise RuntimeError(f"Could not initialize application components: {e}")

# --- Pydantic Models ---

class QueryRequest(BaseModel):
    query: str
    k: int = 5  # Optional, number of documents to retrieve

class QueryResponse(BaseModel):
    query: str
    answer: str
    retrieved_ids: List[str]
    retrieved_contexts: List[Dict[str, any]] # Matches retriever.retrieve output format

# --- API Endpoints ---

@app.get("/")
async def root():
    """
    Root endpoint to check if the API is running.
    """
    return {"message": "Medical Report RAG API is running. POST to /process for queries."}

@app.post("/process", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """
    Processes a user query by retrieving relevant documents and generating an answer.
    """
    logger.info(f"Received query: '{request.query}' with k={request.k}")

    if not all_docs and (vector_manager is None or vector_manager.index is None or vector_manager.index.ntotal == 0):
        logger.error("Query received, but no documents are loaded and the vector store is empty.")
        # It's better to raise HTTPException for API errors
        raise HTTPException(
            status_code=503, 
            detail="Error: No documents have been loaded into the system, and the vector store is empty. Please add data to the 'data/reports' directory and restart."
        )

    try:
        # 1. Retrieve relevant documents
        logger.debug(f"Retrieving documents for query: '{request.query}'")
        retrieved_docs: List[Dict[str, any]] = doc_retriever.retrieve(query=request.query, k=request.k)
        
        retrieved_ids = [doc['id'] for doc in retrieved_docs]
        logger.info(f"Retrieved {len(retrieved_docs)} documents with IDs: {retrieved_ids}")
        if not retrieved_docs:
            logger.warning(f"No documents retrieved for query: '{request.query}'")

        # 2. Generate response using LLM
        logger.debug(f"Generating LLM response for query: '{request.query}'")
        answer = llm_handler.generate_response(query=request.query, retrieved_contexts=retrieved_docs)
        logger.info(f"Generated answer snippet: \"{answer[:100]}...\"")

        return QueryResponse(
            query=request.query,
            answer=answer,
            retrieved_ids=retrieved_ids,
            retrieved_contexts=retrieved_docs
        )
    except Exception as e:
        logger.error(f"Error during query processing for query '{request.query}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {str(e)}")
        answer=answer,
        retrieved_ids=retrieved_ids,
        retrieved_contexts=retrieved_docs
    )

# To run (from the project root directory, assuming this file is api/main.py):
# uvicorn api.main:app --reload --port 8000
#
# Example (after starting the server):
# curl -X POST "http://localhost:8000/process" -H "Content-Type: application/json" -d '{"query": "What are the common symptoms of flu?", "k": 3}'

if __name__ == "__main__":
    # This block is for direct execution (e.g., python api/main.py)
    # It's more common to run FastAPI with Uvicorn as shown above.
    print("Starting Uvicorn server for Medical Report RAG API...")
    import uvicorn
    # Get the directory of the current file (api/main.py)
    # Then get its parent directory (api/)
    # Then get the name of that directory, which is 'api'
    module_name = Path(__file__).parent.name # Should be 'api'
    app_module_path = f"{module_name}.main:app"
    
    # Check if running in a context where project root is in PYTHONPATH
    # For direct execution, it's often simpler to add project root to sys.path
    import sys
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
        logger.info(f"Temporarily added {PROJECT_ROOT} to sys.path for direct execution.")

    # Re-check imports if sys.path was modified
    # This is primarily for the direct execution scenario (python api/main.py)
    try:
        from data_ingestion.ingest import load_documents_from_directory
        from vector_store.vector_store_manager import VectorStoreManager
        from retriever.retriever import DocumentRetriever
        from llm_integration.llm_handler import LLMHandler
        logger.info("Imports re-checked successfully after potential sys.path modification.")
    except ImportError as e:
        logger.error(f"ImportError after sys.path modification, this indicates a path issue: {e}", exc_info=True)
        logger.error("Please try running with 'uvicorn api.main:app --reload' from the project root directory.")
        sys.exit(1)

    logger.info(f"Attempting to run Uvicorn with module path: {app_module_path}")
    uvicorn.run(app_module_path, host="0.0.0.0", port=8000, reload=True)
