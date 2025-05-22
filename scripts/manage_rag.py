import argparse
from pathlib import Path
import sys
import os # os is imported as per requirement, though Pathlib is mostly used
import logging

# Get a logger instance
logger = logging.getLogger(__name__)

# Ensure the script can find other project modules if run directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Now that PROJECT_ROOT is in sys.path, we can import project modules
try:
    from data_ingestion.ingest import load_documents_from_directory
    from vector_store.vector_store_manager import VectorStoreManager
except ImportError as e:
    # Configure basic logging for this error message if it hasn't been configured yet
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.critical(f"Error importing project modules: {e}", exc_info=True)
    logger.critical(f"PROJECT_ROOT ({PROJECT_ROOT}) was added to sys.path. Current sys.path: {sys.path}")
    logger.critical("Make sure you are running this script from a location where it can find the project's root directory,")
    logger.critical("or ensure that the project's root directory is in your PYTHONPATH.")
    sys.exit(1)

DATA_DIR = PROJECT_ROOT / "data" / "reports"
VECTOR_STORE_DIR = PROJECT_ROOT / "data" / "vector_store_data"

def main():
    # --- Basic Logging Setup ---
    # This will be configured once when main() is called.
    # If other modules called by this script also configure basicConfig, it might lead to unexpected behavior.
    # Ideally, basicConfig is called once at the application entry point.
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    parser = argparse.ArgumentParser(description="Manage RAG system components.")
    parser.add_argument("command", choices=["build_store"], help="Command to execute")
    parser.add_argument("-f", "--force", action="store_true", help="Force rebuild of the vector store even if it exists")

    args = parser.parse_args()

    if args.command == "build_store":
        logger.info("Starting the vector store building process...")

        # Ensure DATA_DIR exists
        if not DATA_DIR.exists() or not DATA_DIR.is_dir():
            logger.error(f"Error: Data directory {DATA_DIR} does not exist or is not a directory.")
            logger.error("Please create it and populate it with .txt reports.")
            sys.exit(1)
        
        # Ensure VECTOR_STORE_DIR exists, create if not
        try:
            VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
            logger.info(f"Vector store directory is set to: {VECTOR_STORE_DIR}")
        except Exception as e:
            logger.error(f"Could not create vector store directory {VECTOR_STORE_DIR}: {e}", exc_info=True)
            sys.exit(1)

        try:
            logger.info(f"Loading documents from: {DATA_DIR}")
            all_docs = load_documents_from_directory(str(DATA_DIR)) # This function now uses logging

            if not all_docs:
                logger.warning(f"No documents found in {DATA_DIR}. Cannot build vector store.")
                sys.exit(1) # Exit if no documents, as store would be empty.
            
            logger.info(f"Loaded {len(all_docs)} documents.")

            vector_manager = VectorStoreManager(store_dir=str(VECTOR_STORE_DIR)) # This class now uses logging

            store_exists_and_loaded = vector_manager.load_store() # This method now uses logging

            if args.force or not store_exists_and_loaded:
                if args.force:
                    logger.info("Force flag detected. Rebuilding vector store...")
                elif not store_exists_and_loaded:
                    logger.info("No existing vector store found or failed to load. Building new store...")
                
                vector_manager.build_store(all_docs) # This method now uses logging
                logger.info("Vector store built/rebuilt successfully.")
            else:
                logger.info("Vector store already exists and appears up-to-date. Use --force to rebuild.")
                logger.info(f"Found {vector_manager.index.ntotal if vector_manager.index else 'N/A'} vectors in existing store.")

        except Exception as e: # Catch-all for unexpected errors during the build process
            logger.error(f"An unexpected error occurred during the 'build_store' process: {e}", exc_info=True)
            sys.exit(1)

    else:
        # Should not happen due to 'choices' in add_argument, but good for safety
        logger.error(f"Unknown command: {args.command}")
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
