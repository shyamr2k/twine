import os
from pathlib import Path
from typing import List, Dict
import logging

# Get a logger instance
logger = logging.getLogger(__name__)

def load_documents_from_directory(directory_path: str) -> List[Dict[str, str]]:
    """
    Loads documents from the specified directory.

    Args:
        directory_path: The path to the directory containing the documents.

    Returns:
        A list of dictionaries, where each dictionary represents a document
        and has keys 'id' (filename) and 'content'.
    """
    logger.info(f"Loading documents from directory: {directory_path}")
    documents = []
    processed_count = 0
    file_count = 0
    try:
        for filename in os.listdir(directory_path):
            file_count += 1
            file_path = Path(directory_path) / filename
            if file_path.is_file() and file_path.suffix == '.txt':
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    documents.append({'id': filename, 'content': content})
                    processed_count +=1
                    logger.debug(f"Successfully read and processed file: {file_path}")
                except (IOError, OSError) as e: # Catch specific file operation errors
                    logger.error(f"Error reading file {file_path}: {e}", exc_info=True)
                except Exception as e: # Catch any other unexpected errors during file processing
                    logger.error(f"Unexpected error processing file {file_path}: {e}", exc_info=True)
            # Placeholder for PDF processing logic
            # elif file_path.is_file() and file_path.suffix == '.pdf':
            #     logger.info(f"PDF file {filename} found, processing not yet implemented.")
            #     # Add PDF processing logic here
            #     # content = process_pdf(file_path) # Example
            #     # documents.append({'id': filename, 'content': content})
    except (FileNotFoundError, PermissionError) as e:
        logger.error(f"Error accessing directory {directory_path}: {e}", exc_info=True)
        return [] # Return empty list if directory itself is inaccessible
    except Exception as e:
        logger.error(f"An unexpected error occurred while listing files in {directory_path}: {e}", exc_info=True)
        return []


    logger.info(f"Found {file_count} files in directory, successfully processed {processed_count} .txt files.")
    return documents

# Placeholder for PDF processing function
# def process_pdf(file_path: Path) -> str:
#     """
#     (Not yet implemented) Processes a PDF file to extract text.
#     """
#     # Implementation for PDF extraction would go here
#     # For example, using pypdf2 or pymupdf
#     raise NotImplementedError("PDF processing is not yet implemented.")

if __name__ == "__main__":
    # Setup basic logging for standalone execution
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[logging.StreamHandler()])

    # Example usage:
    # Assuming this script is in 'data_ingestion' and the data is in 'data/reports'
    # The relative path from 'data_ingestion' to 'data/reports' is '../data/reports'
    sample_directory = Path(__file__).resolve().parent.parent / "data" / "reports"
    
    logger.info(f"Attempting to load documents from: {sample_directory}")
    if not sample_directory.exists() or not sample_directory.is_dir():
        logger.error(f"Error: Directory not found or is not a directory: {sample_directory}")
        # Create a dummy directory and file for demonstration if it doesn't exist
        # This is mainly for cases where the script is run in an environment
        # where the 'data/reports' directory isn't set up as expected.
        logger.info("Creating a dummy 'data/reports' directory with a sample file for demonstration.")
        try:
            sample_directory.mkdir(parents=True, exist_ok=True)
            dummy_file_path = sample_directory / "dummy_report.txt"
            with open(dummy_file_path, 'w', encoding='utf-8') as f:
                f.write("This is a dummy report for testing purposes.")
            logger.info(f"Dummy file created: {dummy_file_path}")
        except Exception as e:
            logger.error(f"Could not create dummy directory/file for demonstration: {e}", exc_info=True)


    loaded_docs = load_documents_from_directory(str(sample_directory))

    if loaded_docs:
        logger.info(f"\nSuccessfully loaded {len(loaded_docs)} document(s):")
        for doc in loaded_docs:
            logger.info(f"ID: {doc['id']}")
            logger.info(f"Content: \"{doc['content'][:100]}...\"") # Print first 100 chars
            logger.info("-" * 20)
    else:
        logger.warning("No documents were loaded. Check the directory path and file types.")

    # Example of how PDF processing might be called (currently placeholder)
    # pdf_file_example = sample_directory / "example.pdf"
    # if pdf_file_example.exists():
    #     try:
    #         pdf_text = process_pdf(pdf_file_example)
    #         logger.info(f"\nProcessed PDF (example): {pdf_text[:100]}...")
    #     except NotImplementedError as e:
    #         logger.warning(f"\nPDF processing error: {e}")
