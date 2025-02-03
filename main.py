from config import PDF_FOLDER
from rag_pipeline import process_multiple_pdfs, query_rag
from utils.logger_config import logger
import argparse


def main(process_pdfs: bool = False) -> None:
    """
    Main function to either process PDFs and/or query the RAG system.

    Args:
        process_pdfs (bool): Whether to process PDFs before querying.
    """
    if process_pdfs:
        logger.info("Processing all PDFs in %s...", PDF_FOLDER)
        process_multiple_pdfs(PDF_FOLDER)

    # User Input for Query
    user_query = input("\nEnter your query: ").strip()

    if not user_query:
        logger.warning("No query provided. Exiting.")
        return

    # Retrieve AI Response
    ai_response = query_rag(user_query)

    logger.info("\nAI Response:\n%s", ai_response)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the RAG pipeline.")
    parser.add_argument(
        "--process_pdfs", action="store_true", help="Process PDFs before querying."
    )
    args = parser.parse_args()

    main(process_pdfs=args.process_pdfs)

# query = "Who should install the CCTV Surveillance Camera and devices?"
# query = "Why should we install the CCTV Surveillance Camera and devices?"
