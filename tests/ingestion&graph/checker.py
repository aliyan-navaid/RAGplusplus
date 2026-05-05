import sys
import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# 1. Setup paths
# Current file is in: RAGplusplus/tests/ingestion&graph/checker.py
# Root is 2 levels up: RAGplusplus/
current_dir = Path(__file__).resolve().parent
root_dir = current_dir.parents[1] 
src_path = root_dir / "src"

# 2. Add src to Python path
if str(src_path) not in sys.path:
    sys.path.append(str(src_path))

# Now we can import using absolute paths from 'src'
from ingestion.graph_ingestor import GraphIngestor

def run_test():
    # Load .env from the root folder
    load_dotenv(dotenv_path=root_dir / ".env")
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("Checker")

    # Define the PDF path (in the same folder as checker.py)
    pdf_path = current_dir / "Faizan_Jawaid_Resume.pdf"

    if not pdf_path.exists():
        logger.error(f"❌ Could not find PDF at: {pdf_path}")
        return

    try:
        # Initialize the ingestor
        ingestor = GraphIngestor()
        
        logger.info(f"🚀 Starting ingestion for: {pdf_path.name}")
        
        # Run the pipeline
        result = ingestor.ingest_pdf_file(
            pdf_path=str(pdf_path),
            title="Faizan Resume Test",
            use_ocr=True
        )

        if result.get("success"):
            print("\n" + "✅" * 15)
            print(f"DATABASE: ragplusplus")
            print(f"Status: SUCCESS")
            print(f"Nodes Created: {result.get('sections_created', 0)} Chunks, {result.get('entities_created', 0)} Entities")
            print(f"Total Relationships: {result.get('relationships_created', 0)}")
            print(f"Document ID: {result.get('doc_id', 'N/A')}")
            print("✅" * 15)
        else:
            print("\n" + "❌" * 15)
            print(f"Status: FAILED")
            print(f"Error: {result.get('message', 'Unknown error')}")
            print("❌" * 15)

    except KeyboardInterrupt:
        logger.warning("⚠️ Process interrupted by user")
    except Exception as e:
        logger.error(f"❌ Ingestion failed: {str(e)}", exc_info=True)

if __name__ == "__main__":
    run_test()