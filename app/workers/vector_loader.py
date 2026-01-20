from time import sleep
import logging

from app.core.qa_engine import reload_retriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("🧠 Background worker started")
    sleep(5)
    reload_retriever()
    logger.info("✅ Vector store loaded")
