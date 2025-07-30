from agno.agent import Agent
from agno.embedder.openai import OpenAIEmbedder
from agno.knowledge.markdown import MarkdownKnowledgeBase, MarkdownReader 
import os
from agno.document.chunking.semantic import SemanticChunking
from agno.vectordb.pgvector import PgVector, SearchType
from agno.models.openrouter import OpenRouter
from dotenv import load_dotenv
import os

load_dotenv()

db_url = os.getenv("DATABASE_URL")

pdf_dir = os.path.join(os.path.dirname(__file__), "pdfs")

# 1) Embedding
embedder = OpenAIEmbedder(id="text-embedding-3-large")

# 2) Vector DB with hybrid retrieval
vector_db = PgVector(
    table_name="agent_knowledge",
    db_url=db_url,
    search_type=SearchType.hybrid,
    embedder=embedder,
    auto_upgrade_schema=False
)

# 3) Load documents with semantic chunks
knowledge_base = MarkdownKnowledgeBase(
    reader=MarkdownReader(),
    vector_db=vector_db,
    chunker=SemanticChunking(chunk_size=1000, similarity_threshold=0.5),
)
knowledge_base.load_document(
    path=pdf_dir,
    metadata={"user_id": "jordan_mitchell", "document_type": "cv", "year": 2025},
    recreate=True,  # Set to True only for the first run, then set to False
)