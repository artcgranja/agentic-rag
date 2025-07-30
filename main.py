from agno.agent import Agent, AgentKnowledge
from agno.embedder.openai import OpenAIEmbedder
from agno.vectordb.pgvector import PgVector, SearchType
from agno.models.openrouter import OpenRouter
from agno.tools.reasoning import ReasoningTools
from dotenv import load_dotenv
import os

load_dotenv()
db_url  = os.getenv("DATABASE_URL")
api_key = os.getenv("OPENROUTER_API_KEY")

embedder = OpenAIEmbedder(id="text-embedding-3-large")

vector_db = PgVector(
    table_name="agent_knowledge",
    schema="ai",
    db_url=db_url,
    search_type=SearchType.hybrid,
    embedder=embedder,
    auto_upgrade_schema=True
)

rag_agent = AgentKnowledge(
    vector_db=vector_db,
)

professor_agent = Agent(
    name="Professor nerd‑o RAG",
    role="Especialista em Nerd‑o com busca avançada",
    model=OpenRouter(id="google/gemini-2.5-flash", api_key=api_key, temperature=0.7),
    knowledge=rag_agent,
    tools=[ReasoningTools(add_instructions=True)],
    show_tool_calls=True,
    markdown=True,
)

def pergunta_genérica(texto):
    print("🤖 Resposta Genérica:")
    professor_agent.print_response(texto, stream=True)

if __name__ == "__main__":
    pergunta_genérica("O que é a Nerd-o?")