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
    show_tool_calls=True,
    markdown=True,
)

def ask_generic(texto):
    print("🤖 Resposta Genérica:")
    response_stream = professor_agent.run(message=texto, stream=True, stream_intermediate_steps=True)
    for response in response_stream:
        print(response.event)
        print(response.content)
        if "ToolCall" in response.event:
            print(response.tool)
        print("-" * 50 + "\n")

if __name__ == "__main__":
    ask_generic("O que é a Nerd-o?")