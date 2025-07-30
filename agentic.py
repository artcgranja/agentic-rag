from agno.agent import Agent
from agno.models.openrouter import OpenRouter
from agno.tools.reasoning import ReasoningTools
from agno.tools import tool
from langchain_community.vectorstores import PGVector
from langchain_openai import OpenAIEmbeddings
from typing import List, Dict, Any
import os

# Configuração do vector store
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

vector_store = PGVector(
    collection_name="documents",
    connection_string=os.getenv("PGVECTOR_URL"),
    embedding_function=embeddings,
)

@tool
def semantic_search(query: str, k: int = 5, score_threshold: float = 0.7) -> str:
    """Busca semântica na base de conhecimento."""
    
    try:
        docs = vector_store.similarity_search_with_score(query, k=k)
        
        # Filtrar por score
        filtered_docs = [(doc, score) for doc, score in docs if score >= score_threshold]
        
        if not filtered_docs:
            return f"Nenhum documento encontrado com score >= {score_threshold}"
        
        results = []
        for doc, score in filtered_docs:
            results.append({
                "content": doc.page_content[:300] + "...",
                "score": score,
                "metadata": doc.metadata
            })
        
        formatted = "\n".join([
            f"**Score**: {r['score']:.4f} | **Fonte**: {r['metadata'].get('source', 'N/A')}\n"
            f"**Conteúdo**: {r['content']}"
            for r in results
        ])
        
        return formatted
        
    except Exception as e:
        return f"Erro na busca semântica: {str(e)}"

@tool
def similarity_search_with_relevance(query: str, k: int = 3) -> str:
    """Busca com relevance score normalizado."""
    
    try:
        docs = vector_store.similarity_search_with_relevance_scores(query, k=k)
        
        if not docs:
            return "Nenhum documento encontrado."
        
        results = []
        for doc, relevance in docs:
            results.append(
                f"**Relevância**: {relevance:.4f}\n"
                f"**Fonte**: {doc.metadata.get('source', 'N/A')}\n"
                f"**Conteúdo**: {doc.page_content[:400]}..."
            )
        
        return "\n\n".join(results)
        
    except Exception as e:
        return f"Erro na busca por relevância: {str(e)}"

# Agente com múltiplas ferramentas de busca
professor_agent = Agent(
    name="Professor Advanced RAG",
    role="Especialista com Busca Avançada", 
    model=OpenRouter(
        id="meta-llama/llama-4-maverick",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        temperature=0.7,
    ),
    
    tools=[
        semantic_search,
        similarity_search_with_relevance,
        ReasoningTools(add_instructions=True)
    ],
    
    instructions=[
        "Use as ferramentas de busca para encontrar informações relevantes.",
        "Comece com semantic_search para busca geral.",
        "Use similarity_search_with_relevance para busca mais precisa.",
        "Analise os scores de relevância antes de usar as informações.",
        "Combine informações de múltiplas buscas quando necessário.",
        "Cite sempre as fontes e scores de relevância.",
    ],
    
    show_tool_calls=True,
    markdown=True,
)

def main():
    print("🦙 Professor Llama-4 Maverick + LangChain PGVector iniciado!")
    print("Powered by OpenRouter + LangChain")
    print("Digite 'sair' para encerrar\n")
    
    while True:
        user_input = input("💭 Digite sua pergunta: ")
        
        if user_input.lower() in ['sair', 'exit', 'quit']:
            print("👋 Até logo!")
            break
            
        try:
            print("\n🤖 **Resposta do Llama-4 Maverick:**")
            print("-" * 50)
            
            professor_agent.print_response(user_input, stream=True)
            
            print("-" * 50 + "\n")
            
        except Exception as e:
            print(f"❌ Erro: {e}")

if __name__ == "__main__":
    main()