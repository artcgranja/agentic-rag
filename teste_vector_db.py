from agno.agent import Agent
from agno.models.openrouter import OpenRouter
from agno.tools.reasoning import ReasoningTools
from agno.tools import tool
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVectorStore
from langchain_postgres import PGEngine
from langchain_core.documents import Document
from dotenv import load_dotenv

from typing import List, Dict, Any
import os
import asyncio

load_dotenv()

# Inicialização global do engine
engine = PGEngine.from_connection_string(
    os.getenv("DATABASE_URL"),
)

# Configuração do vector store
embeddings = OpenAIEmbeddings()

# Variável global para o vectorstore
vectorstore = None

async def initialize_vectorstore():
    """Inicializa a tabela e o vectorstore de forma assíncrona."""
    global vectorstore
    
    try:
        # 1. Primeiro, inicializar a tabela com a estrutura correta
        print("Criando tabela document_vectors...")
        await engine.ainit_vectorstore_table(
            table_name="document_vectors",
            vector_size=1536,
            overwrite_existing=True,
            content_column="content",
            embedding_column="embedding", 
            metadata_json_column="metadata",
            id_column="langchain_id"
        )
        print("Tabela criada com sucesso!")
        
        # 2. Criar vectorstore
        print("Inicializando VectorStore...")
        vectorstore = await PGVectorStore.create(
            engine=engine,
            embedding_service=embeddings,
            table_name="document_vectors",
        )
        
        print("VectorStore inicializado com sucesso!")
        return True
        
    except Exception as e:
        print(f"Erro ao inicializar vectorstore: {str(e)}")
        return False

async def add_sample_documents():
    """Adiciona documentos de exemplo."""
    documents = [
        Document(
            page_content="A nerd-o é uma startup de AI para escolas que desenvolve soluções educacionais inovadoras.",
            metadata={"source": "sobre_empresa", "categoria": "institucional"}
        ),
        Document(
            page_content="A nerd-o tem 3 sócios: Arthur, Rossetto e Gordon, que trabalham juntos no desenvolvimento de tecnologias educacionais.",
            metadata={"source": "equipe", "categoria": "pessoas"}
        ),
        Document(
            page_content="A empresa foca em inteligência artificial aplicada à educação, criando ferramentas para melhorar o aprendizado.",
            metadata={"source": "missao", "categoria": "tecnologia"}
        ),
        Document(
            page_content="As soluções da nerd-o incluem sistemas de recomendação personalizados e análise de desempenho estudantil.",
            metadata={"source": "produtos", "categoria": "servicos"}
        )
    ]
    
    try:
        print(f"Adicionando {len(documents)} documentos...")
        ids = await vectorstore.aadd_documents(documents)
        print(f"Documentos adicionados com IDs: {ids}")
        return True
    except Exception as e:
        print(f"Erro ao adicionar documentos: {str(e)}")
        return False

async def semantic_search(query: str, k: int = 5, score_threshold: float = 0.7) -> str:
    """Busca semântica na base de conhecimento (versão assíncrona)."""
    
    try:
        # Usar versão assíncrona da busca com score
        docs_with_scores = await vectorstore.asimilarity_search_with_score(query, k=k)
        
        if not docs_with_scores:
            return "Nenhum documento encontrado."
        
        # Filtrar por score (menor score = mais similar)
        filtered_docs = [(doc, score) for doc, score in docs_with_scores if score <= (1.0 - score_threshold)]
        
        if not filtered_docs:
            return f"Nenhum documento encontrado com relevância suficiente (threshold: {score_threshold})"
        
        results = []
        for doc, score in filtered_docs:
            similarity = 1.0 - score  # Converter distância para similaridade
            results.append({
                "content": doc.page_content,
                "similarity": similarity,
                "metadata": doc.metadata
            })
        
        # Formatar resultado
        formatted = f"🔍 **Busca**: {query}\n\n"
        formatted += "\n\n".join([
            f"**Similaridade**: {r['similarity']:.4f} | **Fonte**: {r['metadata'].get('source', 'N/A')} | **Categoria**: {r['metadata'].get('categoria', 'N/A')}\n"
            f"📄 **Conteúdo**: {r['content']}"
            for r in results
        ])
        
        return formatted
        
    except Exception as e:
        print(f"Erro na busca semântica: {str(e)}")
        return f"Erro na busca: {str(e)}"

async def test_multiple_queries():
    """Testa várias consultas diferentes."""
    queries = [
        "O que é a nerd-o?",
        "Quem são os sócios da empresa?",
        "Que tipo de tecnologia a empresa desenvolve?",
        "Quais produtos a nerd-o oferece?",
        "Como a IA é aplicada na educação?"
    ]
    
    for query in queries:
        print(f"\n{'='*60}")
        result = await semantic_search(query, k=3, score_threshold=0.5)
        print(result)
        print(f"{'='*60}")

async def main():
    """Função principal assíncrona."""
    try:
        print("🚀 Iniciando setup do VectorStore...")
        
        # 1. Inicializar vectorstore
        if not await initialize_vectorstore():
            return
        
        # 2. Adicionar documentos de exemplo
        if not await add_sample_documents():
            return
        
        # 3. Testar buscas
        print("\n🧪 Iniciando testes de busca...")
        await test_multiple_queries()
        
        print("\n✅ Testes concluídos com sucesso!")
        
    except Exception as e:
        print(f"❌ Erro na execução: {str(e)}")
    
    finally:
        # Fechar conexão
        await engine.close()
        print("\n🔌 Conexão fechada.")

if __name__ == "__main__":
    # Executar função assíncrona
    asyncio.run(main())