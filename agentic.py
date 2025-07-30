from agno.agent import Agent
from agno.models.openrouter import OpenRouter
from agno.tools.reasoning import ReasoningTools
from agno.tools import tool
from langchain_openai import OpenAIEmbeddings
from langchain_postgres import PGVectorStore
from langchain_postgres import PGEngine
from langchain_core.documents import Document
from dotenv import load_dotenv

import os
import asyncio

load_dotenv()

# Configuração global
DATABASE_URL = os.getenv("DATABASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Inicialização do engine
engine = PGEngine.from_connection_string(DATABASE_URL)

# Configuração do vector store
embeddings = OpenAIEmbeddings()

# Variável global para o vectorstore
vectorstore = None

def setup_vectorstore():
    """Configura o vectorstore de forma síncrona para uso com agno."""
    global vectorstore
    
    def run_async_setup():
        async def async_setup():
            global vectorstore
            
            try:
                print("🔧 Configurando vectorstore...")
                
                # Documentos de exemplo sobre a nerd-o
                documents = [
                    Document(
                        page_content="A nerd-o é uma startup de AI para escolas que desenvolve soluções educacionais inovadoras usando inteligência artificial.",
                        metadata={"source": "sobre_empresa", "categoria": "institucional", "tipo": "descrição"}
                    ),
                    Document(
                        page_content="A nerd-o tem 3 sócios principais: Arthur, Rossetto e Gordon, que trabalham juntos no desenvolvimento de tecnologias educacionais avançadas.",
                        metadata={"source": "equipe", "categoria": "pessoas", "tipo": "fundadores"}
                    ),
                    Document(
                        page_content="A empresa foca em inteligência artificial aplicada à educação, criando ferramentas personalizadas para melhorar o aprendizado dos estudantes.",
                        metadata={"source": "missao", "categoria": "tecnologia", "tipo": "objetivos"}
                    ),
                    Document(
                        page_content="As soluções da nerd-o incluem sistemas de recomendação personalizados, análise de desempenho estudantil e ferramentas de ensino adaptativo.",
                        metadata={"source": "produtos", "categoria": "servicos", "tipo": "ofertas"}
                    ),
                    Document(
                        page_content="A nerd-o utiliza machine learning e processamento de linguagem natural para criar experiências de aprendizado personalizadas para cada estudante.",
                        metadata={"source": "tecnologia", "categoria": "ml", "tipo": "metodologia"}
                    ),
                    Document(
                        page_content="Os fundadores da nerd-o têm experiência combinada em educação, tecnologia e pesquisa, trazendo uma visão única para o mercado educacional.",
                        metadata={"source": "fundadores", "categoria": "background", "tipo": "experiencia"}
                    ),
                    Document(
                        page_content="A missão da nerd-o é democratizar o acesso a educação de qualidade através de tecnologias de inteligência artificial acessíveis e eficazes.",
                        metadata={"source": "missao", "categoria": "valores", "tipo": "propósito"}
                    ),
                    Document(
                        page_content="A nerd-o desenvolve plataformas que analisam o progresso individual dos alunos e sugerem caminhos de aprendizado otimizados.",
                        metadata={"source": "funcionalidades", "categoria": "features", "tipo": "capacidades"}
                    )
                ]
                
                # Criar vectorstore com documentos
                vectorstore = await PGVectorStore.afrom_documents(
                    documents=documents,
                    embedding=embeddings,
                    engine=engine,
                    table_name="document_vectors",
                    schema_name="public"
                )
                
                print(f"✅ VectorStore configurado com {len(documents)} documentos!")
                return True
                
            except Exception as e:
                print(f"❌ Erro ao configurar vectorstore: {str(e)}")
                return False
        
        # Executar setup assíncrono
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(async_setup())
        finally:
            loop.close()
    
    return run_async_setup()

# Configurar vectorstore na inicialização
setup_success = setup_vectorstore()
if not setup_success:
    print("❌ Falha na configuração do vectorstore!")
    exit(1)

@tool
def semantic_search(query: str, k: int = 5) -> str:
    """Busca semântica na base de conhecimento sobre a nerd-o."""
    
    try:
        # Buscar com scores
        docs_with_scores = vectorstore.similarity_search_with_score(query, k=k)
        
        if not docs_with_scores:
            return "Nenhum documento encontrado na base de conhecimento."
        
        results = []
        for doc, distance in docs_with_scores:
            # Converter distância para similaridade (0-1, onde 1 é mais similar)
            similarity = max(0, 1 - distance)
            
            results.append({
                "content": doc.page_content,
                "similarity": similarity,
                "source": doc.metadata.get('source', 'N/A'),
                "categoria": doc.metadata.get('categoria', 'N/A'),
                "tipo": doc.metadata.get('tipo', 'N/A')
            })
        
        # Formatar resultados
        formatted_results = []
        for i, result in enumerate(results, 1):
            formatted_results.append(
                f"📄 **Resultado {i}** (Similaridade: {result['similarity']:.3f})\n"
                f"**Fonte**: {result['source']} | **Categoria**: {result['categoria']} | **Tipo**: {result['tipo']}\n"
                f"**Conteúdo**: {result['content']}\n"
            )
        
        return "\n".join(formatted_results)
        
    except Exception as e:
        return f"❌ Erro na busca semântica: {str(e)}"

@tool
def filtered_search(query: str, categoria: str = None, tipo: str = None, k: int = 3) -> str:
    """Busca semântica com filtros por categoria ou tipo."""
    
    try:
        # Criar filtro baseado nos parâmetros
        filter_dict = {}
        if categoria:
            filter_dict["categoria"] = categoria
        if tipo:
            filter_dict["tipo"] = tipo
        
        # Buscar com filtro
        if filter_dict:
            docs = vectorstore.similarity_search(query, k=k, filter=filter_dict)
        else:
            docs = vectorstore.similarity_search(query, k=k)
        
        if not docs:
            filter_info = f" com filtros {filter_dict}" if filter_dict else ""
            return f"Nenhum documento encontrado{filter_info}."
        
        results = []
        for i, doc in enumerate(docs, 1):
            results.append(
                f"📋 **Documento {i}**\n"
                f"**Fonte**: {doc.metadata.get('source', 'N/A')} | "
                f"**Categoria**: {doc.metadata.get('categoria', 'N/A')} | "
                f"**Tipo**: {doc.metadata.get('tipo', 'N/A')}\n"
                f"**Conteúdo**: {doc.page_content}\n"
            )
        
        filter_info = f"\n🔍 **Filtros aplicados**: {filter_dict}" if filter_dict else ""
        return "\n".join(results) + filter_info
        
    except Exception as e:
        return f"❌ Erro na busca filtrada: {str(e)}"

@tool
def list_available_categories() -> str:
    """Lista as categorias e tipos disponíveis na base de conhecimento."""
    
    try:
        # Buscar todos os documentos para extrair metadados únicos
        all_docs = vectorstore.similarity_search("", k=50)  # Busca ampla
        
        categories = set()
        types = set()
        sources = set()
        
        for doc in all_docs:
            if 'categoria' in doc.metadata:
                categories.add(doc.metadata['categoria'])
            if 'tipo' in doc.metadata:
                types.add(doc.metadata['tipo'])
            if 'source' in doc.metadata:
                sources.add(doc.metadata['source'])
        
        result = "📊 **Metadados disponíveis na base de conhecimento:**\n\n"
        result += f"**Categorias**: {', '.join(sorted(categories))}\n"
        result += f"**Tipos**: {', '.join(sorted(types))}\n" 
        result += f"**Fontes**: {', '.join(sorted(sources))}\n"
        result += f"\n**Total de documentos**: {len(all_docs)}"
        
        return result
        
    except Exception as e:
        return f"❌ Erro ao listar categorias: {str(e)}"

# Agente com múltiplas ferramentas de busca
professor_agent = Agent(
    name="Professor nerd-o RAG",
    role="Especialista em informações sobre a nerd-o com busca avançada", 
    model=OpenRouter(
        id="google/gemini-2.5-flash",
        api_key=OPENROUTER_API_KEY,
        temperature=0.7,
    ),
    
    tools=[
        semantic_search,
        filtered_search,
        list_available_categories,
        ReasoningTools(add_instructions=True)
    ],
    
    instructions=[
        "Você é um especialista em informações sobre a startup nerd-o.",
        "Use as ferramentas de busca para encontrar informações precisas na base de conhecimento.",
        "Comece com semantic_search para buscas gerais sobre a nerd-o.",
        "Use filtered_search quando precisar de informações específicas por categoria ou tipo.",
        "Use list_available_categories para explorar que tipos de informações estão disponíveis.",
        "Sempre cite as fontes e scores de similaridade quando disponíveis.",
        "Se não encontrar informações suficientes, seja honesto sobre as limitações.",
        "Combine informações de múltiplas buscas quando necessário para dar respostas completas.",
        "Priorize informações com maior similaridade/relevância.",
    ],
    
    show_tool_calls=True,
    markdown=True,
)

def main():
    print("🤖 Professor nerd-o RAG + Vector Database iniciado!")
    print("Powered by OpenRouter + LangChain PGVector")
    print("Base de conhecimento carregada com informações sobre a nerd-o")
    print("Digite 'sair' para encerrar\n")
    
    while True:
        user_input = input("💭 Pergunte sobre a nerd-o: ")
        
        if user_input.lower() in ['sair', 'exit', 'quit']:
            print("👋 Até logo!")
            break
            
        try:
            print("\n🤖 **Resposta do Professor nerd-o:**")
            print("-" * 50)
            
            professor_agent.print_response(user_input, stream=True)
            
            print("-" * 50 + "\n")
            
        except Exception as e:
            print(f"❌ Erro: {e}")

if __name__ == "__main__":
    main()