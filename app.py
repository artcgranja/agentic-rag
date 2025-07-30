import streamlit as st
from agno.agent import Agent, AgentKnowledge
from agno.embedder.openai import OpenAIEmbedder
from agno.vectordb.pgvector import PgVector, SearchType
from agno.models.openrouter import OpenRouter
from agno.tools.reasoning import ReasoningTools
from dotenv import load_dotenv
import os

load_dotenv()

st.set_page_config(page_title="Chat Agno RAG", page_icon="🤖")

@st.cache_resource
def get_agent():
    embedder = OpenAIEmbedder(id="text-embedding-3-large")
    vector_db = PgVector(
        table_name="agent_knowledge",
        schema="ai",
        db_url=os.getenv("DATABASE_URL"),
        search_type=SearchType.hybrid,
        embedder=embedder,
        auto_upgrade_schema=True
    )
    
    return Agent(
        name="Professor Nerd-o RAG",
        role="Especialista em Nerd-o",
        model=OpenRouter(
            id="google/gemini-2.5-flash", 
            api_key=os.getenv("OPENROUTER_API_KEY"), 
            temperature=0.7
        ),
        knowledge=AgentKnowledge(vector_db=vector_db),
#        tools=[ReasoningTools(add_instructions=True)],
        show_tool_calls=True,
        markdown=True,
    )

@st.fragment
def show_live_event(event_type, message):
    """Fragment para mostrar eventos em tempo real"""
    if event_type == "info":
        st.info(message)
    elif event_type == "success":
        st.success(message)
    elif event_type == "warning":
        st.warning(message)

def stream_chat(agent, message):
    response_text = ""
    
    # Placeholder para resposta final
    response_placeholder = st.empty()
    
    try:
        # Stream de eventos
        response_stream = agent.run(message=message, stream=True, stream_intermediate_steps=True)
        
        for response in response_stream:
            
            # 1. Tool iniciada - criar elemento imediatamente
            if response.event == "ToolCallStartedEvent":
                tool_name = getattr(response, 'tool', {}).tool_name if hasattr(response, 'tool') else "Tool"
                
                if tool_name == "search_knowledge_base":
                    st.info("🔍 Buscando na base de conhecimento...")
                elif tool_name == "think":
                    st.info("🧠 Pensando sobre a questão...")
                else:
                    st.info(f"🔧 Executando: {tool_name}...")
            
            # 2. Tool concluída
            elif response.event == "ToolCallCompletedEvent":
                tool_name = getattr(response, 'tool', {}).tool_name if hasattr(response, 'tool') else "Tool"
                
                if tool_name == "search_knowledge_base":
                    st.success("✅ Documentos encontrados na base de conhecimento")
                elif tool_name == "think":
                    st.success("💭 Análise concluída")
                else:
                    st.success(f"✅ {tool_name} concluída")
            
            # 3. Reasoning step
            elif response.event == "ReasoningStepEvent":
                step = getattr(response, 'content', None)
                if hasattr(step, 'title') and hasattr(step, 'confidence'):
                    confidence_emoji = "🎯" if step.confidence >= 0.9 else "🤔" if step.confidence >= 0.7 else "❓"
                    st.info(f"{confidence_emoji} **{step.title}** (Confiança: {step.confidence})")
            
            # 4. Reasoning completo
            elif response.event == "ReasoningCompleted":
                st.success("✨ Raciocínio concluído - gerando resposta...")
            
            # 5. Conteúdo da resposta em streaming
            elif response.event == "RunResponseContent":
                content = getattr(response, 'content', '')
                if content:
                    response_text += content
                    response_placeholder.markdown(response_text)
            
            # 6. Resposta final
            elif response.event == "RunCompleted":
                # Garantir resposta final
                final_content = getattr(response, 'content', '')
                if final_content and final_content.strip():
                    response_placeholder.markdown(final_content)
                    response_text = final_content
                elif response_text.strip():
                    response_placeholder.markdown(response_text)
                else:
                    response_placeholder.warning("⚠️ Resposta vazia recebida")
                
                # Mostrar fontes se disponíveis
                extra_data = getattr(response, 'extra_data', None)
                if extra_data and hasattr(extra_data, 'references') and extra_data.references:
                    with st.expander("📚 Fontes consultadas", expanded=False):
                        for ref in extra_data.references:
                            st.markdown(f"**Busca:** `{ref.query}`")
                            st.markdown(f"**{len(ref.references)} documentos encontrados**")
                            
                            for i, doc in enumerate(ref.references[:2], 1):
                                st.markdown(f"📄 **{doc['name']}** (Página {doc['meta_data']['page']})")
                                preview = doc['content'][:200] + "..." if len(doc['content']) > 200 else doc['content']
                                st.text(preview)
                                if i < min(2, len(ref.references)):
                                    st.divider()
                break
        
        return response_text
    
    except Exception as e:
        st.error(f"❌ Erro durante execução: {str(e)}")
        return None

# Interface principal
st.title("🤖 Chat Agno RAG")
st.markdown("*Sistema inteligente com busca RAG e reasoning*")

agent = get_agent()

# Sidebar
with st.sidebar:
    st.header("⚙️ Controles")
    
    if st.button("🧹 Limpar Chat"):
        st.session_state.messages = []
        st.rerun()
    
    st.divider()
    st.markdown("### 📊 Status")
    st.markdown("✅ Agente carregado")
    st.markdown("✅ RAG ativo")
    st.markdown("✅ Reasoning habilitado")
    
    st.divider()
    st.markdown("### 💡 Dica")
    st.markdown("Observe o processo de reasoning em tempo real!")

# Inicializar mensagens
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostrar histórico de mensagens
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input do usuário
if prompt := st.chat_input("Digite sua pergunta sobre Nerd-o..."):
    # Adicionar mensagem do usuário
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Gerar resposta do assistente
    with st.chat_message("assistant"):
        response = stream_chat(agent, prompt)
        
        if response and response.strip():
            st.session_state.messages.append({"role": "assistant", "content": response})