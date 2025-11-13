import streamlit as st

from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from sagebot.loader import *
from sagebot.rag import *
from sagebot.progress import *
from sagebot.work_rag import *
from sagebot.work_rag import carregar_indice_existente
import os
os.environ.setdefault("USER_AGENT", "SageBot/1.0 (Streamlit)")




TIPOS_DE_ARQUIVOS =[
    'url','.md','.pdf'
]

CONFIG_MODELOS ={'Groq': {'modelos': ['llama-3.1-8b-instant', 'llama-3.3-70b-versatile'],
                          'chat': ChatGroq},
                  'OpenAI': {'modelos': [ 'gpt-4o', 'gpt-4.1-nano'],
                             'chat': ChatOpenAI}}

if "memoria" not in st.session_state:
   st.session_state["memoria"] = []

MENSAGENS_EXEMPLOS =[
    ("ASSISTANT", "Olá, como você está? Estou aqui para lhe ajudar sobre duvidas da documentação da AWS"),
]

def carrega_arquivos(tipo_arquivos,arquivo):
    if tipo_arquivos == '.md':
        if not arquivo:
            st.error("Envie pelo menos um .md")
            st.stop()
        documento = carrega_md(arquivo)  
        return documento

    if tipo_arquivos == '.pdf':
        paths = salvar_tmp(arquivo, '.pdf')
        if not paths:
            st.error("Envie pelo menos um PDF.")
            st.stop()
        docs = [carrega_pdf(p) for p in paths]
        documento = "\n\n".join(docs)

    if tipo_arquivos == 'url':
        documento=carrega_site(arquivo)

    return documento

def setup_chain(provedor, modelo, api_key):
    system_message = """
    Você é um assistente técnico especializado em AWS chamado SageBot.
    Responda com precisão, didática e cite fontes oficiais quando possível.
    Baseie-se estritamente no CONTEXTO fornecido; se faltar informação, diga o que falta e sugira fontes (docs.aws.amazon.com).
    """
    st.session_state['system_message'] = system_message

    template = ChatPromptTemplate.from_messages([
        ('system', '{system_message}'),
        MessagesPlaceholder('chat_history'),
        ('human', "Contexto:\n```\n{context}\n```\n\nPergunta: {input}")
    ])

    if provedor == "Groq":
        chat = CONFIG_MODELOS[provedor]['chat'](model=modelo, groq_api_key=api_key)
    else:
        chat = CONFIG_MODELOS[provedor]['chat'](model=modelo, api_key=api_key)

    st.session_state['chain'] = template | chat


def carrega_modelo(provedor, modelo, api_key, tipo_arquivo,arquivo,embed_model, dims, k):

    documento=carrega_arquivos(tipo_arquivo,arquivo)

    system_message = """
    Você é um assistente técnico especializado em AWS chamado SageBot.
    Responda com precisão, didática e cite fontes oficiais quando possível.
    Baseie-se estritamente no CONTEXTO fornecido; se faltar informação, diga o que falta e sugira fontes (docs.aws.amazon.com).
    """
    st.session_state['system_message'] = system_message

    template = ChatPromptTemplate.from_messages([
        ('system', '{system_message}'),
        MessagesPlaceholder('chat_history'),
        ('human',
        "Contexto:\n```\n{context}\n```\n\nPergunta: {input}")
    ])

    if provedor == "Groq":
        chat = CONFIG_MODELOS[provedor]['chat'](model=modelo, groq_api_key=api_key)
    else:
        chat = CONFIG_MODELOS[provedor]['chat'](model=modelo, api_key=api_key)
    chain = template | chat
    st.session_state['chain'] = chain

    iniciar_async(documento, embed_model=embed_model, dims=dims, k=k)


def tail_messages(messages, max_pairs=6):
    return messages[-(max_pairs*2):]


def pagina_chat():
    st.header("🤖 SageBot", divider=True)

    chain = st.session_state.get('chain')
    if chain is None:
        st.info('Inicialize o SageBot na barra lateral.')
        st.stop()

    render_status()

    mensagens = st.session_state.get('memoria', [])
    for mensagem in mensagens:
      chat = st.chat_message(mensagem.type)
      chat.markdown(mensagem.content)
    
    
    input_usuario = st.chat_input('Digite sua duvida:')
    if not input_usuario:
        if st.session_state.pop("__needs_rerun", False):
            st.rerun()
        return
    
    chat = st.chat_message("human")
    chat.markdown(input_usuario)

    retriever = st.session_state.get("retriever",None)
    contexto = "RAG ainda inicializando. Responda de forma geral e cite fontes oficiais."

    if retriever is not None:
        try:
            docs = retriever.invoke(input_usuario)
            contexto = "\n\n".join(d.page_content for d in docs) if docs else "N/A"
        except Exception as e:
            st.warning(f"Não foi possível consultar o retriever ainda: {e}")
        
    

    hist = tail_messages(st.session_state['memoria'], max_pairs=6)
    ai = st.chat_message("ai")

    resposta = ai.write_stream(
        chain.stream({
            'system_message': st.session_state.get('system_message',''),
            'context': contexto,
            'input': input_usuario, 
            'chat_history': hist
            }))

    st.session_state['memoria'].append(HumanMessage(content=input_usuario))
    st.session_state['memoria'].append(AIMessage(content=resposta))

    if st.session_state.pop("__needs_rerun", False):
        st.rerun()    

def ui_modelos():
    st.header("Seleção de Modelos", divider=True)
    provedor = st.selectbox(
        "Selecione o provedor dos modelos:",
        list(CONFIG_MODELOS.keys()),
        key="prov_select"
    )
    modelo = st.selectbox(
        "Selecione o modelo específico:",
        CONFIG_MODELOS[provedor]['modelos'],
        key="modelo_select"
    )
    api_key = st.text_input(
        f"Insira sua API Key para o provedor {provedor}: ",
        value=st.session_state.get(f'api_key_{provedor}', ''),
        type="password",
        key="prov_api_key"
    )
    st.session_state[f'api_key_{provedor}'] = api_key

    st.markdown("### Embeddings")
    embed_model = st.selectbox(
        "Modelo de embeddings (OpenAI)",
        ["text-embedding-3-small", "text-embedding-3-large"],
        index=0,
        help="small: mais barato/rápido; large: mais qualidade.",
        key="embed_model_select"
    )
    dims = None
    k = st.slider("Top-K do Retriever", 1, 10, 4, key="k_slider")

    openai_key = st.text_input("OPENAI_API_KEY", type="password", key="openai_key_input")
    if openai_key:
        os.environ["OPENAI_API_KEY"] = openai_key

    return provedor, modelo, api_key, embed_model, dims, k


def ui_fonte_dados():
    """Retorna (modo_fonte, tipo_arquivo, arquivo, indice_escolhido)"""
    from work_rag import listar_indices_existentes

    st.header("Fonte de dados", divider=True)
    modo_fonte = st.radio(
        "Como deseja usar a base?",
        ["Usar índice existente", "Indexar novos arquivos/URL"],
        index=1,
        key="modo_fonte_radio"
    )

    tipo_arquivo, arquivo, indice_escolhido = None, None, None

    if modo_fonte == "Usar índice existente":
        indices = listar_indices_existentes()
        if not indices:
            st.warning("Nenhum índice encontrado em data/index/. Faça uma indexação primeiro.")
        indice_escolhido = st.selectbox(
            "Selecione um índice persistido:",
            indices if indices else ["—"],
            disabled=not indices,
            key="indice_existente_select"
        )
    else:
        st.markdown("### Upload / Fonte de Dados")
        tipo_arquivo = st.selectbox(
            "Selecione o tipo de arquivo:",
            TIPOS_DE_ARQUIVOS,
            key="tipo_arquivo_select"
        )
        if tipo_arquivo == 'url':
            arquivo = st.text_input(
                "Insira a URL:",
                placeholder="ex: https://docs.aws.amazon.com/sagemaker/latest/dg/whatis.html",
                key="url_input"
            )
            if arquivo:
                st.success("✅ URL recebida com sucesso!")
        elif tipo_arquivo == '.md':
            arquivo = st.file_uploader(
                "Faça upload de arquivos Markdown (.md)",
                type=['md'],
                accept_multiple_files=True,
                key="md_uploader"
            )
            if arquivo:
                st.success(f"✅ {len(arquivo)} arquivo(s) .md carregado(s)!")
        elif tipo_arquivo == '.pdf':
            arquivo = st.file_uploader(
                "Faça upload de arquivos PDF",
                type=['pdf'],
                accept_multiple_files=True,
                key="pdf_uploader"
            )
            if arquivo:
                st.success(f"✅ {len(arquivo)} arquivo(s) PDF carregado(s)!")

    return modo_fonte, tipo_arquivo, arquivo, indice_escolhido



    st.subheader("LLM & Embeddings")

    indice_escolhido = None
    tipo_arquivo = None
    arquivo = None
    modo_fonte = "Indexar novos arquivos/URL"

    tabs =st.tabs(('Upload de Arquivos','Seleção de Modelos'))

    with tabs[0]:
        st.header("Upload de Arquivos", divider=True)
        tipo_arquivo = st.selectbox("Selecione o tipo de arquivo:", TIPOS_DE_ARQUIVOS)
        if tipo_arquivo == 'url':
            arquivo = st.text_input("Insira a URL:")
            if arquivo:
                st.success("url recebida com sucesso!")
        if tipo_arquivo == '.md':
            arquivo = st.file_uploader("Faça o upload dos seus arquivos:", type=['md'], accept_multiple_files=True)
            if arquivo:
                st.success("arquivos carregados com sucesso!")
        if tipo_arquivo == '.pdf':
            arquivo = st.file_uploader("Faça o upload dos seus arquivos:", type=['pdf'], accept_multiple_files=True)
            if arquivo:
                st.success("arquivos carregados com sucesso!")
        

    with tabs[1]:
        st.header("Seleção de Modelos", divider=True)
        provedor = st.selectbox("Selecione o provedor dos modelos:", list(CONFIG_MODELOS.keys()))
        modelo= st.selectbox("Selecione o modelo específico:", CONFIG_MODELOS[provedor]['modelos'])
        api_key = st.text_input(
            f"Insira sua API Key para o provedor {provedor}: ",
            value=st.session_state.get(f'api_key_{provedor}',''),
            type="password")
        st.session_state[f'api_key_{provedor}'] = api_key
    
    
    st.markdown("### Embeddings")
    embed_model = st.selectbox(
        "Modelo de embeddings (OpenAI)",
        ["text-embedding-3-small", "text-embedding-3-large"],
        index=0,
        help="small: mais barato/rápido; large: mais qualidade (custa mais)."
    )
    dims = None  
    k = st.slider("Top-K do Retriever", 1, 10, 4)

    openai_key = st.text_input("OPENAI_API_KEY", type="password")
    if openai_key:
        os.environ["OPENAI_API_KEY"] = openai_key


    st.markdown("### Fonte de dados")
    modo_fonte = st.radio("Como deseja usar a base?", ["Usar índice existente", "Indexar novos arquivos/URL"], index=1)

    
    arquivo = None
    tipo_arquivo = None

    if modo_fonte == "Indexar novos arquivos/URL":
        st.markdown("### Upload / Fonte de Dados")
        tipo_arquivo = st.selectbox("Selecione o tipo de arquivo:", TIPOS_DE_ARQUIVOS)

        if tipo_arquivo == 'url':
            arquivo = st.text_input("Insira a URL:")
            if arquivo:
                st.success("✅ URL recebida com sucesso!")
        elif tipo_arquivo == '.md':
            arquivo = st.file_uploader("Faça upload de arquivos Markdown (.md)", type=['md'], accept_multiple_files=True)
            if arquivo:
                st.success(f"✅ {len(arquivo)} arquivo(s) .md carregado(s) com sucesso!")
        elif tipo_arquivo == '.pdf':
            arquivo = st.file_uploader("Faça upload de arquivos PDF", type=['pdf'], accept_multiple_files=True)
            if arquivo:
                st.success(f"✅ {len(arquivo)} arquivo(s) PDF carregado(s) com sucesso!")


   

    col1, col2 = st.columns(2)
    with col1:

        can_init = False

    if modo_fonte == "Usar índice existente":
        can_init = carregar_indice_existente is not None
    else:
        # exige entrada válida dependendo do tipo de arquivo
        if tipo_arquivo == 'url' and arquivo:
            can_init = True
        elif tipo_arquivo in ['.md', '.pdf'] and arquivo:
            can_init = True

    btn = st.button("Inicializar SageBot", use_container_width=True, disabled=not can_init)

    if btn:
        if not api_key:
            st.error("Por favor, insira a API Key antes de carregar o SageBot.")
            st.stop()

        if modo_fonte == "Usar índice existente":
            carregar_indice_existente(
                h=indice_escolhido,
                embed_model=embed_model, dims=dims, k=k, compressed=True
            )
            setup_chain(provedor, modelo, api_key)
            st.session_state["__needs_rerun"] = True
            st.success(f"Índice {indice_escolhido} carregado com sucesso!")
        else:
            carrega_modelo(provedor, modelo, api_key, tipo_arquivo, arquivo, embed_model, dims, k)
            st.success("SageBot carregado com sucesso!")

    with col2:
        if st.button('Apagar Histórico de Conversa', use_container_width=True):
            st.session_state['memoria'] = []

def sidebar():
    st.subheader("LLM & Embeddings")

    with st.expander("Seleção de Modelos", expanded=True):
        provedor = st.selectbox(
            "Selecione o provedor dos modelos:",
            list(CONFIG_MODELOS.keys()),
            key="prov_select"
        )
        modelo = st.selectbox(
            "Selecione o modelo específico:",
            CONFIG_MODELOS[provedor]['modelos'],
            key="modelo_select"
        )
        api_key = st.text_input(
            f"Insira sua API Key para o provedor {provedor}: ",
            value=st.session_state.get(f'api_key_{provedor}', ''),
            type="password",
            key="prov_api_key"
        )
        st.session_state[f'api_key_{provedor}'] = api_key

        st.markdown("### Embeddings")
        embed_model = st.selectbox(
            "Modelo de embeddings (OpenAI)",
            ["text-embedding-3-small", "text-embedding-3-large"],
            index=0,
            help="small: mais barato/rápido; large: mais qualidade (custa mais).",
            key="embed_model_select"
        )
        dims = None
        k = st.slider("Top-K do Retriever", 1, 10, 4, key="k_slider")

        openai_key = st.text_input("OPENAI_API_KEY (OpenAI)", type="password", key="openai_key_input")
        if openai_key:
            os.environ["OPENAI_API_KEY"] = openai_key


    st.markdown("### Fonte de dados")
    modo_fonte = st.radio(
        "Como deseja usar a base?",
        ["Usar índice existente", "Indexar novos arquivos/URL"],
        index=1,
        key="modo_fonte_radio"
    )

    indice_escolhido = None
    tipo_arquivo = None
    arquivo = None

    if modo_fonte == "Usar índice existente":
        from work_rag import listar_indices_existentes
        indices = listar_indices_existentes()
        if not indices:
            st.warning("Nenhum índice encontrado em data/index/. Faça uma indexação primeiro.")
        indice_escolhido = st.selectbox(
            "Selecione um índice persistido:",
            indices if indices else ["—"],
            disabled=not indices,
            key="indice_existente_select"
        )
    else:
        st.markdown("### Upload / Fonte de Dados")
        tipo_arquivo = st.selectbox(
            "Selecione o tipo de arquivo:",
            TIPOS_DE_ARQUIVOS,
            key="tipo_arquivo_select"
        )
        if tipo_arquivo == 'url':
            arquivo = st.text_input(
                "Insira a URL:",
                placeholder="ex: https://docs.aws.amazon.com/sagemaker/latest/dg/whatis.html",
                key="url_input"
            )
            if arquivo:
                st.success("✅ URL recebida com sucesso!")
        elif tipo_arquivo == '.md':
            arquivo = st.file_uploader(
                "Faça upload de arquivos Markdown (.md)",
                type=['md'],
                accept_multiple_files=True,
                key="md_uploader"
            )
            if arquivo:
                st.success(f"✅ {len(arquivo)} arquivo(s) .md carregado(s) com sucesso!")
        elif tipo_arquivo == '.pdf':
            arquivo = st.file_uploader(
                "Faça upload de arquivos PDF",
                type=['pdf'],
                accept_multiple_files=True,
                key="pdf_uploader"
            )
            if arquivo:
                st.success(f"✅ {len(arquivo)} arquivo(s) PDF carregado(s) com sucesso!")

 
    col1, col2 = st.columns(2)

    with col1:
        # habilita botão apenas quando há entradas válidas
        if modo_fonte == "Usar índice existente":
            can_init = bool(indice_escolhido and indice_escolhido != "—")
        else:
            if tipo_arquivo == 'url':
                can_init = bool(arquivo)
            elif tipo_arquivo in ('.md', '.pdf'):
                can_init = bool(arquivo)
            else:
                can_init = False

        btn = st.button("Inicializar SageBot", use_container_width=True, disabled=not can_init, key="btn_init")

        if btn:
            if not api_key:
                st.error("Por favor, insira a API Key antes de carregar o SageBot.")
                st.stop()

            if modo_fonte == "Usar índice existente":
                carregar_indice_existente(
                    h=indice_escolhido,
                    embed_model=embed_model, dims=dims, k=k, compressed=True
                )
                setup_chain(provedor, modelo, api_key)  # cria o chain (prompt + LLM)
                st.session_state["__needs_rerun"] = True
                st.success(f"Índice {indice_escolhido} carregado com sucesso!")
            else:
                # fluxo de indexar novos arquivos/URL
                carrega_modelo(provedor, modelo, api_key, tipo_arquivo, arquivo, embed_model, dims, k)
                st.success("SageBot carregado com sucesso!")

    with col2:
        if st.button('Apagar Histórico de Conversa', use_container_width=True, key="btn_clear"):
            st.session_state['memoria'] = []

def main():
    with st.sidebar:
        sidebar()
    pagina_chat()
    

if __name__ == '__main__':
    main()