# 🤖 SageBot  — Desafio de NLP 

O **SageBot RAG** é um chatbot inteligente desenvolvido como solução para o **Desafio de NLP**, unindo **LangChain**, **OpenAI Embeddings** e **Streamlit** em uma arquitetura de **RAG (Retrieval-Augmented Generation)**.

Ele permite enriquecer as respostas do modelo com **conteúdo contextual** proveniente de documentos **Markdown (.md)**, **PDFs** ou **URLs**, criando um **assistente técnico contextualizado** — ideal para responder dúvidas baseadas em bases documentais.

---

## 🧠 Objetivo do projeto

O projeto foi desenvolvido como parte do **Desafio de Processamento de Linguagem Natural (NLP)**, com foco em:

- Implementar um pipeline **RAG completo**;
- Aplicar **embeddings semânticos** via OpenAI;
- Integrar **LLMs configuráveis (OpenAI / Groq)**;
- Exibir **status em tempo real** da indexação;
- Oferecer uma interface **simples e interativa via Streamlit**.

---

## ⚙️ Pré-requisitos obrigatórios

Para o funcionamento do chatbot, o usuário **deve obrigatoriamente**:

1. **Escolher uma fonte de conhecimento**, que será usada para enriquecer o contexto do chat:  
   - Arquivos **Markdown (.md)**  
   - Arquivos **PDF (.pdf)**  
   - Ou uma **URL** de um site/documentação pública  

2. **Selecionar o modelo de LLM** que será utilizado:  
   - **OpenAI** → `gpt-4o`, `gpt-4.1-nano`  
   - **Groq** → `llama-3.1-8b-instant`, `llama-3.3-70b-versatile`  

3. **Inserir sua chave pessoal (API Key)** do provedor escolhido:
   - `OPENAI_API_KEY` ou `GROQ_API_KEY`

4. **Escolher o modelo de embeddings OpenAI** (obrigatório):
   - `text-embedding-3-small` *(mais rápido e barato)*  
   - `text-embedding-3-large` *(melhor precisão semântica)*  

Sem esses quatro passos, o chatbot **não funcionará corretamente**.

---

## 🧩 Arquitetura do projeto

```
sagebot-rag/
├── app.py               # Interface principal (Streamlit)
├── loader.py            # Carregamento de MD, PDF e URLs
├── rag.py               # Split, embeddings e FAISS
├── work_rag.py          # Thread de indexação + cache
├── progress.py          # Controle de progresso e logs
├── utils.py             # Funções auxiliares (hash de documentos)
├── requirements.txt     # Dependências Python
├── .env.example         # Modelo de variáveis de ambiente
├── Dockerfile           # Configuração Docker
├── .dockerignore
├── .gitignore
└── data/
    └── index/           # Cache FAISS persistente
```
---

## 🧱 Instalação local (modo desenvolvedor)

1️⃣ **Clonar o repositório**
```bash
git clone https://github.com/seuusuario/sagebot-rag.git
cd sagebot-rag
```

2️⃣ **Criar ambiente virtual**
```bash
python -m venv .venv
.env\Scriptsctivate     # Windows
# ou
source .venv/bin/activate   # Linux/Mac
```

3️⃣ **Instalar dependências**
```bash
pip install -r requirements.txt
```

4️⃣ **Rodar a aplicação**
```bash
streamlit run app.py
```

5️⃣ **Acessar**
👉 [http://localhost:8501](http://localhost:8501)

---

## 🌐 Passo a passo para usar o chatbot (interface web)

### 🧩 1. Carregar o contexto
No menu lateral (sidebar):
- Escolha **um tipo de arquivo** (obrigatório): `.md`, `.pdf` ou `url`
- Faça upload do(s) arquivo(s) ou cole a URL desejada

### ⚙️ 2. Selecionar o modelo
- Selecione o provedor: **OpenAI** ou **Groq**
- Escolha o modelo de linguagem (LLM)
- Insira sua **API Key pessoal** correspondente

### 🧠 3. Configurar embeddings
- Escolha o modelo de embedding:  
  `text-embedding-3-small` ou `text-embedding-3-large`
- Ajuste o parâmetro **Top-K** (quantos resultados similares serão buscados no RAG)

### 🚀 4. Inicializar o SageBot
- Clique no botão **“Inicializar SageBot”**
- Acompanhe o progresso da indexação (split → embed → index → ready)

### 💬 5. Conversar
- Após o status indicar **“Índice RAG pronto ok.”**, digite sua pergunta.
- O bot responderá com base no contexto carregado.
- Se desejar, clique em **“Apagar Histórico de Conversa”** para reiniciar.

---

## 🐳 Execução com Docker

### 1️⃣ Construir imagem
```bash
docker build -t sagebot-rag .
```

### 2️⃣ Rodar container
```bash
docker run -p 8501:8501 --env-file .env sagebot-rag
```

### 3️⃣ Persistir cache FAISS
```bash
docker run -p 8501:8501   --env-file .env   -v $(pwd)/data:/app/data   sagebot-rag
```

> Acesse em: [http://localhost:8501](http://localhost:8501)

---

## ⚡️ Fluxo técnico interno

1. **Entrada:** usuário envia documentos (.md, .pdf) ou URL.  
2. **Split:** o texto é segmentado com `RecursiveCharacterTextSplitter`.  
3. **Embeddings:** vetores criados com `OpenAIEmbeddings`.  
4. **Indexação:** FAISS cria e salva o índice vetorial (`data/index/<hash>`).  
5. **Retriever:** busca semântica recupera os `k` trechos mais similares.  
6. **LLM:** modelo selecionado (OpenAI/Groq) gera resposta contextual.  
7. **UI:** progresso mostrado em tempo real via `progress.py`.

---

## 🧠 Dicas de performance

| Parâmetro | Descrição |
|------------|------------|
| `chunk_size=1500` | ótimo equilíbrio entre custo e recall |
| `batch_size=64` | evita erro de limite de tokens |
| `text-embedding-3-small` | recomendado para builds rápidos |
| `Top-K=4` | resultados mais relevantes e concisos |
| **Cache FAISS** | reduz tempo de indexação e custo de tokens |

---

## 🧰 Troubleshooting

| Erro | Causa | Solução |
|------|--------|----------|
| `OPENAI_API_KEY inválida` | Chave incorreta | Corrigir no `.env` |
| `Rate limit excedido` | Muitas chamadas à API | Aguardar alguns segundos |
| `Requested 302912 tokens...` | Texto muito grande | Ajustar chunk_size ou batch_size |
| `FAISS desatualizado` | Versão incompatível | `pip install --upgrade faiss-cpu` |
| `Streamlit travando` | Chamada de UI na thread errada | Use `render_status()` apenas no main thread |

---

## ✨ Créditos

Desenvolvido por **Mario Jorge Lira de Lima Junior**  

📍 *Manaus — AM*  

Projeto desenvolvido como entrega oficial do **Desafio de NLP** (Laboratório de Sistemas Inteligentes – LSE).
