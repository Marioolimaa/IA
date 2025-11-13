# 🤖 SageBot - Seu Assistente de Documentação Inteligente

SageBot é um assistente técnico especializado em **AWS**, projetado para responder perguntas com base na documentação oficial ou em qualquer outro material fornecido. Construído com Streamlit e LangChain, ele utiliza a técnica de **RAG (Retrieval-Augmented Generation)** para fornecer respostas precisas e didáticas, baseando-se estritamente nos documentos, artigos ou sites utilizados como fonte de conhecimento.

Este projeto foi desenvolvido como um item de portfólio para demonstrar habilidades em desenvolvimento de aplicações de IA, uso de LLMs, e boas práticas de engenharia de software.

## ✨ Funcionalidades Principais

- **Interface Intuitiva**: Uma aplicação web simples e amigável construída com Streamlit.
- **Múltiplas Fontes de Dados**: Suporta ingestão de conhecimento a partir de arquivos `.pdf`, `.md` e URLs de sites públicos.
- **Modelos Flexíveis**: Permite a escolha entre diferentes provedores de LLM, como **OpenAI** e **Groq**, e vários modelos de cada um.
- **Processamento Assíncrono**: A indexação de documentos (a parte mais demorada) é executada em segundo plano, mantendo a interface sempre responsiva.
- **Sistema de Cache Inteligente**: Documentos já indexados são salvos em um cache local (`data/index`). Ao carregar o mesmo documento novamente, o SageBot reutiliza o índice, economizando tempo e custos de API.
- **Retriever Avançado**: Utiliza MMR (Maximum Marginal Relevance) para buscar os trechos mais relevantes e diversos do documento, melhorando a qualidade do contexto enviado ao LLM.

## 🛠️ Tecnologias Utilizadas

- **Python**
- **Streamlit**: Para a interface web.
- **LangChain**: Para orquestrar o pipeline de RAG (splitters, embeddings, retrievers).
- **OpenAI / Groq**: Como provedores dos modelos de linguagem (LLM).
- **FAISS**: Para a criação e busca no banco de dados vetorial.

## 🚀 Como Executar o Projeto

Siga os passos abaixo para configurar e executar o SageBot em sua máquina local.

### 1. Pré-requisitos

- Python 3.9+
- Git

### 2. Clone o Repositório

```bash
git clone https://github.com/SEU-USUARIO/SEU-REPOSITORIO.git
cd SEU-REPOSITORIO
```
> **Nota**: Lembre-se de substituir `SEU-USUARIO/SEU-REPOSITORIO` pelo caminho correto do seu fork/clone.

### 3. Instale as Dependências

Crie um ambiente virtual e instale as bibliotecas necessárias a partir do arquivo `requirements.txt`.

```bash
# Crie e ative um ambiente virtual (recomendado)
python -m venv .venv
source .venv/bin/activate  # No Windows: .venv\Scripts\activate

# Instale as dependências
pip install -r requirements.txt
```

### 4. Configure as Variáveis de Ambiente

O SageBot precisa de chaves de API para se conectar aos serviços de LLM. A forma mais segura de fornecê-las é através de um arquivo `.env`.

Crie um arquivo chamado `.env` na raiz do projeto e adicione suas chaves:

```
# Chave da OpenAI (obrigatória para embeddings)
OPENAI_API_KEY="sk-..."

# Chave da Groq (opcional, se for usar os modelos da Groq)
GROQ_API_KEY="gsk_..."
```
Opcionalmente, você pode inserir as chaves diretamente na interface da aplicação.

### 5. Execute a Aplicação

Com tudo configurado, inicie a aplicação Streamlit:

```bash
streamlit run app.py
```

A aplicação será aberta em seu navegador. Na barra lateral, configure o modelo, forneça sua fonte de dados (faça upload de um arquivo ou insira uma URL) e clique em **"Inicializar SageBot"**. Após a indexação, você poderá começar a conversar!

---