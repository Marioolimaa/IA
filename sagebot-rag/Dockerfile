FROM python:3.11-slim

WORKDIR /app

# 1. Copia e instala as dependências primeiro para aproveitar o cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2. Copia apenas os artefatos necessários da aplicação
COPY app.py .
COPY sagebot/ ./sagebot/

# 3. Expõe a porta e define o comando de execução
ENV USER_AGENT="SageBot/1.0 (Streamlit)"
EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
