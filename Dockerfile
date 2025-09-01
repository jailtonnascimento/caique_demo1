# Dockerfile- mudou o nome
# Usa uma imagem estável do Python 3.11 com Alpine ou Debian
FROM python:3.11.8-slim-bullseye AS base

# Evita perguntas durante a instalação de pacotes
ENV DEBIAN_FRONTEND=noninteractive


# Define o diretório de trabalho
WORKDIR /app


# Instala dependências do sistema (necessárias para compilar numpy/pandas se necessário)
# (Opcional: remover se usar apenas wheels)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
    && apt-get upgrade -y && \
    apt-get dist-upgrade -y && \
    rm -rf /var/lib/apt/lists/*

# Copia requirements
COPY requirements.txt .

# instala os componentes
# Instala dependências
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copia o código da aplicação
COPY . .

# Cria usuário não-root (boa prática de segurança)
RUN groupadd -r streamlit && useradd -r -g streamlit -m appuser && \
    chown -R appuser:streamlit /app

USER appuser
ENV HOME=/home/appuser
 
# Expõe a porta
EXPOSE 8501

# Comando para rodar o Streamlit
CMD ["streamlit", "run", "st_app.py", "--server.port=8501", "--server.address=0.0.0.0"]