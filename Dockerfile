# Imagem base com Python e Streamlit
FROM python:3.9-slim-buster

# Diretório de trabalho dentro do container
WORKDIR /app

# Copiar os arquivos de dependência
COPY requirements.txt requirements.txt

# Instalar as dependências
RUN pip install --no-cache-dir -r requirements.txt

# Copiar o código fonte para o container (agora da pasta app)
COPY app/ .

# Comando para iniciar a aplicação Streamlit (agora dentro da pasta app)
CMD ["streamlit", "run", "main.py"] 