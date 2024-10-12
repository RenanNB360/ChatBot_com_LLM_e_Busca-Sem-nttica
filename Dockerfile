# Imagem base
FROM python:3.12

# Instala as dependências
RUN apt-get update && apt-get install -y build-essential

# Define a pasta de trabalho
WORKDIR /app

# Copia o arquivo de requerimentos para a pasta de trabalho no container
COPY requirements.txt .

# Adiciona o conteúdo da pasta corrente à pasta de trabalho no container
ADD . /app

# Executa a instalação dos pacotes em requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Deixa a porta acessível para acesso via navegador
EXPOSE 8501

# Executa a app
CMD ["streamlit", "run", "app.py"]