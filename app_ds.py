import openai
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import deepseek
from dotenv import load_dotenv
import os

# Carrega as variáveis de ambiente do arquivo .env
load_dotenv()

# Configuração da API OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

# Configuração da API DeepSeek
deepseek.api_key = os.getenv("DEEPSEEK_API_KEY")

def calcular_similaridade(cliente_novo, revendedores):
    """
    Calcula a similaridade entre o cliente novo e os revendedores existentes.
    """
    # Combina as características em strings para análise de similaridade
    cliente_str = " ".join([f"{k}:{v}" for k, v in cliente_novo.items()])
    revendedores_str = [
        (revenda["id"], " ".join([f"{k}:{v}" for k, v in revenda["caracteristicas"].items()]))
        for revenda in revendedores
    ]

    # Vetorização usando TF-IDF
    textos = [cliente_str] + [r[1] for r in revendedores_str]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(textos)

    # Calcula similaridade de cosseno
    similaridades = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    return [
        {"id": revendedores[i]["id"], "similaridade": similaridades[i], "potencial": revendedores[i]["potencial"]}
        for i in range(len(revendedores))
    ]

def gerar_diagnostico(cliente_novo, revendedores):
    """
    Gera o diagnóstico usando DeepSeek.
    """
    similaridades = calcular_similaridade(cliente_novo["caracteristicas"], revendedores)
    similares = sorted(similaridades, key=lambda x: x["similaridade"], reverse=True)[:2]

    # Gera explicação com DeepSeek
    explicacao_prompt = f"""
    Dado o cliente novo com características {cliente_novo["caracteristicas"]},
    ele é mais similar aos seguintes revendedores:
    {similares}.
    Explique o diagnóstico de forma clara e objetiva.
    """
    resposta = deepseek.ChatCompletion.create(
        model="deepseek-llm",
        messages=[
            {"role": "system", "content": "Você é um assistente especializado em análise de dados."},
            {"role": "user", "content": explicacao_prompt}
        ]
    )

    explicacao = resposta["choices"][0]["message"]["content"]

    return {
        "diagnostico": {
            "cliente_novo": cliente_novo["id"],
            "revendedores_similares": similares,
            "explicacao": explicacao.strip()
        }
    }

# Exemplo de uso
if __name__ == "__main__":
    # Lê os dados do arquivo CSV
    df = pd.read_csv("dados.csv")

    # Assume que a primeira linha é o cliente novo
    cliente_row = df.iloc[0]
    cliente_novo = {
        "id": cliente_row["id"],
        "caracteristicas": {
            "volume_vendas": cliente_row["volume_vendas"],
            "regiao": cliente_row["regiao"],
            "categoria_produtos": [x.strip() for x in str(cliente_row["categoria_produtos"]).split(";")],
            "frequencia_compras": cliente_row["frequencia_compras"]
        }
    }

    # As demais linhas são os revendedores existentes
    revendedores_existentes = []
    for _, row in df.iloc[1:].iterrows():
        revendedores_existentes.append({
            "id": row["id"],
            "caracteristicas": {
                "volume_vendas": row["volume_vendas"],
                "regiao": row["regiao"],
                "categoria_produtos": [x.strip() for x in str(row["categoria_produtos"]).split(";")],
                "frequencia_compras": row["frequencia_compras"]
            },
            "potencial": row["potencial"]
        })

    # Geração do diagnóstico
    diagnostico = gerar_diagnostico(cliente_novo, revendedores_existentes)

    # Exibe o resultado
    print(diagnostico)