import openai
import json
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Configuração da API OpenAI
openai.api_key = "SUA_CHAVE_API"

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
    Gera o diagnóstico usando GPT-4.
    """
    similaridades = calcular_similaridade(cliente_novo["caracteristicas"], revendedores)
    similares = sorted(similaridades, key=lambda x: x["similaridade"], reverse=True)[:2]

    # Gera explicação com GPT-4
    explicacao_prompt = f"""
    Dado o cliente novo com características {cliente_novo["caracteristicas"]},
    ele é mais similar aos seguintes revendedores:
    {similares}.
    Explique o diagnóstico de forma clara e objetiva.
    """
    resposta = openai.ChatCompletion.create(
        model="gpt-4",
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
    # Dados simulados
    entrada = {
        "cliente_novo": {
            "id": "cliente_estranho_123",
            "caracteristicas": {
                "volume_vendas": 5000,
                "regiao": "sudeste",
                "categoria_produtos": ["automotivo", "industrial"],
                "frequencia_compras": 12
            }
        },
        "revendedores_existentes": [
            {
                "id": "revenda_azul",
                "caracteristicas": {
                    "volume_vendas": 5200,
                    "regiao": "sudeste",
                    "categoria_produtos": ["automotivo", "industrial"],
                    "frequencia_compras": 11
                },
                "potencial": "alto"
            },
            {
                "id": "revenda_sol",
                "caracteristicas": {
                    "volume_vendas": 4800,
                    "regiao": "sudeste",
                    "categoria_produtos": ["automotivo"],
                    "frequencia_compras": 13
                },
                "potencial": "alto"
            },
            {
                "id": "revenda_verde",
                "caracteristicas": {
                    "volume_vendas": 3000,
                    "regiao": "norte",
                    "categoria_produtos": ["agrícola"],
                    "frequencia_compras": 8
                },
                "potencial": "médio"
            }
        ]
    }

    # Geração do diagnóstico
    diagnostico = gerar_diagnostico(entrada["cliente_novo"], entrada["revendedores_existentes"])

    # Exibe o resultado
    print(json.dumps(diagnostico, indent=4, ensure_ascii=False))