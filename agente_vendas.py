# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import json

# -*- coding: utf-8 -*-

import openai
import json
import os
from typing import Dict, Any


import google.generativeai as genai
 
 
 

from dotenv import load_dotenv

# Defina sua chave da API OpenAI aqui
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

os.environ["DEEPSEEK_API_KEY"] = os.getenv("DEEPSEEK_API_KEY")



# 1. Leitura do CSV
file_path = '../Bases/BI-NNP_2025.csv'  # Ajuste o caminho conforme necessário

# Usando latin1 para evitar problemas com acentos comuns em arquivos brasileiros
df = pd.read_csv(file_path, sep=';', encoding='latin1', dtype=str)

print("✅ Arquivo carregado com sucesso!")
print(f"📊 Dimensões: {df.shape[0]} linhas, {df.shape[1]} colunas\n")

# 2. Função para converter string para float (substitui vírgula por ponto)
def str_to_float(x):
    if pd.isna(x):
        return np.nan
    try:
        return float(x.replace(',', '.'))
    except:
        return np.nan

# 3. Colunas a serem convertidas para numéricas
cols_to_float = [
    'Quant. Fatur', 'Vlr. Total', 'Vl IPI', 'Vl PIS', 'Vl COFINS', 'Vl ICM Item',
    'Unit. Net', 'Unit. Bruto', 'Valor Mat', 'Valor GGF', 'CPV - Total', 'CPV - Unit',
    'Vlr Total Líquido', 'Margem %', 'Marg. Bruta', '% M.P.', 'Cotacao', 'Unit Dolar', 'Total Dolar'
]

for col in cols_to_float:
    if col in df.columns:
        df[col] = df[col].apply(str_to_float)

# Remover linhas com valores críticos nulos
df.dropna(subset=['Vlr. Total', 'CPV - Total', 'Vlr Total Líquido'], inplace=True)

# 4. Cálculos adicionais
df['Lucro (R$)'] = df['Vlr Total Líquido'] - df['CPV - Total']
df['Preço Médio'] = df['Vlr. Total'] / df['Quant. Fatur']
df['Markup'] = df['Preço Médio'] / df['CPV - Unit']

# Classificação de margem
def classificar_margem(margem):
    if margem < 0: return '🔴 Negativa'
    elif margem < 10: return '🟡 Muito Baixa'
    elif margem < 30: return '🟠 Baixa'
    elif margem < 50: return '🟢 Média'
    else: return '🔵 Alta'

df['Categoria Margem'] = df['Margem %'].apply(classificar_margem)

# ===================================================================================
# NOVA FUNÇÃO: Gerar JSON Estratégico para LLM (OpenAI)
# Este JSON será enviado à LLM para gerar análise narrativa de alto impacto
# ===================================================================================

def gerar_json_para_llm(df):
    """
    Gera um JSON estruturado com métricas e insights para análise por LLM.
    O formato é claro, conciso e rico em contexto estratégico.
    """
    faturamento_total = float(df['Vlr. Total'].sum())
    lucro_bruto_total = float(df['Lucro (R$)'].sum())
    margem_media = (lucro_bruto_total / df['Vlr Total Líquido'].sum()) * 100 if df['Vlr Total Líquido'].sum() != 0 else 0

    # Top 3 clientes
    top_clientes = df.groupby('Nome Cliente')['Vlr. Total'].sum().sort_values(ascending=False).head(3)
    clientes_lista = [
        {"cliente": cliente, "faturamento_r$": round(float(valor), 2)} 
        for cliente, valor in top_clientes.items()
    ]

    # Desempenho por gestor
    desempenho_gestores = df.groupby('Gestor de Contas')['Vlr. Total'].sum().sort_values(ascending=False)
    gestores_lista = [
        {"gestor": gestor, "faturamento_r$": round(float(valor), 2)}
        for gestor, valor in desempenho_gestores.items()
    ]

    # Concentração de receita
    concentracao_top3 = (top_clientes.sum() / faturamento_total) * 100

    # Exportações
    exporta = df[df['Pais'] != 'BRASIL']
    exportacoes = {
        "total_r$": float(exporta['Vlr. Total'].sum()) if not exporta.empty else 0,
        "paises": exporta['Pais'].unique().tolist() if not exporta.empty else []
    }

    # Alertas: vendas com prejuízo
    margem_neg = df[df['Margem %'] < 0]
    alertas_prejuizo = []
    for _, row in margem_neg.iterrows():
        alertas_prejuizo.append({
            "cliente": row['Nome Cliente'],
            "produto": row['Descrição'],
            "nota_fiscal": row['Nr. Nota'],
            "prejuizo_r$": round(float(row['Lucro (R$)']), 2),
            "margem_%": round(float(row['Margem %']), 2)
        })

    # Oportunidades: produtos com alta margem (>50%)
    alta_margem = df[df['Margem %'] > 50]
    produtos_alta_margem = []
    for _, row in alta_margem.iterrows():
        produtos_alta_margem.append({
            "produto": row['Descrição'],
            "cliente": row['Nome Cliente'],
            "margem_%": round(float(row['Margem %']), 2),
            "faturamento_r$": round(float(row['Vlr. Total']), 2),
            "segmento": row['Segmentacao']
        })

    # Cross-sell: padrões de compra por segmento
    segmentos = df.groupby('Segmentacao')['Familia Comercial Descricao'].value_counts().to_dict()
    oportunidades_cross_sell = []
    if 'Print' in segmentos:
        oportunidades_cross_sell.append({
            "recomendacao": "Clientes do segmento Print (como DIGITO e SINALFIX) compram produtos cristal (PR-CLEAR), mas podem estar abertos ao PR-WHITE. Alta chance de conversão.",
            "segmento": "Print",
            "produtos_relacionados": ["PR-CLEAR", "PR-WHITE"]
        })
    if 'Pharma/Nutra/Vet' in segmentos:
        oportunidades_cross_sell.append({
            "recomendacao": "Clientes Pharma (CIMED, EPNB) consomem PH-CLEAR. Oportunidade de upsell com linhas especiais (UV, termoformáveis).",
            "segmento": "Pharma/Nutra/Vet",
            "produtos_relacionados": ["PH-CLEAR", "PH-UV"]
        })

    # Estrutura final do JSON
    relatorio_json = {
        "periodo_analisado": "Janeiro/2025",
        "metricas_gerais": {
            "faturamento_total_r$": round(faturamento_total, 2),
            "lucro_bruto_total_r$": round(lucro_bruto_total, 2),
            "margem_media_%": round(margem_media, 1),
            "numero_clientes_unicos": int(df['Nome Cliente'].nunique()),
            "concentracao_receita_top3_clientes_%": round(concentracao_top3, 1)
        },
        "desempenho": {
            "top_3_clientes_por_faturamento": clientes_lista,
            "desempenho_por_gestor": gestores_lista,
            "exportacoes": exportacoes
        },
        "oportunidades": {
            "produtos_com_margem_acima_de_50%": produtos_alta_margem,
            "potenciais_de_cross_sell_e_upsell": oportunidades_cross_sell
        },
        "riscos_criticos": {
            "vendas_com_prejuizo": alertas_prejuizo,
            "observacoes": "Venda com margem negativa representa perda direta de caixa. Revisar política de descontos e formação de preço."
        }
    }

    return relatorio_json

# ================================
# EXECUÇÃO DA FUNÇÃO
# ================================




def chamar_llm_openai(prompt: str, modelo: str = "gpt-4o-mini", temperatura: float = 0.7) -> Dict[str, Any]:
    """
    Função dedicada à chamada da API da OpenAI.
    
    Args:
        prompt (str): O texto completo a ser enviado à LLM.
        modelo (str): Modelo da OpenAI (ex: gpt-4o-mini, gpt-4-turbo, gpt-3.5-turbo)
        temperatura (float): Criatividade da resposta (0 = determinístico, 1 = criativo)
    
    Returns:
        Dict com 'sucesso', 'resposta', e opcionalmente 'erro'
    """
    try:
        resposta = openai.ChatCompletion.create(
            model=modelo,
            messages=[
                {
                    "role": "system",
                    "content": "Você é um Diretor Comercial Sênior. Analise os dados e gere um relatório executivo claro, estratégico e acionável. Fale diretamente com gestores e CEO. Use linguagem impactante, mas baseada em dados. Evite jargões técnicos sem explicação."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=temperatura,
            max_tokens=1500,
            response_format={"type": "text"}  # Pode ser JSON se quiser resposta estruturada
        )
        texto_resposta = resposta.choices[0].message['content'].strip()
        return {
            "sucesso": True,
            "resposta": texto_resposta
        }
    except Exception as e:
        return {
            "sucesso": False,
            "erro": str(e)
        }
    

def chamar_llm_gemini(prompt: str, modelo: str = "gemini-2.5-flash-lite", temperatura: float = 0.7) -> Dict[str, Any]:
    """
    Função dedicada à chamada da API do Google Gemini.
    
    Args:
        prompt (str): O prompt completo a ser enviado à LLM.
        modelo (str): Modelo Gemini (ex: gemini-1.5-pro, gemini-1.0-pro)
        temperatura (float): Controle de criatividade (0 a 1)
    
    Returns:
        Dict com 'sucesso', 'resposta' e opcionalmente 'erro'
    """
    try:
        # Carregar o modelo
        model = genai.GenerativeModel(model_name=modelo)
        
        # Configurar geração
        config = genai.types.GenerationConfig(
            temperature=temperatura,
            max_output_tokens=1500
        )
        
        # Enviar prompt
        response = model.generate_content(
            contents=prompt,
            generation_config=config
        )
        
        return {
            "sucesso": True,
            "resposta": response.text.strip()
        }
    
    except Exception as e:
        return {
            "sucesso": False,
            "erro": str(e)
        }
    
def gerar_prompt_analise_executiva(dados_json: Dict) -> str:
    """
    Transforma o JSON de insights em um prompt poderoso para a LLM.
    """
    prompt = """
Analise profundamente os dados de vendas abaixo e gere um **relatório executivo de alto impacto**, como se fosse apresentar ao CEO e gestores de vendas.

📌 Instruções:
- Use um **título impactante** no início.
- Destaque **3 conquistas estratégicas**.
- Exponha **2 riscos críticos** com dados concretos.
- Aponte **3 oportunidades reais de crescimento imediato** (cross-sell, exportação, margem).
- Dê **recomendações específicas por gestor de conta**.
- Use linguagem clara, direta, com senso de urgência.
- Nada de "Olá, aqui estão os dados...". Vá direto ao ponto.
- Evite termos genéricos como "melhorar performance". Seja acionável.

📊 DADOS PARA ANÁLISE:
""" + json.dumps(dados_json, indent=2, ensure_ascii=False)

    return prompt


relatorio_json = gerar_json_para_llm(df)

# Salvar o JSON para inspeção ou envio à LLM
with open('relatorio_para_llm.json', 'w', encoding='utf-8') as f:
    json.dump(relatorio_json, f, indent=2, ensure_ascii=False)

print("📄 JSON estratégico gerado com sucesso!")
print("📁 Arquivo salvo como 'relatorio_para_llm.json'")
print("\n💡 Próximo passo: envie este JSON para uma LLM (ex: GPT-4) com um prompt poderoso para gerar a análise executiva.")

# Exibir uma prévia (opcional)
print("\n🔍 Prévia do JSON (primeiros 500 caracteres):")
print(json.dumps(relatorio_json, indent=2, ensure_ascii=False)[:500] + "...")

# 1. Gerar o JSON com insights
relatorio_json = gerar_json_para_llm(df)

# 2. Gerar o prompt para a LLM
prompt = gerar_prompt_analise_executiva(relatorio_json)

# 3. Chamar a OpenAI
print("\n🚀 Enviando análise para a OpenAI...")
resposta_llm = chamar_llm_gemini(prompt, modelo="gemini-2.5-flash-lite", temperatura=0.7)

if resposta_llm["sucesso"]:
    print("\n💡 ANÁLISE DA LLM (OpenAI):\n")
    print(resposta_llm["resposta"])
    
    # Salvar resultado
    with open('analise_executiva_openai.txt', 'w', encoding='utf-8') as f:
        f.write(resposta_llm["resposta"])
    print("\n✅ Análise salva em 'analise_executiva_openai.txt'")
else:
    print(f"❌ Erro ao chamar OpenAI: {resposta_llm['erro']}")
