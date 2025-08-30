# -*- coding: utf-8 -*-
import streamlit as st
import plotly.express as px
 
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import json
import os
from typing import Dict, Any
import streamlit as st

# === CHAMADAS ÀS LLMs ===
import openai
import google.generativeai as genai
from dotenv import load_dotenv
 

# Configurar página primeiro
st.set_page_config(
    page_title="📊 Agente de IA | Análise de Vendas",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilo CSS personalizado
st.markdown("""
    <style>
    .reportview-container {
        background: #f8f9fa;
    }
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }
    h1, h2, h3 {
        color: #1a3b5d;
    }
    .stButton>button {
        background-color: #1a3b5d;
        color: white;
        border-radius: 8px;
        height: 40px;
    }
    .stButton>button:hover {
        background-color: #2c5f8c;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.05);
        margin-bottom: 1rem;
        border-left: 4px solid #1a3b5d;
    }
    .alert-red {
        background-color: #ffebee;
        color: #c62828;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .alert-green {
        background-color: #e8f5e8;
        color: #2e7d32;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Título
st.title("📈 Agente de IA: Análise Estratégica de Vendas")
st.markdown("Analise seus dados de vendas e gere **insights acionáveis com IA generativa**.")

# === SIDEBAR ===
with st.sidebar:
    st.header("⚙️ Configurações")
    
    uploaded_file = st.file_uploader(
        "📤 Carregue seu CSV de vendas",
        type=["csv"],
        help="O arquivo deve conter colunas como 'Vlr. Total', 'Margem %', 'Nome Cliente', etc."
    )
    
    st.markdown("---")
    
    llm_choice = st.selectbox(
        "🧠 Escolha a IA",
        options=["Gemini", "OpenAI"],
        index=0
    )
    
    model_map = {
        "Gemini": "gemini-1.5-flash-latest",
        "OpenAI": "gpt-4o-mini"
    }
    
    temperature = st.slider("Creatividade (0 = precisa, 1 = criativa)", 0.0, 1.0, 0.7)
    
    st.markdown("---")
    st.markdown("### 📂 Sobre o formato")
    st.caption("Esperado: separador `;`, encoding `latin1`, valores com `,` como decimal.")

# === VALIDAÇÃO E LEITURA DO CSV ===
if not uploaded_file:
    st.info("👆 Por favor, carregue um arquivo CSV no menu lateral para começar.")
    st.stop()

try:
    df = pd.read_csv(uploaded_file, sep=';', encoding='latin1', dtype=str)
    st.success("✅ Arquivo carregado com sucesso!")
except Exception as e:
    st.error(f"❌ Erro ao ler o CSV: {e}")
    st.stop()

if df.empty:
    st.warning("⚠️ O arquivo está vazio.")
    st.stop()

# === PROCESSAMENTO DE DADOS ===
def str_to_float(x):
    if pd.isna(x): return np.nan
    try: return float(x.replace(',', '.'))
    except: return np.nan

cols_to_float = [
    'Quant. Fatur', 'Vlr. Total', 'Vl IPI', 'Vl PIS', 'Vl COFINS', 'Vl ICM Item',
    'Unit. Net', 'Unit. Bruto', 'Valor Mat', 'Valor GGF', 'CPV - Total', 'CPV - Unit',
    'Vlr Total Líquido', 'Margem %', 'Marg. Bruta', '% M.P.', 'Cotacao', 'Unit Dolar', 'Total Dolar'
]

for col in cols_to_float:
    if col in df.columns:
        df[col] = df[col].apply(str_to_float)

df.dropna(subset=['Vlr. Total', 'CPV - Total', 'Vlr Total Líquido'], inplace=True)

# Cálculos
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

# === GERAÇÃO DO JSON PARA LLM ===
def gerar_json_para_llm(df):
    faturamento_total = float(df['Vlr. Total'].sum())
    lucro_bruto_total = float(df['Lucro (R$)'].sum())
    margem_media = (lucro_bruto_total / df['Vlr Total Líquido'].sum()) * 100 if df['Vlr Total Líquido'].sum() != 0 else 0

    top_clientes = df.groupby('Nome Cliente')['Vlr. Total'].sum().sort_values(ascending=False).head(3)
    clientes_lista = [{"cliente": c, "faturamento_r$": round(v, 2)} for c, v in top_clientes.items()]

    desempenho_gestores = df.groupby('Gestor de Contas')['Vlr. Total'].sum().sort_values(ascending=False)
    gestores_lista = [{"gestor": g, "faturamento_r$": round(v, 2)} for g, v in desempenho_gestores.items()]

    concentracao_top3 = (top_clientes.sum() / faturamento_total) * 100

    exporta = df[df['Pais'] != 'BRASIL']
    exportacoes = {
        "total_r$": float(exporta['Vlr. Total'].sum()) if not exporta.empty else 0,
        "paises": exporta['Pais'].unique().tolist() if not exporta.empty else []
    }

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

    return {
        "periodo_analisado": "Dados Carregados",
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
            "potenciais_de_cross_sell_e_upsell": [
                {"recomendacao": "Clientes do segmento Print podem comprar produtos complementares (PR-CLEAR → PR-WHITE)."},
                {"recomendacao": "Expansão para Mercosul com base em sucesso no Uruguai."}
            ]
        },
        "riscos_criticos": {
            "vendas_com_prejuizo": alertas_prejuizo,
            "observacoes": "Revisar políticas de desconto e formação de preço."
        }
    }

# Gerar JSON
relatorio_json = gerar_json_para_llm(df)



load_dotenv()

# openai.api_key = os.getenv("OPENAI_API_KEY")
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# genai.configure(api_key=GOOGLE_API_KEY)


openai.api_key = st.secrets["OPENAI_API_KEY"]
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])



from fpdf import FPDF
from datetime import datetime

def gerar_pdf_profissional(relatorio_json, analise_ia, df, output_path="relatorio_executivo.pdf"):
    """
    Gera um PDF corporativo com logo, análise da IA, gráficos (como imagem) e métricas.
    """
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # --- Capa ---
    pdf.set_fill_color(30, 59, 93)  # Azul escuro
    pdf.rect(0, 0, 210, 297, 'F')  # Fundo azul
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Arial", "B", 28)
    pdf.cell(0, 100, "Relatório de Vendas", ln=True, align="C")
    pdf.set_font("Arial", "", 18)
    pdf.cell(0, 10, f"{relatorio_json['periodo_analisado']}", ln=True, align="C")
    pdf.cell(0, 10, f"Gerado em: {datetime.now().strftime('%d/%m/%Y')}", ln=True, align="C")
    pdf.ln(40)
    pdf.set_font("Arial", "", 14)
    pdf.cell(0, 10, "Análise Inteligente com IA Generativa", ln=True, align="C")
    pdf.set_text_color(0, 0, 0)

    # --- Página 1: Métricas e Análise ---
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.set_text_color(30, 59, 93)
    pdf.cell(0, 10, "Resumo Executivo", ln=True)
    pdf.ln(5)

    # Métricas
    pdf.set_font("Arial", "", 12)
    metricas = relatorio_json["metricas_gerais"]
    pdf.cell(0, 8, f"- Faturamento Total: R$ {metricas['faturamento_total_r$']:,.2f}", ln=True)
    pdf.cell(0, 8, f"- Lucro Bruto: R$ {metricas['lucro_bruto_total_r$']:,.2f}", ln=True)
    pdf.cell(0, 8, f"- Margem Média: {metricas['margem_media_%']}%", ln=True)
    pdf.cell(0, 8, f"- Concentração Top 3: {metricas['concentracao_receita_top3_clientes_%']}% do faturamento", ln=True)
    pdf.ln(10)

    # Análise da IA
    pdf.set_font("Arial", "B", 14)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 10, "Análise da Inteligência Artificial", ln=True)
    pdf.set_font("Arial", "", 12)
    pdf.set_text_color(50, 50, 50)
    
    # Quebrar texto em linhas
    for line in analise_ia.split("\n"):
        if line.strip():
            # Substituir caracteres não suportados
            safe_line = line.strip().replace("→", "->")
            pdf.cell(0, 6, safe_line, ln=True)

    # --- Página 2: Gráficos (exemplo: salvar como PNG e inserir)
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.set_text_color(30, 59, 93)
    pdf.cell(0, 10, "Painel de Desempenho", ln=True)
    pdf.ln(10)

    # Aqui você pode salvar os gráficos como imagens e inserir
    # Exemplo (se você salvar fig1 como 'fig1.png'):
    # pdf.image("fig1.png", x=10, w=190)

    # Placeholder
    pdf.set_font("Arial", "I", 12)
    pdf.cell(0, 10, "Gráfico 1: Margem por Cliente (Top 10)", ln=True)
    # pdf.image("fig1.png", x=15, w=180)  # Descomente quando salvar figuras

    pdf.ln(10)
    pdf.cell(0, 10, "Gráfico 2: Faturamento por Segmento + Margem", ln=True)
    # pdf.image("fig2.png", x=15, w=180)

    # --- Finalizar ---
    pdf.set_font("Arial", "I", 10)
    pdf.set_text_color(100, 100, 100)
    pdf.ln(20)
    pdf.cell(0, 10, "© 2025 | Agente de IA - Análise de Vendas | Confidencial", align="C")

    # Salvar
    pdf.output(output_path)
    return output_path

def chamar_llm_openai(prompt, modelo="gpt-4o-mini", temperatura=0.7):
    try:
        resposta = openai.ChatCompletion.create(
            model=modelo,
            messages=[
                {"role": "system", "content": "Você é um Diretor Comercial Sênior. Gere relatórios executivos diretos, estratégicos e acionáveis."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperatura,
            max_tokens=1500
        )
        return {"sucesso": True, "resposta": resposta.choices[0].message['content'].strip()}
    except Exception as e:
        return {"sucesso": False, "erro": str(e)}

def chamar_llm_gemini(prompt, modelo="gemini-1.5-flash-latest", temperatura=0.7):
    try:
        model = genai.GenerativeModel(model_name=modelo)
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(temperature=temperatura, max_output_tokens=1500)
        )
        return {"sucesso": True, "resposta": response.text.strip()}
    except Exception as e:
        return {"sucesso": False, "erro": str(e)}

def gerar_prompt_avancado(relatorio_json):
    """
    Gera um prompt que força a LLM a ir além do óbvio.
    """
    prompt = """
Você é um **Analista de Inteligência de Negócios com foco em vendas e lucratividade**. 
Seu papel não é repetir o que está na superfície, mas **revelar o que está escondido nos dados**.

📌 Instruções rigorosas:
- **NÃO mencione métricas básicas** como "faturamento total" ou "top 3 clientes" a menos que estejam ligadas a um insight não óbvio.
- **NÃO use frases genéricas** como "melhorar margens" ou "aumentar vendas".
- **FOQUE em descobertas que surpreenderiam até o gestor de conta**.
- **PROPOSTAS devem ser acionáveis, específicas e baseadas em padrões ocultos**.
- Use linguagem de **consultoria de alto impacto**, como se estivesse apresentando ao COO ou ao CEO.

🔍 Busque e destaque:
1. **Padrões de compra não óbvios** (ex: cliente A compra produto X, mas ignora Y, que é complementar).
2. **Anomalias de lucratividade** (ex: alto faturamento com baixa margem — por quê?).
3. **Oportunidades de cross-sell baseadas em segmento, não em volume**.
4. **Riscos silenciosos** (ex: um gestor com alta venda, mas todos os clientes com margem abaixo da média).
5. **Potencial de exportação com base em perfis de cliente, não apenas geografia**.

📊 DADOS:
"""
    prompt += json.dumps(relatorio_json, indent=2, ensure_ascii=False)
    return prompt 


def gerar_prompt_analise_executiva(dados_json):
    return """
Analise os dados abaixo e gere um relatório executivo com:
- Título impactante
- 3 conquistas
- 2 riscos críticos
- 3 oportunidades
- Recomendações por gestor

Responda de forma profissional, clara e objetiva.

Mantenha um formato amigável e navegável.

Foque em qualidade da análise mais do que em volume de texto.

DADOS:
""" + json.dumps(dados_json, indent=2, ensure_ascii=False)


# Função para prompt de auditoria (adicione no topo, com os outros prompts)
def gerar_prompt_auditoria(relatorio_json):
    """
    Gera um prompt para obter uma auditoria estratégica com priorização de ações.
    Foco: riscos silenciosos, ineficiências e recomendações acionáveis e priorizadas.
    """
    prompt = """
Você é um **Auditor Estratégico de Vendas e Lucratividade**, com experiência em análise de margens, formação de preço e eficiência comercial.
Seu papel é **identificar falhas ocultas, ineficiências e oportunidades de alto impacto**, e transformá-las em um plano de ação claro e priorizado.

📌 Instruções rigorosas:
1. Liste **3 RISCOS CRÍTICOS** (ex: venda com prejuízo, gestor com má lucratividade, concentração de receita)
2. Liste **3 OPORTUNIDADES DE ALTO IMPACTO** (ex: produtos com alta margem e baixo volume, cross-sell óbvio, expansão de exportação)
3. Para cada item, atribua uma **PRIORIDADE: Alta, Média ou Baixa**
4. Justifique com dados: impacto financeiro (R$), urgência, facilidade de implementação
5. Dê uma **ação concreta, específica e acionável** (com nome de gestor, cliente ou produto)

📌 Formato obrigatório:
- **[Alta] Cliente X: problema com produto Y**
  → Impacto: R$ XX.XXX
  → Ação: [Ação específica, ex: "Felipe deve revisar o desconto aplicado no pedido 47159"]

📌 Não generalize. Seja direto, técnico e acionável.

📊 DADOS:
"""
    prompt += json.dumps(relatorio_json, indent=2, ensure_ascii=False)
    return prompt

# === BOTÃO DE ANÁLISE ===
# === BOTÃO DE ANÁLISE ===
if st.sidebar.button("🚀 Gerar Análise com IA", type="primary"):
    with st.spinner(f"Gerando análises com {llm_choice}..."):
        # Gerar os três prompts
        prompt_basico = gerar_prompt_analise_executiva(relatorio_json)
        prompt_avancado = gerar_prompt_avancado(relatorio_json)
        prompt_auditoria = gerar_prompt_auditoria(relatorio_json)  # Nova função abaixo

        # Chamar LLM para cada nível
        if llm_choice == "OpenAI":
            resultado_basico = chamar_llm_openai(prompt_basico, temperatura=0.5)
            resultado_avancado = chamar_llm_openai(prompt_avancado, temperatura=0.7)
            resultado_auditoria = chamar_llm_openai(prompt_auditoria, temperatura=0.6)
        else:
            resultado_basico = chamar_llm_gemini(prompt_basico, temperatura=0.5)
            resultado_avancado = chamar_llm_gemini(prompt_avancado, temperatura=0.7)
            resultado_auditoria = chamar_llm_gemini(prompt_auditoria, temperatura=0.6)

        # Salvar todos os resultados
        st.session_state['resultado_basico'] = resultado_basico
        st.session_state['resultado_avancado'] = resultado_avancado
        st.session_state['resultado_auditoria'] = resultado_auditoria
        st.success("✅ Análises geradas com sucesso!")



# === EXIBIÇÃO DOS RESULTADOS (com 5 abas) ===
if 'resultado_basico' in st.session_state:
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📋 Resumo Executivo",
        "🔍 Análise Avançada",
        "🛠️ Auditoria & Ações",
        "📊 Dashboard",
        "🔍 Dados Técnicos"
    ])

    # --- ABAS DE ANÁLISE ---
    with tab1:
        st.subheader("📌 Relatório Executivo")
        res = st.session_state['resultado_basico']
        if res["sucesso"]:
            st.markdown(f"<div style='line-height:1.7'>{res['resposta']}</div>", unsafe_allow_html=True)
        else:
            st.error(f"❌ Erro no Resumo Executivo: {res['erro']}")

    with tab2:
        st.subheader("🔍 Inteligência de Negócios: O que os Dados Escondem")
        res = st.session_state['resultado_avancado']
        if res["sucesso"]:
            st.markdown(f"<div style='line-height:1.7'>{res['resposta']}</div>", unsafe_allow_html=True)
        else:
            st.error(f"❌ Erro na Análise Avançada: {res['erro']}")

    with tab3:
        st.subheader("🛠️ Auditoria de Vendas e Priorização de Ações")
        res = st.session_state['resultado_auditoria']
        if res["sucesso"]:
            st.markdown("### 🔎 Riscos e Oportunidades Priorizados")
            st.markdown(f"<div style='line-height:1.7'>{res['resposta']}</div>", unsafe_allow_html=True)
        else:
            st.error(f"❌ Erro na Auditoria: {res['erro']}")

    # --- ABA DE GRÁFICOS (mantida) ---
    with tab4:
        st.subheader("📈 Painel de Desempenho Comercial")
        
        por_cliente = df.groupby('Nome Cliente').agg({
            'Vlr. Total': 'sum', 'Lucro (R$)': 'sum', 'Margem %': 'mean'
        }).round(2).reset_index()

        por_segmento = df.groupby('Segmentacao').agg({
            'Vlr. Total': 'sum', 'Margem %': 'mean'
        }).reset_index()

        por_produto = df.groupby(['Descrição', 'Familia Comercial Descricao']).agg({
            'Vlr. Total': 'sum', 'Margem %': 'mean', 'Quant. Fatur': 'sum'
        }).reset_index()

        por_gestor_perf = df.groupby('Gestor de Contas').agg({
            'Vlr. Total': 'sum', 'Lucro (R$)': 'sum', 'Margem %': 'mean'
        }).reset_index()

        # Gráfico 1: Margem por Cliente (Top 10)
        top10_clientes = por_cliente.sort_values('Vlr. Total', ascending=False).head(10)
        fig1 = px.bar(
            top10_clientes,
            x='Nome Cliente',
            y='Margem %',
            color='Margem %',
            color_continuous_scale=['red', 'orange', 'green'],
            title="1. Margem % por Cliente (Top 10 por Faturamento)",
            text='Margem %'
        )
        fig1.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        st.plotly_chart(fig1, use_container_width=True)

        # Gráfico 2: Segmento - Faturamento + Margem
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(x=por_segmento['Segmentacao'], y=por_segmento['Vlr. Total'], name='Faturamento', marker_color='#1a3b5d'))
        fig2.add_trace(go.Scatter(x=por_segmento['Segmentacao'], y=por_segmento['Margem %'], mode='lines+markers+text',
                                  name='Margem Média (%)', line=dict(color='#e74c3c', width=3),
                                  text=[f"{val:.1f}%" for val in por_segmento['Margem %']], textposition="top center"))
        fig2.update_layout(
            title="2. Faturamento por Segmento com Margem Média",
            yaxis=dict(title="Faturamento (R$)"),
            yaxis2=dict(title="Margem Média (%)", overlaying="y", side="right"),
            legend=dict(x=0.1, y=1.15, orientation="h"),
            height=500
        )
        st.plotly_chart(fig2, use_container_width=True)

        # Gráfico 3: Treemap - Produtos
        fig3 = px.treemap(
            por_produto,
            path=['Familia Comercial Descricao', 'Descrição'],
            values='Vlr. Total',
            color='Margem %',
            color_continuous_scale='RdYlGn',
            title="3. Participação dos Produtos (Tamanho = Faturamento, Cor = Margem %)"
        )
        fig3.update_traces(textinfo="label+value+percent entry")
        st.plotly_chart(fig3, use_container_width=True)

        # Gráfico 4: Gestores - Participação
        fig4 = px.pie(por_gestor_perf, names='Gestor de Contas', values='Vlr. Total',
                      title="4. Participação no Faturamento por Gestor", hole=0.4)
        fig4.update_traces(textinfo='percent+label')
        st.plotly_chart(fig4, use_container_width=True)

        # Gráfico 5: Gestores - Volume vs. Rentabilidade
        # Ajustar o tamanho das bolhas mantendo a informação de lucro/prejuízo
        min_lucro = por_gestor_perf['Lucro (R$)'].min()
        por_gestor_perf['Tamanho_Bolha'] = por_gestor_perf['Lucro (R$)'] - min_lucro + 1  # Deslocar para positivo
        
        fig5 = px.scatter(
            por_gestor_perf,
            x='Vlr. Total',
            y='Margem %',
            size='Tamanho_Bolha',
            color='Lucro (R$)',  # Colorir com base no lucro real
            color_continuous_scale=['red', 'yellow', 'green'],  # Vermelho para prejuízo, verde para lucro
            hover_name='Gestor de Contas',
            hover_data={
                'Lucro (R$)': ':,.2f',  # Formatar com separador de milhares e 2 decimais
                'Tamanho_Bolha': False,  # Esconder coluna auxiliar
                'Vlr. Total': ':,.2f',
                'Margem %': ':.1f'
            },
            title="5. Gestores: Volume vs. Rentabilidade<br><sup>Cor: Vermelho=Prejuízo, Verde=Lucro | Tamanho = Volume de Operação</sup>",
            labels={'Lucro (R$)': 'Lucro/Prejuízo (R$)'}
        )
        fig5.add_hline(y=relatorio_json['metricas_gerais']['margem_media_%'], 
                       line_dash="dash", line_color="gray", annotation_text="Média Geral")
        st.plotly_chart(fig5, use_container_width=True)

    # --- ABA TÉCNICA ---
    with tab5:
        st.json(relatorio_json, expanded=False)
        st.download_button(
            "📥 Baixar JSON",
            data=json.dumps(relatorio_json, indent=2, ensure_ascii=False),
            file_name="relatorio_para_llm.json",
            mime="application/json"
        )

    # Alerta geral
    if relatorio_json["riscos_criticos"]["vendas_com_prejuizo"]:
        st.markdown('<div class="alert-red">⚠️ <b>Atenção:</b> Existem vendas com <b>margem negativa</b>.</div>', unsafe_allow_html=True)

# Botão de PDF
if st.sidebar.button("📥 Gerar PDF Executivo"):
    if 'resultado_basico' not in st.session_state:
        st.error("❌ Primeiro, gere a análise com IA!")
    else:
        with st.spinner("📄 Gerando PDF profissional..."):
            caminho_pdf = gerar_pdf_profissional(
                relatorio_json,
                st.session_state['resultado_basico']["resposta"],
                df
            )
            with open(caminho_pdf, "rb") as f:
                st.download_button(
                    "⬇️ Baixar Relatório em PDF",
                    f,
                    file_name="relatorio_executivo_vendas.pdf",
                    mime="application/pdf"
                )

else:
    st.info("👈 Carregue um arquivo e clique em 'Gerar Análise com IA' para começar.")