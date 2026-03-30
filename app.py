import os
import streamlit as st
import pandas as pd
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS

st.set_page_config(
    page_title="Dashboard - Reclame Aqui Hapvida",
    page_icon="📊",
    layout="wide"
)

CSV_PATH = "dados/RECLAMEAQUI_HAPVIDA.csv"
SHP_PATH = "assets/mapa_brasil/BR_UF_2024.shp"


@st.cache_data
def carregar_dados():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"Arquivo não encontrado: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    df.columns = [c.strip().upper() for c in df.columns]

    # Garantias mínimas
    if "STATUS" not in df.columns:
        df["STATUS"] = "Não informado"
    df["STATUS"] = df["STATUS"].fillna("Não informado").astype(str).str.strip()

    if "DESCRICAO" not in df.columns:
        df["DESCRICAO"] = ""
    df["DESCRICAO"] = df["DESCRICAO"].fillna("").astype(str)

    if "CASOS" not in df.columns:
        df["CASOS"] = 1
    df["CASOS"] = pd.to_numeric(df["CASOS"], errors="coerce").fillna(0)

    if "ESTADO" not in df.columns:
        df["ESTADO"] = "Não informado"
    df["ESTADO"] = df["ESTADO"].fillna("Não informado").astype(str).str.strip().str.upper()

    if "TAMANHO_TEXTO" not in df.columns:
        df["TAMANHO_TEXTO"] = df["DESCRICAO"].str.len()
    df["TAMANHO_TEXTO"] = pd.to_numeric(df["TAMANHO_TEXTO"], errors="coerce").fillna(0)

    if "CATEGORIA_AJUSTADA" not in df.columns:
        df["CATEGORIA_AJUSTADA"] = "Não informada"
    df["CATEGORIA_AJUSTADA"] = df["CATEGORIA_AJUSTADA"].fillna("Não informada").astype(str).str.strip()

    # Padronização de estado
    mapa_estados = {
        "ACRE": "AC", "ALAGOAS": "AL", "AMAPÁ": "AP", "AMAPA": "AP",
        "AMAZONAS": "AM", "BAHIA": "BA", "CEARÁ": "CE", "CEARA": "CE",
        "DISTRITO FEDERAL": "DF", "ESPÍRITO SANTO": "ES", "ESPIRITO SANTO": "ES",
        "GOIÁS": "GO", "GOIAS": "GO", "MARANHÃO": "MA", "MARANHAO": "MA",
        "MATO GROSSO": "MT", "MATO GROSSO DO SUL": "MS", "MINAS GERAIS": "MG",
        "PARÁ": "PA", "PARA": "PA", "PARAÍBA": "PB", "PARAIBA": "PB",
        "PARANÁ": "PR", "PARANA": "PR", "PERNAMBUCO": "PE", "PIAUÍ": "PI",
        "PIAUI": "PI", "RIO DE JANEIRO": "RJ", "RIO GRANDE DO NORTE": "RN",
        "RIO GRANDE DO SUL": "RS", "RONDÔNIA": "RO", "RONDONIA": "RO",
        "RORAIMA": "RR", "SANTA CATARINA": "SC", "SÃO PAULO": "SP",
        "SAO PAULO": "SP", "SERGIPE": "SE", "TOCANTINS": "TO"
    }
    df["ESTADO"] = df["ESTADO"].replace(mapa_estados)

    # Remover estados inválidos
    estados_invalidos = {"--", "- - --", "NAN", "NONE", "", "NÃO INFORMADO", "NAO INFORMADO"}
    df = df[~df["ESTADO"].isin(estados_invalidos)].copy()

    # Datas
    if all(col in df.columns for col in ["ANO", "MES", "DIA"]):
        df["ANO"] = pd.to_numeric(df["ANO"], errors="coerce")
        df["MES"] = pd.to_numeric(df["MES"], errors="coerce")
        df["DIA"] = pd.to_numeric(df["DIA"], errors="coerce")

        df["DATA"] = pd.to_datetime(
            dict(year=df["ANO"], month=df["MES"], day=df["DIA"]),
            errors="coerce"
        )
    else:
        df["DATA"] = pd.NaT

    if df["DATA"].notna().any():
        df["ANO_MES"] = df["DATA"].dt.to_period("M").astype(str)
    else:
        df["ANO_MES"] = None

    bins = [0, 200, 500, 1000, 2000, 999999]
    labels = [
        "Muito curto (0-200)",
        "Curto (201-500)",
        "Médio (501-1000)",
        "Longo (1001-2000)",
        "Muito longo (2000+)"
    ]
    df["FAIXA_TEXTO"] = pd.cut(
        df["TAMANHO_TEXTO"],
        bins=bins,
        labels=labels,
        include_lowest=True
    )

    return df


@st.cache_data
def carregar_mapa_ibge():
    if not os.path.exists(SHP_PATH):
        raise FileNotFoundError(f"Shapefile não encontrado: {SHP_PATH}")

    gdf = gpd.read_file(SHP_PATH)
    gdf = gdf.set_geometry(gdf.geometry.name)

    # Renomeia colunas sem quebrar a geometria
    col_geom = gdf.geometry.name
    rename_map = {c: c.upper() for c in gdf.columns if c != col_geom}
    gdf = gdf.rename(columns=rename_map)

    if "SIGLA_UF" in gdf.columns:
        gdf["ESTADO"] = gdf["SIGLA_UF"].astype(str).str.upper()
    elif "SIGLA" in gdf.columns:
        gdf["ESTADO"] = gdf["SIGLA"].astype(str).str.upper()
    elif "NM_UF" in gdf.columns:
        nome_para_sigla = {
            "ACRE": "AC", "ALAGOAS": "AL", "AMAPÁ": "AP", "AMAPA": "AP",
            "AMAZONAS": "AM", "BAHIA": "BA", "CEARÁ": "CE", "CEARA": "CE",
            "DISTRITO FEDERAL": "DF", "ESPÍRITO SANTO": "ES", "ESPIRITO SANTO": "ES",
            "GOIÁS": "GO", "GOIAS": "GO", "MARANHÃO": "MA", "MARANHAO": "MA",
            "MATO GROSSO": "MT", "MATO GROSSO DO SUL": "MS", "MINAS GERAIS": "MG",
            "PARÁ": "PA", "PARA": "PA", "PARAÍBA": "PB", "PARAIBA": "PB",
            "PARANÁ": "PR", "PARANA": "PR", "PERNAMBUCO": "PE", "PIAUÍ": "PI",
            "PIAUI": "PI", "RIO DE JANEIRO": "RJ", "RIO GRANDE DO NORTE": "RN",
            "RIO GRANDE DO SUL": "RS", "RONDÔNIA": "RO", "RONDONIA": "RO",
            "RORAIMA": "RR", "SANTA CATARINA": "SC", "SÃO PAULO": "SP",
            "SAO PAULO": "SP", "SERGIPE": "SE", "TOCANTINS": "TO"
        }
        gdf["ESTADO"] = gdf["NM_UF"].astype(str).str.upper().map(nome_para_sigla)
    else:
        raise ValueError(f"Não encontrei coluna de UF no shapefile. Colunas: {gdf.columns.tolist()}")

    return gdf


def aplicar_filtros(df):
    st.sidebar.header("Filtros globais")

    anos = sorted([int(a) for a in df["ANO"].dropna().unique().tolist()]) if "ANO" in df.columns else []
    estados = sorted(df["ESTADO"].dropna().unique().tolist())
    status_list = sorted(df["STATUS"].dropna().unique().tolist())
    categorias = sorted(df["CATEGORIA_AJUSTADA"].dropna().unique().tolist())
    faixas = sorted([str(f) for f in df["FAIXA_TEXTO"].dropna().unique().tolist()])

    anos_sel = st.sidebar.multiselect("Ano", anos, default=anos)
    estados_sel = st.sidebar.multiselect("Estado", estados, default=estados)
    status_sel = st.sidebar.multiselect("Status", status_list, default=status_list)
    categorias_sel = st.sidebar.multiselect("Categoria ajustada", categorias, default=categorias)
    faixas_sel = st.sidebar.multiselect("Faixa do tamanho do texto", faixas, default=faixas)

    df_filtrado = df.copy()

    if anos_sel:
        df_filtrado = df_filtrado[df_filtrado["ANO"].isin(anos_sel)]
    if estados_sel:
        df_filtrado = df_filtrado[df_filtrado["ESTADO"].isin(estados_sel)]
    if status_sel:
        df_filtrado = df_filtrado[df_filtrado["STATUS"].isin(status_sel)]
    if categorias_sel:
        df_filtrado = df_filtrado[df_filtrado["CATEGORIA_AJUSTADA"].isin(categorias_sel)]
    if faixas_sel:
        df_filtrado = df_filtrado[df_filtrado["FAIXA_TEXTO"].astype(str).isin(faixas_sel)]

    return df_filtrado


def gerar_wordcloud(texto):
    stopwords_pt = {
        "de", "da", "do", "das", "dos", "e", "em", "a", "o", "as", "os",
        "para", "por", "com", "sem", "um", "uma", "que", "não", "na", "no",
        "nas", "nos", "ao", "aos", "às", "como", "mais", "menos", "já",
        "foi", "ser", "tem", "tinha", "meu", "minha", "seu", "sua", "pra",
        "porque", "quando", "onde", "sobre", "muito", "muita", "muitas",
        "muitos", "ainda", "após", "depois", "antes"
    }

    wc = WordCloud(
        width=1200,
        height=500,
        background_color="white",
        stopwords=STOPWORDS.union(stopwords_pt)
    ).generate(texto)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    return fig


try:
    df = carregar_dados()
    gdf_mapa = carregar_mapa_ibge()
except Exception as e:
    st.error(f"Erro ao carregar os dados: {e}")
    st.stop()

df_filtrado = aplicar_filtros(df)

st.title("Dashboard - Reclame Aqui Hapvida")
st.markdown("Análise exploratória das reclamações com filtros globais e visualizações interativas.")

if df_filtrado.empty:
    st.warning("Os filtros selecionados não retornaram registros.")
    st.stop()

total_reclamacoes = int(df_filtrado["CASOS"].sum())
total_estados = df_filtrado["ESTADO"].nunique()
status_mais_comum = df_filtrado["STATUS"].mode()[0] if not df_filtrado.empty else "-"
media_tamanho = round(df_filtrado["TAMANHO_TEXTO"].mean(), 1)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total de reclamações", total_reclamacoes)
c2.metric("Estados com ocorrência", total_estados)
c3.metric("Status mais comum", status_mais_comum)
c4.metric("Média de caracteres", media_tamanho)

st.subheader("Evolução temporal das reclamações")

if df_filtrado["DATA"].notna().any():
    serie = (
        df_filtrado.dropna(subset=["DATA"])
        .groupby("DATA", as_index=False)["CASOS"]
        .sum()
        .sort_values("DATA")
    )
    serie["MEDIA_MOVEL_7"] = serie["CASOS"].rolling(7, min_periods=1).mean()

    fig_tempo = go.Figure()
    fig_tempo.add_trace(go.Scatter(
        x=serie["DATA"],
        y=serie["CASOS"],
        mode="lines+markers",
        name="Casos"
    ))
    fig_tempo.add_trace(go.Scatter(
        x=serie["DATA"],
        y=serie["MEDIA_MOVEL_7"],
        mode="lines",
        name="Média móvel (7)"
    ))
    fig_tempo.update_layout(
        height=420,
        xaxis_title="Data",
        yaxis_title="Quantidade"
    )
    st.plotly_chart(fig_tempo, width="stretch")
else:
    st.info("Não há datas válidas para gerar a série temporal.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Mapa de reclamações por estado")

    mapa_df = (
        df_filtrado.groupby("ESTADO", as_index=False)["CASOS"]
        .sum()
    )

    gdf_plot = gdf_mapa.merge(mapa_df, on="ESTADO", how="left")
    gdf_plot["CASOS"] = gdf_plot["CASOS"].fillna(0)

    fig_mapa = px.choropleth(
        gdf_plot,
        geojson=gdf_plot.geometry.__geo_interface__,
        locations=gdf_plot.index,
        color="CASOS",
        hover_name="ESTADO",
        projection="mercator",
        color_continuous_scale="Reds"
    )
    fig_mapa.update_geos(fitbounds="locations", visible=False)
    fig_mapa.update_layout(height=500, margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig_mapa, width="stretch")

with col2:
    st.subheader("Ranking de reclamações por estado")

    ranking_estados = (
        df_filtrado.groupby("ESTADO", as_index=False)["CASOS"]
        .sum()
        .sort_values("CASOS", ascending=False)
    )

    fig_estados = px.bar(
        ranking_estados,
        x="ESTADO",
        y="CASOS",
        text_auto=True
    )
    fig_estados.update_layout(
        height=500,
        xaxis_title="Estado",
        yaxis_title="Casos"
    )
    st.plotly_chart(fig_estados, width="stretch")

col3, col4 = st.columns(2)

with col3:
    st.subheader("Distribuição por status")

    status_df = (
        df_filtrado.groupby("STATUS", as_index=False)["CASOS"]
        .sum()
        .sort_values("CASOS", ascending=False)
    )

    fig_status = px.pie(
        status_df,
        names="STATUS",
        values="CASOS",
        hole=0.4
    )
    fig_status.update_layout(height=450)
    st.plotly_chart(fig_status, width="stretch")

with col4:
    st.subheader("Categorias ajustadas mais frequentes")

    cat_df = (
        df_filtrado.groupby("CATEGORIA_AJUSTADA", as_index=False)["CASOS"]
        .sum()
        .sort_values("CASOS", ascending=False)
        .head(10)
    )

    fig_cat = px.bar(
        cat_df,
        x="CASOS",
        y="CATEGORIA_AJUSTADA",
        orientation="h",
        text_auto=True
    )
    fig_cat.update_layout(
        height=450,
        xaxis_title="Casos",
        yaxis_title="Categoria"
    )
    fig_cat.update_yaxes(categoryorder="total ascending")
    st.plotly_chart(fig_cat, width="stretch")

col5, col6 = st.columns(2)

with col5:
    st.subheader("Tamanho do texto por status")

    fig_box = px.box(
        df_filtrado,
        x="STATUS",
        y="TAMANHO_TEXTO",
        points="outliers"
    )
    fig_box.update_layout(
        height=450,
        xaxis_title="Status",
        yaxis_title="Quantidade de caracteres"
    )
    st.plotly_chart(fig_box, width="stretch")

with col6:
    st.subheader("Distribuição do tamanho das reclamações")

    fig_hist = px.histogram(
        df_filtrado,
        x="TAMANHO_TEXTO",
        color="STATUS",
        nbins=30,
        marginal="box"
    )
    fig_hist.update_layout(
        height=450,
        xaxis_title="Quantidade de caracteres",
        yaxis_title="Frequência"
    )
    st.plotly_chart(fig_hist, width="stretch")

st.subheader("Evolução mensal por status")

if df_filtrado["ANO_MES"].notna().any():
    mensal_status = (
        df_filtrado.groupby(["ANO_MES", "STATUS"], as_index=False)["CASOS"]
        .sum()
        .sort_values("ANO_MES")
    )

    fig_mensal = px.line(
        mensal_status,
        x="ANO_MES",
        y="CASOS",
        color="STATUS",
        markers=True
    )
    fig_mensal.update_layout(
        height=420,
        xaxis_title="Ano-Mês",
        yaxis_title="Casos"
    )
    st.plotly_chart(fig_mensal, width="stretch")
else:
    st.info("Não há datas válidas para gerar a evolução mensal.")

st.subheader("WordCloud das descrições")

texto_total = " ".join(df_filtrado["DESCRICAO"].dropna().astype(str).tolist()).strip()

if texto_total:
    fig_wc = gerar_wordcloud(texto_total)
    st.pyplot(fig_wc)
else:
    st.info("Não há textos suficientes para gerar a WordCloud.")


st.divider() # Adiciona uma linha visual para separar as seções
st.subheader("Análise Operacional e Regional")

col7, col8 = st.columns(2)

# 1. Gráfico de Cidades
with col7:
    st.markdown("**Top 10 Cidades com Mais Reclamações**")
    if "LOCAL" in df_filtrado.columns:
        df_cidades = df_filtrado.groupby('LOCAL', as_index=False)['CASOS'].sum()
        df_cidades = df_cidades.sort_values('CASOS', ascending=False).head(10)
        df_cidades = df_cidades.sort_values('CASOS', ascending=True) # Para o Plotly exibir o maior no topo

        fig_cid = px.bar(
            df_cidades, x='CASOS', y='LOCAL', orientation='h', text_auto=True,
            color_discrete_sequence=['#ef553b']
        )
        fig_cid.update_layout(height=400, xaxis_title="Casos", yaxis_title="", margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_cid, use_container_width=True)

# 2. Gráfico de Dia da Semana
with col8:
    st.markdown("**Reclamações por Dia da Semana**")
    if "DIA_DA_SEMANA" in df_filtrado.columns:
        mapa_dias = {0: 'Segunda', 1: 'Terça', 2: 'Quarta', 3: 'Quinta', 4: 'Sexta', 5: 'Sábado', 6: 'Domingo'}
        df_dias = df_filtrado.groupby('DIA_DA_SEMANA', as_index=False)['CASOS'].sum()
        df_dias['Nome do Dia'] = df_dias['DIA_DA_SEMANA'].map(mapa_dias)

        fig_dias = px.bar(
            df_dias, x='Nome do Dia', y='CASOS', text_auto=True,
            category_orders={"Nome do Dia": ["Segunda", "Terça", "Quarta", "Quinta", "Sexta", "Sábado", "Domingo"]},
            color_discrete_sequence=['#636efa']
        )
        fig_dias.update_layout(height=400, xaxis_title="", yaxis_title="Casos", margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_dias, use_container_width=True)


# 3. Análise Profunda das Dores (Colunas de Problemas Específicos)
st.subheader("Principais Queixas e Dores dos Clientes")

colunas_problemas = [
    'Administrativo', 'Atraso na entrega', 'Clínicas Médicas', 'Cobrança indevida',
    'Demora para autorização de consultas, exames e cirurgias',
    'Dificuldade para agendamento de exames-consultas', 'Exames Lab e imagens',
    'Mau Atendimento', 'Problemas de infraestrutura', 'Qualidade do serviço',
    'Qualidade do serviço prestado', 'Rede de Atendimento', 'Reembolso de pagamento'
]

# Pega apenas as colunas que existem no dataframe para evitar erros
colunas_existentes = [col for col in colunas_problemas if col in df_filtrado.columns]

if colunas_existentes:
    df_problemas = df_filtrado[colunas_existentes].sum().reset_index()
    df_problemas.columns = ["Problema", "Total de Casos"]
    df_problemas = df_problemas[df_problemas["Total de Casos"] > 0].sort_values("Total de Casos", ascending=True)

    fig_prob = px.bar(
        df_problemas, x="Total de Casos", y="Problema", orientation="h", text_auto=True,
        color='Total de Casos', color_continuous_scale='Reds'
    )
    fig_prob.update_layout(height=500, xaxis_title="Quantidade de Casos Reportados", yaxis_title="", showlegend=False)
    st.plotly_chart(fig_prob, use_container_width=True)    

st.subheader("Amostra dos dados filtrados")

colunas_tabela = [
    c for c in [
        "DATA", "ESTADO", "LOCAL", "TEMA", "STATUS", "CATEGORIA_AJUSTADA", 
        "TAMANHO_TEXTO", "DESCRICAO", "URL"
    ] if c in df_filtrado.columns
]

df_tabela = df_filtrado[colunas_tabela].copy()

if "DATA" in df_tabela.columns:
    df_tabela = df_tabela.sort_values("DATA", ascending=False)

# Usa o column_config do Streamlit para criar a coluna com Links Clicáveis
st.dataframe(
    df_tabela.head(50), 
    use_container_width=True,
    hide_index=True,
    column_config={
        "URL": st.column_config.LinkColumn("Link Direto", display_text="Acessar Reclamação"),
        "DATA": st.column_config.DateColumn("Data da Ocorrência", format="DD/MM/YYYY"),
        "TEMA": "Título da Reclamação",
        "LOCAL": "Cidade"
    }
)