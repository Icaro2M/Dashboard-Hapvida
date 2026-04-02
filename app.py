import os
import streamlit as st
import pandas as pd
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import unicodedata
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

    # Tratamento de Datas
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
        # Extração para lógica semanal
        df["NUM_SEMANA"] = df["DATA"].dt.isocalendar().week.astype(int)
        df["ANO_EXTRAIDO"] = df["DATA"].dt.year.astype(int)
    else:
        df["ANO_MES"] = None
        df["NUM_SEMANA"] = None
        df["ANO_EXTRAIDO"] = None

    bins = [0, 200, 500, 1000, 2000, 999999]
    labels = ["Muito curto (0-200)", "Curto (201-500)", "Médio (501-1000)", "Longo (1001-2000)", "Muito longo (2000+)"]
    df["FAIXA_TEXTO"] = pd.cut(df["TAMANHO_TEXTO"], bins=bins, labels=labels, include_lowest=True)

    return df

@st.cache_data
def carregar_mapa_ibge():
    if not os.path.exists(SHP_PATH):
        raise FileNotFoundError(f"Shapefile não encontrado: {SHP_PATH}")
    gdf = gpd.read_file(SHP_PATH)
    col_geom = gdf.geometry.name
    rename_map = {c: c.upper() for c in gdf.columns if c != col_geom}
    gdf = gdf.rename(columns=rename_map)
    if "SIGLA_UF" in gdf.columns:
        gdf["ESTADO"] = gdf["SIGLA_UF"].astype(str).str.upper()
    return gdf

def aplicar_filtros(df):
    st.sidebar.header("Filtros globais")
    anos = sorted([int(a) for a in df["ANO"].dropna().unique().tolist()])
    estados = sorted(df["ESTADO"].dropna().unique().tolist())
    status_list = sorted(df["STATUS"].dropna().unique().tolist())
    
    anos_sel = st.sidebar.multiselect("Ano", anos, default=anos)
    estados_sel = st.sidebar.multiselect("Estado", estados, default=estados)
    status_sel = st.sidebar.multiselect("Status", status_list, default=status_list)

    df_filtrado = df.copy()
    if anos_sel:
        df_filtrado = df_filtrado[df_filtrado["ANO"].isin(anos_sel)]
    if estados_sel:
        df_filtrado = df_filtrado[df_filtrado["ESTADO"].isin(estados_sel)]
    if status_sel:
        df_filtrado = df_filtrado[df_filtrado["STATUS"].isin(status_sel)]

    return df_filtrado

def gerar_wordcloud(texto):
    def sem_acentos(palavra):
        return "".join(
            c for c in unicodedata.normalize("NFKD", palavra)
            if not unicodedata.combining(c)
        )

    # Bloco de Stopwords fornecido exatamente como solicitado
    stopwords_pt = {
        "de", "da", "do", "das", "dos", "e", "em", "a", "o", "as", "os",
        "para", "por", "com", "sem", "um", "uma", "que", "não", "na", "no",
        "nas", "nos", "ao", "aos", "às", "como", "mais", "menos", "já",
        "foi", "ser", "tem", "tinha", "meu", "minha", "seu", "sua", "pra",
        "porque", "quando", "onde", "sobre", "muito", "muita", "muitas",
        "muitos", "ainda", "após", "depois", "antes", "é", "sendo", "só", 
        "também", "tb", "q", "pq", "onde", "ainda", "após", "antes", "sendo",
        "vc", "você", "voces", "voce", "voces","então", "entao", "assim", "assim como", 
        "além disso", "além de", "além",
    }

    stopwords_pt_base = {
        "a", "à", "ao", "aos", "aquela", "aquelas", "aquele", "aqueles", "aquilo",
        "as", "até", "com", "como", "da", "das", "de", "dela", "delas", "dele", "deles",
        "depois", "do", "dos", "e", "ela", "elas", "ele", "eles", "em", "entre",
        "era", "eram", "essa", "essas", "esse", "esses", "esta", "está", "estão",
        "estamos", "estar", "estas", "estava", "estavam", "este", "esteja", "estejam",
        "estejamos", "estes", "esteve", "estive", "estivemos", "estiveram", "estiver",
        "estivera", "estiverem", "estivermos", "estivesse", "estivessem", "estou",
        "eu", "foi", "fomos", "for", "fora", "foram", "forem", "formos", "fosse",
        "fossem", "fui", "há", "haja", "hajam", "hajamos", "haver", "hei", "houve",
        "houvemos", "houveram", "houver", "houvera", "houverei", "houverem",
        "houveremos", "houveria", "houveriam", "houvermos", "houvesse", "houvessem",
        "isso", "isto", "já", "lhe", "lhes", "mais", "mas", "me", "mesmo", "meu",
        "meus", "minha", "minhas", "muito", "muitos", "muita", "muitas", "na", "não",
        "nas", "nem", "no", "nos", "nossa", "nossas", "nosso", "nossos", "num", "numa",
        "o", "os", "ou", "para", "pela", "pelas", "pelo", "pelos", "por", "qual",
        "quando", "que", "quem", "se", "sem", "ser", "seu", "seus", "sua", "suas",
        "também", "te", "tem", "tendo", "tenho", "ter", "tinha", "tinham", "tive",
        "tivemos", "tiveram", "todo", "todos", "toda", "todas", "tua", "tuas", "um",
        "uma", "você", "vocês", "vos", "nós", "às", "sobre", "porque", "pra", "pro",
        "pros", "pras", "nesse", "nessa", "nesses", "nessas", "neste", "nesta",
        "nestes", "nestas", "daqui", "dali", "daquele", "daqueles", "daquela",
        "daquelas", "dessa", "desse", "dessas", "desses", "aí", "aqui", "ali", "lá",
        "cá", "cada", "algum", "alguma", "alguns", "algumas", "outro", "outra",
        "outros", "outras", "pouco", "pouca", "poucos", "poucas", "mesma", "mesmas",
        "mesmos", "só", "tambem", "nao", "voce", "voces", "tb", "q", "pq", "onde",
        "ainda", "após", "antes", "sendo", "muitos", "ainda", 
        "após", "depois", "antes", "é", "sendo", "só", 
        "também", "tb", "q", "pq", "onde", "ainda", "após", "antes", "sendo",
        "vc", "você", "voces", "voce", "voces","então", "entao", "assim", "assim como", 
        "além disso", "além de", "além", "fiz", "fazer", "vai", "vão", "vou", "vamos", "vão", 
        "fizemos", "fizeram", "fizer", "fizerem", "vez", "ir", "será", "serão", "seremos", "seria", 
        "seriam", "estará", "estarão", "estaremos", "quase", "quais", "qualquer", "quanta", 
        "quantas", "quanto", "quantos", "sabe", "sabem", "sou"
    }

    stopwords_pt |= stopwords_pt_base
    stopwords_pt |= {sem_acentos(p) for p in stopwords_pt_base}

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

# --- LÓGICA DO DASHBOARD ---

try:
    df = carregar_dados()
    gdf_mapa = carregar_mapa_ibge()
except Exception as e:
    st.error(f"Erro ao carregar dados: {e}")
    st.stop()

df_filtrado = aplicar_filtros(df)

st.title("Dashboard - Reclame Aqui Hapvida")

# Métricas Principais
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total de reclamações", int(df_filtrado["CASOS"].sum()))
c2.metric("Estados com ocorrência", df_filtrado["ESTADO"].nunique())
c3.metric("Status mais comum", df_filtrado["STATUS"].mode()[0] if not df_filtrado.empty else "-")
c4.metric("Média de caracteres", round(df_filtrado["TAMANHO_TEXTO"].mean(), 1))

# Gráficos de Mapa e Ranking
col1, col2 = st.columns(2)
with col1:
    st.subheader("Mapa de Reclamações")
    mapa_df = df_filtrado.groupby("ESTADO", as_index=False)["CASOS"].sum()
    gdf_plot = gdf_mapa.merge(mapa_df, on="ESTADO", how="left").fillna(0)
    fig_mapa = px.choropleth(gdf_plot, geojson=gdf_plot.geometry.__geo_interface__, locations=gdf_plot.index, color="CASOS", hover_name="ESTADO", color_continuous_scale="Reds")
    fig_mapa.update_geos(fitbounds="locations", visible=False)
    st.plotly_chart(fig_mapa, use_container_width=True)

with col2:
    st.subheader("Ranking por Estado")
    ranking = df_filtrado.groupby("ESTADO", as_index=False)["CASOS"].sum().sort_values("CASOS", ascending=False)
    st.plotly_chart(px.bar(ranking, x="ESTADO", y="CASOS", text_auto=True), use_container_width=True)

# --- EVOLUÇÃO SEMANAL (ANO E SEMANAS COMPLETAS) ---
st.divider()
st.subheader("📊 Evolução Semanal por Status")

if df_filtrado["DATA"].notna().any():
    anos_disponiveis = sorted(df_filtrado["ANO_EXTRAIDO"].unique())
    ano_selecionado = st.selectbox("Selecione o ano para detalhamento semanal", anos_disponiveis, index=len(anos_disponiveis)-1)
    
    df_ano = df_filtrado[df_filtrado["ANO_EXTRAIDO"] == ano_selecionado]
    
    semanal_status = df_ano.groupby(["NUM_SEMANA", "STATUS"], as_index=False)["CASOS"].sum()

    # Garantir que todas as semanas do ano apareçam no gráfico (1 a 52)
    todas_semanas = list(range(1, 53))
    todos_status = df_filtrado["STATUS"].unique()
    grid = pd.MultiIndex.from_product([todas_semanas, todos_status], names=["NUM_SEMANA", "STATUS"]).to_frame(index=False)
    df_plot_semanal = pd.merge(grid, semanal_status, on=["NUM_SEMANA", "STATUS"], how="left").fillna(0)

    fig_semanal = px.line(
        df_plot_semanal,
        x="NUM_SEMANA",
        y="CASOS",
        color="STATUS",
        markers=True,
        labels={"NUM_SEMANA": "Semana do Ano", "CASOS": "Total de Casos"}
    )

    fig_semanal.update_layout(
        xaxis=dict(
            tickmode='linear',
            tick0=1,
            dtick=1,
            range=[0.5, 52.5],
            title_standoff=15
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=500
    )
    st.plotly_chart(fig_semanal, use_container_width=True)

# WordCloud e Tabela
st.divider()
st.subheader("Nuvem de Palavras (Descrições)")
texto_total = " ".join(df_filtrado["DESCRICAO"].dropna().astype(str).tolist()).strip()
if texto_total:
    st.pyplot(gerar_wordcloud(texto_total))
else:
    st.info("Texto insuficiente para gerar nuvem.")

st.subheader("Amostra de Dados Filtrados")
st.dataframe(df_filtrado[["DATA", "ESTADO", "STATUS", "DESCRICAO"]].head(50), use_container_width=True)