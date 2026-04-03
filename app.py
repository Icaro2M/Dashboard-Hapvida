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

MESES_ORDEM = [
    "Jan", "Fev", "Mar", "Abr", "Mai", "Jun",
    "Jul", "Ago", "Set", "Out", "Nov", "Dez",
]

CATEGORY_COLUMNS = [
    "ADMINISTRATIVO",
    "ATRASO NA ENTREGA",
    "CLÍNICAS MÉDICAS",
    "COBRANÇA INDEVIDA",
    "DEMORA PARA AUTORIZAÇÃO DE CONSULTAS, EXAMES E CIRURGIAS",
    "DIFICULDADE PARA AGENDAMENTO DE EXAMES-CONSULTAS",
    "EXAMES LAB E IMAGENS",
    "MAU ATENDIMENTO",
    "PROBLEMAS DE INFRAESTRUTURA",
    "QUALIDADE DO SERVIÇO",
    "QUALIDADE DO SERVIÇO PRESTADO",
    "REDE DE ATENDIMENTO",
    "REEMBOLSO DE PAGAMENTO",
]


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

    df["ANO_SEMANA"] = pd.NA
    if df["DATA"].notna().any():
        df["ANO_MES"] = df["DATA"].dt.to_period("M").astype(str)
        mask_data = df["DATA"].notna()
        iso = df.loc[mask_data, "DATA"].dt.isocalendar()
        df.loc[mask_data, "ANO_SEMANA"] = (
            iso["year"].astype(str) + "-S" + iso["week"].astype(str).str.zfill(2)
        )
    else:
        df["ANO_MES"] = None

    # Remoção de outliers via IQR em TAMANHO_TEXTO (antes de qualquer estatística/gráfico)
    df = df.dropna(subset=["TAMANHO_TEXTO"])
    q1 = df["TAMANHO_TEXTO"].quantile(0.25)
    q3 = df["TAMANHO_TEXTO"].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    df = df[
        (df["TAMANHO_TEXTO"] >= lower_bound)
        & (df["TAMANHO_TEXTO"] <= upper_bound)
    ].copy()

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

    for col in CATEGORY_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

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
    faixas = sorted([str(f) for f in df["FAIXA_TEXTO"].dropna().unique().tolist()])
    categorias_disponiveis = [col for col in CATEGORY_COLUMNS if col in df.columns]

    anos_sel = st.sidebar.multiselect("Ano", anos, default=anos)
    estados_sel = st.sidebar.multiselect("Estado", estados, default=estados)
    status_sel = st.sidebar.multiselect("Status", status_list, default=status_list)
    categorias_sel = st.sidebar.multiselect("Categoria", categorias_disponiveis, default=[])
    faixas_sel = st.sidebar.multiselect("Faixa do tamanho do texto", faixas, default=faixas)

    df_filtrado = df.copy()

    if anos_sel:
        df_filtrado = df_filtrado[df_filtrado["ANO"].isin(anos_sel)]
    if estados_sel:
        df_filtrado = df_filtrado[df_filtrado["ESTADO"].isin(estados_sel)]
    if status_sel:
        df_filtrado = df_filtrado[df_filtrado["STATUS"].isin(status_sel)]
    if categorias_sel:
        mascara_categoria = df_filtrado[categorias_sel].sum(axis=1) > 0
        df_filtrado = df_filtrado[mascara_categoria]
    if faixas_sel:
        df_filtrado = df_filtrado[df_filtrado["FAIXA_TEXTO"].astype(str).isin(faixas_sel)]

    return df_filtrado


stopwords_pt = {
    "é",
    "de", "a", "o", "que", "e", "do", "da", "em", "um", "para", "com",
    "não", "uma", "os", "no", "se", "na", "por", "mais", "as", "dos", "como",
    "mas", "foi", "ao", "ele", "das", "tem", "à", "seu", "sua", "ou", "ser",
    "quando", "muito", "há", "nos", "já", "está", "eu", "também", "só", "pelo",
    "pela", "até", "isso", "ela", "entre", "era", "depois", "sem", "mesmo",
    "aos", "ter", "seus", "quem", "nas", "me", "esse", "eles", "estão", "você",
    "tinha", "foram", "essa", "num", "nem", "suas", "meu", "às", "minha",
    "têm", "numa", "pelos", "elas", "havia", "seja", "qual", "será", "nós",
    "tenho", "lhe", "deles", "essas", "esses", "pelas", "este", "fosse",
    "dele", "tu", "te", "vocês", "vos", "lhes", "meus", "minhas", "teu",
    "tua", "teus", "tuas", "nosso", "nossa", "nossos", "nossas", "dela",
    "delas", "esta", "estes", "estas", "aquele", "aquela", "aqueles",
    "aquelas", "isto", "aquilo", "estou", "está", "estamos", "estão",
    "estive", "esteve", "estivemos", "estiveram", "estava", "estávamos",
    "estavam", "estivera", "estivéramos", "esteja", "estejamos", "estejam",
    "estivesse", "estivéssemos", "estivessem", "estiver", "estivermos",
    "estiverem", "hei", "há", "havemos", "hão", "houve", "houvemos",
    "houveram", "houvera", "houvéramos", "haja", "hajamos", "hajam",
    "houvesse", "houvéssemos", "houvessem", "houver", "houvermos",
    "houverem", "houverei", "houverá", "houveremos", "houverão",
    "houveria", "houveríamos", "houveriam", "sou", "somos", "são", "era",
    "éramos", "eram", "fui", "fomos", "foram", "fora", "fôramos",
    "ainda", "dia", "dias", "pois", "onde", "todo", "toda", "todos", "todas",
    "nao", "porque", "sobre", "pode", "fazer", "vez", "bem", "aqui",
}

def gerar_wordcloud(texto):
    todas_stopwords = STOPWORDS.union(stopwords_pt)
    wc = WordCloud(
        width=1200,
        height=500,
        background_color="white",
        stopwords=todas_stopwords,
    ).generate(texto.lower())

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
    serie["MEDIA_MOVEL_3"] = serie["CASOS"].rolling(3, min_periods=1).mean()

    fig_tempo = go.Figure()
    fig_tempo.add_trace(go.Scatter(
        x=serie["DATA"],
        y=serie["CASOS"],
        mode="lines+markers",
        name="Casos"
    ))
    fig_tempo.add_trace(go.Scatter(
        x=serie["DATA"],
        y=serie["MEDIA_MOVEL_3"],
        mode="lines",
        name="Média móvel (3)"
    ))
    fig_tempo.update_layout(
        height=420,
        xaxis_title="Data",
        yaxis_title="Quantidade"
    )
    st.plotly_chart(fig_tempo, width="stretch")
else:
    st.info("Não há datas válidas para gerar a série temporal.")

st.subheader("Casos por mês do ano (Jan–Dez)")

if "MES" in df_filtrado.columns and df_filtrado["MES"].notna().any():
    df_mes = df_filtrado.copy()
    df_mes["MES"] = pd.to_numeric(df_mes["MES"], errors="coerce")
    df_mes = df_mes[df_mes["MES"].between(1, 12)]
    if not df_mes.empty:
        por_mes = df_mes.groupby("MES", as_index=False)["CASOS"].sum()
        base_meses = pd.DataFrame({"MES": range(1, 13)})
        por_mes = base_meses.merge(por_mes, on="MES", how="left").fillna({"CASOS": 0})
        por_mes["MÊS"] = por_mes["MES"].astype(int).map(lambda m: MESES_ORDEM[m - 1])
        fig_meses_ano = px.bar(
            por_mes,
            x="MÊS",
            y="CASOS",
            text_auto=True,
            category_orders={"MÊS": MESES_ORDEM},
        )
        fig_meses_ano.update_layout(
            height=420,
            xaxis_title="Mês",
            yaxis_title="Casos (soma no período filtrado)",
        )
        st.plotly_chart(fig_meses_ano, width="stretch")
    else:
        st.info("Não há registros com mês válido (1–12).")
else:
    st.info("Não há coluna Mês nos dados filtrados.")

st.subheader("Distribuição de casos por semana (ISO)")

if df_filtrado["ANO_SEMANA"].notna().any():
    semana_df = (
        df_filtrado.dropna(subset=["ANO_SEMANA"])
        .groupby("ANO_SEMANA", as_index=False)["CASOS"]
        .sum()
        .sort_values("ANO_SEMANA")
    )
    fig_semana = px.bar(
        semana_df,
        x="ANO_SEMANA",
        y="CASOS",
        text_auto=True,
    )
    fig_semana.update_layout(
        height=420,
        xaxis_title="Ano e semana (ex.: 2022-S01 = semana 1 do ano)",
        yaxis_title="Casos",
    )
    fig_semana.update_xaxes(tickangle=45)
    st.plotly_chart(fig_semana, width="stretch")
else:
    st.info("Não há datas válidas para agrupar por semana.")

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
    st.subheader("Categorias mais frequentes")

    categorias_validas = [col for col in CATEGORY_COLUMNS if col in df_filtrado.columns]

    cat_df = (
        df_filtrado[categorias_validas]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )
    cat_df.columns = ["CATEGORIA", "CASOS"]

    rotulo_categoria = {
        "ADMINISTRATIVO": "Administrativo",
        "ATRASO NA ENTREGA": "Atraso na entrega",
        "CLÍNICAS MÉDICAS": "Clínicas médicas",
        "COBRANÇA INDEVIDA": "Cobrança indevida",
        "DEMORA PARA AUTORIZAÇÃO DE CONSULTAS, EXAMES E CIRURGIAS": "Demora para autorização de consultas, exames e cirurgias",
        "DIFICULDADE PARA AGENDAMENTO DE EXAMES-CONSULTAS": "Dificuldade para agendamento de exames-consultas",
        "EXAMES LAB E IMAGENS": "Exames lab e imagens",
        "MAU ATENDIMENTO": "Mau atendimento",
        "PROBLEMAS DE INFRAESTRUTURA": "Problemas de infraestrutura",
        "QUALIDADE DO SERVIÇO": "Qualidade do serviço",
        "QUALIDADE DO SERVIÇO PRESTADO": "Qualidade do serviço prestado",
        "REDE DE ATENDIMENTO": "Rede de atendimento",
        "REEMBOLSO DE PAGAMENTO": "Reembolso de pagamento",
    }
    cat_df["CATEGORIA"] = cat_df["CATEGORIA"].map(rotulo_categoria).fillna(cat_df["CATEGORIA"])

    fig_cat = px.bar(
        cat_df,
        x="CASOS",
        y="CATEGORIA",
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
    st.subheader("Análise de Densidade: Tamanho do Texto por Status")

    fig_violin = px.violin(
        df_filtrado,
        x="STATUS",
        y="TAMANHO_TEXTO",
        color="STATUS",
        color_discrete_sequence=px.colors.sequential.Viridis,
        box=True,
    )
    fig_violin.update_layout(
        height=500,
        xaxis_title="Status da Reclamação",
        yaxis_title="Quantidade de Caracteres",
        showlegend=False,
    )
    fig_violin.update_xaxes(tickangle=45)
    st.plotly_chart(fig_violin, width="stretch")

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

if (
    df_filtrado["ANO_MES"].notna().any()
    and df_filtrado["DATA"].notna().any()
):
    mensal_status = (
        df_filtrado.groupby(["ANO_MES", "STATUS"], as_index=False)["CASOS"]
        .sum()
    )
    dmin = df_filtrado["DATA"].min()
    dmax = df_filtrado["DATA"].max()
    todas_meses = pd.period_range(
        dmin.to_period("M"),
        dmax.to_period("M"),
        freq="M",
    )
    todas_str = [str(p) for p in todas_meses]
    statuses = sorted(df_filtrado["STATUS"].dropna().unique().tolist())
    if statuses and todas_str:
        idx = pd.MultiIndex.from_product(
            [todas_str, statuses],
            names=["ANO_MES", "STATUS"],
        )
        mensal_status = (
            mensal_status.set_index(["ANO_MES", "STATUS"])
            .reindex(idx, fill_value=0)
            .reset_index()
            .sort_values(["ANO_MES", "STATUS"])
        )
        fig_mensal = px.line(
            mensal_status,
            x="ANO_MES",
            y="CASOS",
            color="STATUS",
            markers=True,
            category_orders={"ANO_MES": todas_str},
        )
        fig_mensal.update_layout(
            height=420,
            xaxis_title="Ano-Mês",
            yaxis_title="Casos",
        )
        fig_mensal.update_xaxes(type="category", tickangle=45)
        st.plotly_chart(fig_mensal, width="stretch")
    else:
        st.info("Não há status ou meses suficientes para a evolução mensal.")
else:
    st.info("Não há datas válidas para gerar a evolução mensal.")

st.subheader("WordCloud das descrições")

texto_total = " ".join(df_filtrado["DESCRICAO"].dropna().astype(str).tolist()).strip()

if texto_total:
    fig_wc = gerar_wordcloud(texto_total)
    st.pyplot(fig_wc)
else:
    st.info("Não há textos suficientes para gerar a WordCloud.")

st.divider()
st.subheader("Análise Operacional e Regional")

col7, col8 = st.columns(2)

with col7:
    st.markdown("**Top 10 cidades com mais reclamações**")
    if "LOCAL" in df_filtrado.columns:
        df_cidades = df_filtrado.groupby("LOCAL", as_index=False)["CASOS"].sum()
        df_cidades = df_cidades.sort_values("CASOS", ascending=False).head(10)
        df_cidades = df_cidades.sort_values("CASOS", ascending=True)
        fig_cid = px.bar(
            df_cidades,
            x="CASOS",
            y="LOCAL",
            orientation="h",
            text_auto=True,
            color_discrete_sequence=["#ef553b"],
        )
        fig_cid.update_layout(
            height=400,
            xaxis_title="Casos",
            yaxis_title="",
            margin=dict(l=0, r=0, t=30, b=0),
        )
        st.plotly_chart(fig_cid, width="stretch")
    else:
        st.info("Coluna LOCAL não disponível nos dados.")

with col8:
    st.markdown("**Reclamações por dia da semana**")
    if "DIA_DA_SEMANA" in df_filtrado.columns:
        mapa_dias = {
            0: "Segunda",
            1: "Terça",
            2: "Quarta",
            3: "Quinta",
            4: "Sexta",
            5: "Sábado",
            6: "Domingo",
        }
        df_dias = df_filtrado.copy()
        df_dias["DIA_DA_SEMANA"] = pd.to_numeric(df_dias["DIA_DA_SEMANA"], errors="coerce")
        df_dias = df_dias.dropna(subset=["DIA_DA_SEMANA"])
        if not df_dias.empty:
            df_dias["DIA_DA_SEMANA"] = df_dias["DIA_DA_SEMANA"].astype(int)
            df_dias = df_dias.groupby("DIA_DA_SEMANA", as_index=False)["CASOS"].sum()
            df_dias["Nome do Dia"] = df_dias["DIA_DA_SEMANA"].map(mapa_dias)
            df_dias = df_dias.dropna(subset=["Nome do Dia"])
            ordem_dias = [
                "Segunda", "Terça", "Quarta", "Quinta",
                "Sexta", "Sábado", "Domingo",
            ]
            fig_dias = px.bar(
                df_dias,
                x="Nome do Dia",
                y="CASOS",
                text_auto=True,
                category_orders={"Nome do Dia": ordem_dias},
                color_discrete_sequence=["#636efa"],
            )
            fig_dias.update_layout(
                height=400,
                xaxis_title="",
                yaxis_title="Casos",
                margin=dict(l=0, r=0, t=30, b=0),
            )
            st.plotly_chart(fig_dias, width="stretch")
        else:
            st.info("Sem valores válidos em DIA_DA_SEMANA.")
    else:
        st.info("Coluna DIA_DA_SEMANA não disponível nos dados.")

st.subheader("Principais queixas e dores dos clientes")

# Nomes das colunas dummy no CSV após normalização (maiúsculas em carregar_dados)
colunas_problemas_chave = [
    "ADMINISTRATIVO",
    "ATRASO NA ENTREGA",
    "CLÍNICAS MÉDICAS",
    "COBRANÇA INDEVIDA",
    "DEMORA PARA AUTORIZAÇÃO DE CONSULTAS, EXAMES E CIRURGIAS",
    "DIFICULDADE PARA AGENDAMENTO DE EXAMES-CONSULTAS",
    "EXAMES LAB E IMAGENS",
    "MAU ATENDIMENTO",
    "PROBLEMAS DE INFRAESTRUTURA",
    "QUALIDADE DO SERVIÇO",
    "QUALIDADE DO SERVIÇO PRESTADO",
    "REDE DE ATENDIMENTO",
    "REEMBOLSO DE PAGAMENTO",
]
rotulo_problema = {
    "ADMINISTRATIVO": "Administrativo",
    "ATRASO NA ENTREGA": "Atraso na entrega",
    "CLÍNICAS MÉDICAS": "Clínicas médicas",
    "COBRANÇA INDEVIDA": "Cobrança indevida",
    "DEMORA PARA AUTORIZAÇÃO DE CONSULTAS, EXAMES E CIRURGIAS": (
        "Demora para autorização de consultas, exames e cirurgias"
    ),
    "DIFICULDADE PARA AGENDAMENTO DE EXAMES-CONSULTAS": (
        "Dificuldade para agendamento de exames-consultas"
    ),
    "EXAMES LAB E IMAGENS": "Exames lab e imagens",
    "MAU ATENDIMENTO": "Mau atendimento",
    "PROBLEMAS DE INFRAESTRUTURA": "Problemas de infraestrutura",
    "QUALIDADE DO SERVIÇO": "Qualidade do serviço",
    "QUALIDADE DO SERVIÇO PRESTADO": "Qualidade do serviço prestado",
    "REDE DE ATENDIMENTO": "Rede de atendimento",
    "REEMBOLSO DE PAGAMENTO": "Reembolso de pagamento",
}
colunas_existentes = [c for c in colunas_problemas_chave if c in df_filtrado.columns]

if colunas_existentes:
    df_problemas = df_filtrado[colunas_existentes].sum().reset_index()
    df_problemas.columns = ["_chave", "Total de Casos"]
    df_problemas["Problema"] = df_problemas["_chave"].map(rotulo_problema).fillna(
        df_problemas["_chave"]
    )
    df_problemas = df_problemas.drop(columns=["_chave"])
    df_problemas = df_problemas[df_problemas["Total de Casos"] > 0].sort_values(
        "Total de Casos", ascending=True
    )
    fig_prob = px.bar(
        df_problemas,
        x="Total de Casos",
        y="Problema",
        orientation="h",
        text_auto=True,
        color="Total de Casos",
        color_continuous_scale="Reds",
    )
    fig_prob.update_layout(
        height=500,
        xaxis_title="Quantidade de casos reportados",
        yaxis_title="",
        showlegend=False,
    )
    st.plotly_chart(fig_prob, width="stretch")
else:
    st.info("Não há colunas de problemas dummy no conjunto de dados filtrado.")

st.subheader("Amostra dos dados filtrados")

colunas_tabela = [
    c
    for c in [
        "DATA",
        "ESTADO",
        "LOCAL",
        "TEMA",
        "STATUS",
        "CATEGORIA_AJUSTADA",
        "TAMANHO_TEXTO",
        "DESCRICAO",
        "URL",
    ]
    if c in df_filtrado.columns
]

df_tabela = df_filtrado[colunas_tabela].copy()

if "DATA" in df_tabela.columns:
    df_tabela = df_tabela.sort_values("DATA", ascending=False)

cfg_tabela = {}
if "URL" in df_tabela.columns:
    cfg_tabela["URL"] = st.column_config.LinkColumn(
        "Link direto", display_text="Acessar reclamação"
    )
if "DATA" in df_tabela.columns:
    cfg_tabela["DATA"] = st.column_config.DateColumn(
        "Data da ocorrência", format="DD/MM/YYYY"
    )
if "TEMA" in df_tabela.columns:
    cfg_tabela["TEMA"] = "Título da reclamação"
if "LOCAL" in df_tabela.columns:
    cfg_tabela["LOCAL"] = "Cidade"

st.dataframe(
    df_tabela.head(50),
    width="stretch",
    hide_index=True,
    column_config=cfg_tabela,
)