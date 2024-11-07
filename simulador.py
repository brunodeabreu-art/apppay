import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from PIL import Image

# Function to remove outliers
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# Create the simulator
st.set_page_config(
    page_title="Simulador de Spread",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items=None,
    page_icon="📊",
)

# Aplicar tema escuro
st.markdown("""
    <style>
        /* Reset geral para fundo preto */
        .stApp {
            background-color: #0E1117;
        }
        
        /* Tema escuro */
        :root {
            --primary-color: #ff4b4b;
            --background-color: #0E1117;
            --secondary-background-color: #262730;
            --text-color: #FAFAFA;
            --font: "Source Sans Pro", sans-serif;
        }

        /* Fundo preto para o corpo principal */
        .main {
            background-color: #0E1117 !important;
        }

        /* Fundo preto para a sidebar */
        [data-testid="stSidebar"] {
            background-color: #262730 !important;
        }
        
        /* Cores de texto */
        .stMarkdown, .stText, .stTitle {
            color: #FAFAFA !important;
        }
        
        h1, h2, h3, h4, h5, h6 {
            color: #FAFAFA !important;
        }
        
        label {
            color: #FAFAFA !important;
        }
        
        [data-testid="stMetricLabel"] {
            color: #FAFAFA !important;
        }
        
        /* Ajuste de cores para widgets */
        .stSlider, .stSelectbox, .stMultiSelect {
            background-color: #262730 !important;
            color: #FAFAFA !important;
        }

        /* Cores para cards e containers */
        div[data-testid="stMetricValue"] {
            background-color: #262730 !important;
            color: #FAFAFA !important;
        }

        .metric-card {
            background-color: #262730 !important;
            border: 1px solid #4A4A4A !important;
        }
    </style>
""", unsafe_allow_html=True)

# Configuração base para todos os gráficos Plotly
layout_config = {
    'plot_bgcolor': '#0E1117',
    'paper_bgcolor': '#0E1117',
    'font': {'color': '#FAFAFA'},
    'xaxis': {
        'gridcolor': '#262730',
        'color': '#FAFAFA'
    },
    'yaxis': {
        'gridcolor': '#262730',
        'color': '#FAFAFA'
    },
    'legend': {
        'font': {'color': '#FAFAFA'}
    }
}

# Mover o upload do arquivo para antes do sidebar
st.markdown('<div class="content-section">', unsafe_allow_html=True)
st.markdown("### Upload de Dados")
uploaded_file = st.file_uploader("Selecione o arquivo CSV com os dados para análise", type=['csv'])
st.markdown('</div>', unsafe_allow_html=True)

# Inicializar df
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        df['TicketMedio'] = df['VolumeTotal'] / df['NumeroOperacoes']
    except Exception as e:
        st.error(f"Erro ao ler o arquivo: {str(e)}")
        st.stop()
else:
    st.warning("Por favor, faça upload de um arquivo CSV para continuar.")
    st.stop()

# Agora que temos df, podemos continuar com o sidebar
with st.sidebar:
    try:
        logo = Image.open('logoamigo.png')
        st.markdown('<div style="padding: 0; margin: 0;">', unsafe_allow_html=True)
        st.image(logo, width=240)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.error("Logo não encontrada")
    
    st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
    st.markdown('<p style="font-size: 1.1rem; font-weight: 500; color: var(--text);">Configurações</p>', unsafe_allow_html=True)
    st.markdown('<p style="font-size: 0.9rem; color: var(--text-light); margin-bottom: 1rem;">Ajuste os parâmetros da simulação</p>', unsafe_allow_html=True)
    
    # Agora podemos usar df com segurança
    bandas_disponiveis = ['Todas'] + sorted(df['BandaCliente'].unique().tolist())
    banda_selecionada = st.selectbox(
        "Banda do Cliente",
        options=bandas_disponiveis,
        index=0
    )
    
    # Target Rate com slider
    target_rate = st.slider(
        "Taxa Alvo (%)",
        min_value=0.0,
        max_value=5.0,
        value=2.0,
        step=0.1,
        format="%0.1f%%"
    ) / 100
    
    # Taxa de Conversão com slider
    conversion_rate = st.slider(
        "Taxa de Conversão (%)",
        min_value=0.0,
        max_value=100.0,
        value=30.0,
        step=5.0,
        format="%0.1f%%"
    ) / 100
    
    # Controles de Outliers
    st.markdown('<div style="margin-top: 2rem;">', unsafe_allow_html=True)
    st.markdown("### Controle de Outliers")
    
    col1, col2 = st.columns(2)
    
    with col1:
        remove_outliers_volume = st.checkbox(
            "Outliers Volume",
            help="Remove valores extremos de volume usando método IQR"
        )
    
    with col2:
        remove_outliers_taxa = st.checkbox(
            "Outliers Taxa",
            help="Remove valores extremos de taxa usando método IQR"
        )
    
    st.markdown('</div>', unsafe_allow_html=True)

# Aplicar filtros aos dados
if banda_selecionada != 'Todas':
    df_filtered = df[df['BandaCliente'] == banda_selecionada].copy()
else:
    df_filtered = df.copy()

# Contador de outliers removidos
outliers_removidos = 0
tamanho_original = len(df_filtered)

# Aplicar remoção de outliers de volume se checkbox estiver marcado
if remove_outliers_volume:
    df_filtered = remove_outliers(df_filtered, 'VolumeMediaMensal')
    outliers_volume = tamanho_original - len(df_filtered)
    if outliers_volume > 0:
        st.sidebar.info(f"Removidos {outliers_volume} outliers de volume")
    tamanho_original = len(df_filtered)

# Aplicar remoção de outliers de taxa se checkbox estiver marcado
if remove_outliers_taxa:
    df_filtered = remove_outliers(df_filtered, 'TaxaMediaPonderada')
    outliers_taxa = tamanho_original - len(df_filtered)
    if outliers_taxa > 0:
        st.sidebar.info(f"Removidos {outliers_taxa} outliers de taxa")

# Header principal
st.markdown("""
    <div class="main-header">
        <h1>Simulador de Spread</h1>
        <p style='color: var(--text-light);'>Análise e Simulação de Taxas com Machine Learning</p>
    </div>
""", unsafe_allow_html=True)

# Organizar conteúdo em seções
st.markdown('<div class="content-section">', unsafe_allow_html=True)
st.markdown("### Métricas Principais")

# Usar colunas do Streamlit com classes CSS personalizadas
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    clientes_acima = df[df['TaxaMediaPonderada'] > target_rate]
    clientes_abaixo = df[df['TaxaMediaPonderada'] <= target_rate]

    above_rate = len(clientes_acima)
    below_rate = len(clientes_abaixo)
    # Ajustar volume potencial considerando taxa de conversão
    volume_mensal_acumulado = clientes_acima['VolumeMediaMensal'].sum() * conversion_rate
    projecao_13_meses = volume_mensal_acumulado * 13
    ticket_medio_global = df['TicketMedio'].mean()

    col1.metric("Clientes Acima da Taxa", f"{above_rate} ({(above_rate/len(df)*100):.1f}%)")
    col2.metric("Clientes Abaixo da Taxa", f"{below_rate} ({(below_rate/len(df)*100):.1f}%)")
    col3.metric("Volume Mensal Potencial", f"R$ {volume_mensal_acumulado:,.2f}")
    col4.metric("Projeção 13 Meses", f"R$ {projecao_13_meses:,.2f}")
    st.markdown('</div>', unsafe_allow_html=True)

# Organizar gráficos em containers
st.markdown('<div class="chart-container">', unsafe_allow_html=True)

# Distribution of rates by band
fig_dist = go.Figure()

for banda in df['BandaCliente'].unique():
    dados_banda = df[df['BandaCliente'] == banda]['TaxaMediaPonderada']
    
    if len(dados_banda) > 1:
        media = dados_banda.mean()
        desvio = dados_banda.std()
        x = np.linspace(dados_banda.min(), dados_banda.max(), 100)
        y = stats.norm.pdf(x, media, desvio)
        
        fig_dist.add_trace(go.Scatter(
            x=x,
            y=y,
            name=f'Banda {banda}',
            mode='lines',
            fill='tozeroy',
            opacity=0.6
        ))

fig_dist.add_vline(
    x=target_rate,
    line_dash="dash",
    line_color="red",
    annotation_text=f"Taxa Alvo: {target_rate:.1%}"
)

fig_dist.update_layout(
    title='Distribuição Normal das Taxas por Banda',
    xaxis_title='Taxa Média Ponderada',
    yaxis_title='Densidade',
    showlegend=True,
    height=500,
    **layout_config
)

st.plotly_chart(fig_dist, use_container_width=True)

# Box Plot of rates by band
fig_box = px.box(
    df,
    x='BandaCliente',
    y='TaxaMediaPonderada',
    title='Distribuição das Taxas por Banda (Box Plot)',
    labels={'TaxaMediaPonderada': 'Taxa Média Ponderada', 'BandaCliente': 'Banda do Cliente'}
)

# Aplicar configurações de tema escuro ao box plot
fig_box.update_layout(
    plot_bgcolor='#0E1117',
    paper_bgcolor='#0E1117',
    font={'color': '#FAFAFA'},
    xaxis={'gridcolor': '#262730', 'color': '#FAFAFA'},
    yaxis={'gridcolor': '#262730', 'color': '#FAFAFA'}
)

# Adicionar linha da taxa alvo
fig_box.add_hline(
    y=target_rate,
    line_dash="dash",
    line_color="red",
    annotation_text=f"Taxa Alvo: {target_rate:.1%}"
)

st.plotly_chart(fig_box, use_container_width=True)

# Scatter Plot: Volume vs Rate
fig_scatter = px.scatter(
    df,
    x='VolumeMediaMensal',
    y='TaxaMediaPonderada',
    color='BandaCliente',
    size='TicketMedio',
    title='Relação entre Volume, Taxa e Ticket Médio',
    labels={
        'VolumeMediaMensal': 'Volume Médio Mensal (R$)',
        'TaxaMediaPonderada': 'Taxa Média Ponderada',
        'BandaCliente': 'Banda do Cliente',
        'TicketMedio': 'Ticket Médio'
    }
)

# Aplicar configurações de tema escuro
fig_scatter.update_layout(
    plot_bgcolor='#0E1117',
    paper_bgcolor='#0E1117',
    font={'color': '#FAFAFA'},
    xaxis={'gridcolor': '#262730', 'color': '#FAFAFA'},
    yaxis={'gridcolor': '#262730', 'color': '#FAFAFA'},
    legend={'font': {'color': '#FAFAFA'}}
)

# Adicionar linha da taxa alvo
fig_scatter.add_hline(
    y=target_rate,
    line_dash="dash",
    line_color="red",
    annotation_text=f"Taxa Alvo: {target_rate:.1%}"
)

st.plotly_chart(fig_scatter, use_container_width=True)



# Volume Analysis by Category
st.markdown("### Análise de Volume")

volume_dist = df.groupby(['BandaCliente', df['TaxaMediaPonderada'] > target_rate]
                              ).size().unstack()
volume_dist.columns = ['Abaixo da Taxa', 'Acima da Taxa']

fig_vol = go.Figure()
for col in volume_dist.columns:
    fig_vol.add_trace(go.Bar(
        name=col,
        x=volume_dist.index,
        y=volume_dist[col],
        text=volume_dist[col],
        textposition='auto',
    ))

fig_vol.update_layout(
    title='Distribuição de Clientes por Volume e Taxa',
    barmode='stack',
    height=500,
    **layout_config
)

st.plotly_chart(fig_vol, use_container_width=True)

# Potential Volume by Category
volume_potential = df[df['TaxaMediaPonderada'] > target_rate].groupby('BandaCliente')['VolumeMediaMensal'].sum().reset_index()
volume_potential['VolumeMediaMensal'] = volume_potential['VolumeMediaMensal'] * conversion_rate

fig_pot = px.bar(
    volume_potential,
    x='BandaCliente',
    y='VolumeMediaMensal',
    title=f'Volume Mensal Potencial por Categoria (Taxa > {target_rate:.1%}, Conversão: {conversion_rate:.1%})',
    labels={'BandaCliente': 'Banda do Cliente', 'VolumeMediaMensal': 'Volume Potencial (R$)'},
    **layout_config
)

st.plotly_chart(fig_pot, use_container_width=True)

# Detailed Statistics
st.markdown("### Estatísticas Detalhadas")
analysis_df = df_filtered.groupby('BandaCliente').agg({
    'TaxaMediaPonderada': ['count', 'mean', 'min', 'max', 'std'],
    'VolumeMediaMensal': ['sum', 'mean', 'std'],
    'TicketMedio': ['mean', 'std'],
    'VolumeTotal': 'sum'
}).round(4)

analysis_df.columns = [
    'Quantidade', 'Taxa Média', 'Taxa Mínima', 'Taxa Máxima', 'Desvio Padrão Taxa',
    'Volume Mensal Total', 'Volume Mensal Médio', 'Desvio Padrão Volume',
    'Ticket Médio', 'Desvio Padrão Ticket', 'Volume Total'
]

st.dataframe(
    analysis_df.style.format({
        'Taxa Média': '{:.2%}',
        'Taxa Mínima': '{:.2%}',
        'Taxa Máxima': '{:.2%}',
        'Desvio Padrão Taxa': '{:.2%}',
        'Volume Mensal Total': 'R$ {:,.2f}',
        'Volume Mensal Médio': 'R$ {:,.2f}',
        'Desvio Padrão Volume': 'R$ {:,.2f}',
        'Ticket Médio': 'R$ {:,.2f}',
        'Desvio Padrão Ticket': 'R$ {:,.2f}',
        'Volume Total': 'R$ {:,.2f}'
    })
)

# Download button
st.download_button(
    label="Download da Análise Completa",
    data=analysis_df.to_csv().encode('utf-8'),
    file_name='analise_completa.csv',
    mime='text/csv'
)

# Após as métricas de conversão existentes, adicionar nova seção
st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
st.markdown("""
    <div class="metric-container">
        <h3>Análises Complementares</h3>
    </div>
""", unsafe_allow_html=True)

# Análise de distribuição de volume
st.markdown("#### Distribuição do Volume")
help_volume = """
Este gráfico mostra como o volume mensal está distribuído entre os clientes.
- Concentração à direita indica maior número de clientes com volumes altos
- Concentração à esquerda indica maior número de clientes com volumes baixos
"""
st.info(help_volume)

fig_vol_dist = px.histogram(
    df_filtered,
    x='VolumeMediaMensal',
    nbins=50,
    title='Distribuição do Volume Médio Mensal',
    marginal='box'
)
st.plotly_chart(fig_vol_dist, use_container_width=True)

# Análise de quartis
st.markdown("#### Análise de Quartis")
st.info("Os quartis dividem os dados em 4 partes iguais, ajudando a entender a distribuição dos valores.")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Quartis de Volume**")
    quartis_volume = df_filtered['VolumeMediaMensal'].quantile([0.25, 0.5, 0.75])
    for q, v in zip(['25%', '50%', '75%'], quartis_volume):
        st.metric(f"{q} dos volumes estão abaixo de", f"R$ {v:,.2f}")

with col2:
    st.markdown("**Quartis de Taxa**")
    quartis_taxa = df_filtered['TaxaMediaPonderada'].quantile([0.25, 0.5, 0.75])
    for q, v in zip(['25%', '50%', '75%'], quartis_taxa):
        st.metric(f"{q} das taxas estão abaixo de", f"{v:.2%}")

# Análise de correlação
st.markdown("#### Correlação Volume x Taxa")
st.info("""
A correlação indica a força da relação entre volume e taxa.
- Valores próximos a -1: forte relação negativa (quando um aumenta, outro diminui)
- Valores próximos a 1: forte relação positiva (ambos aumentam ou diminuem juntos)
- Valores próximos a 0: pouca ou nenhuma relação
""")

corr = df_filtered['VolumeMediaMensal'].corr(df_filtered['TaxaMediaPonderada'])
st.metric("Correlação", f"{corr:.2f}")

# Segmentação por faixas de volume
st.markdown("#### Análise por Faixas de Volume")
st.info("""
Divisão dos clientes em 5 grupos de acordo com o volume mensal,
permitindo analisar o comportamento das taxas em cada faixa.
""")

df_filtered['FaixaVolume'] = pd.qcut(
    df_filtered['VolumeMediaMensal'],
    q=5,
    labels=['Muito Baixo', 'Baixo', 'Médio', 'Alto', 'Muito Alto']
)

analise_faixa = df_filtered.groupby('FaixaVolume').agg({
    'VolumeMediaMensal': ['mean', 'count'],
    'TaxaMediaPonderada': 'mean'
}).round(4)

analise_faixa.columns = ['Volume Médio', 'Quantidade Clientes', 'Taxa Média']

st.dataframe(
    analise_faixa.style.format({
        'Volume Médio': 'R$ {:,.2f}',
        'Taxa Média': '{:.2%}'
    })
)

# Visualização da relação Volume x Taxa por faixa
fig_faixas = px.box(
    df_filtered,
    x='FaixaVolume',
    y='TaxaMediaPonderada',
    title='Distribuição das Taxas por Faixa de Volume',
    labels={
        'FaixaVolume': 'Faixa de Volume',
        'TaxaMediaPonderada': 'Taxa Média Ponderada'
    }
)
st.plotly_chart(fig_faixas, use_container_width=True)

# New Machine Learning Section
st.markdown("## Análise Preditiva")

# Data preparation for ML
features_for_clustering = ['TaxaMediaPonderada', 'VolumeMediaMensal', 'TicketMedio']
X_cluster = df_filtered[features_for_clustering].copy()

# Normalization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

# K-means Clustering
st.markdown("### Segmentação de Clientes (K-means)")
n_clusters = st.slider("Número de Segmentos", min_value=2, max_value=8, value=4)

kmeans = KMeans(n_clusters=n_clusters, random_state=42)
df_filtered['Cluster'] = kmeans.fit_predict(X_scaled)

# 3D Cluster Visualization
fig_3d = px.scatter_3d(
    df_filtered,
    x='VolumeMediaMensal',
    y='TaxaMediaPonderada',
    z='TicketMedio',
    color='Cluster',
    title='Segmentaão 3D de Clientes',
    labels={
        'VolumeMediaMensal': 'Volume Médio Mensal',
        'TaxaMediaPonderada': 'Taxa Média Ponderada',
        'TicketMedio': 'Ticket Médio'
    }
)
st.plotly_chart(fig_3d, use_container_width=True)

# Continuação do código anterior...

# Cluster Analysis
cluster_analysis = df_filtered.groupby('Cluster').agg({
    'TaxaMediaPonderada': ['mean', 'count'],
    'VolumeMediaMensal': 'mean',
    'TicketMedio': 'mean'
}).round(4)

cluster_analysis.columns = ['Taxa Média', 'Quantidade Clientes', 'Volume Médio', 'Ticket Médio']
st.markdown("#### Características dos Segmentos")
st.dataframe(
    cluster_analysis.style.format({
        'Taxa Mdia': '{:.2%}',
        'Volume Médio': 'R$ {:,.2f}',
        'Ticket Médio': 'R$ {:,.2f}'
    })
)





# Nova seção de análises avançadas (inserir antes do footer)
st.markdown("""
    <div style='background-color: #f8f9fa; padding: 1.5rem; border-radius: 8px; margin: 2rem 0;'>
        <h2 style='color: #2E4057; margin-bottom: 1rem;'>Análise Aprofundada de Volume e Taxa</h2>
    </div>
""", unsafe_allow_html=True)

# 1. Estatísticas Avançadas
st.markdown("### Métricas Estatísticas Avançadas")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Volume Médio Mensal")
    vol_stats = {
        'Média': df_filtered['VolumeMediaMensal'].mean(),
        'Mediana': df_filtered['VolumeMediaMensal'].median(),
        'Desvio Padrão': df_filtered['VolumeMediaMensal'].std(),
        'Coef. de Variação': df_filtered['VolumeMediaMensal'].std() / df_filtered['VolumeMediaMensal'].mean(),
        'Assimetria': df_filtered['VolumeMediaMensal'].skew(),
        'Curtose': df_filtered['VolumeMediaMensal'].kurtosis()
    }
    
    for metric, value in vol_stats.items():
        if metric == 'Coef. de Variação':
            st.metric(metric, f"{value:.2%}")
        elif metric in ['Assimetria', 'Curtose']:
            st.metric(metric, f"{value:.2f}")
        else:
            st.metric(metric, f"R$ {value:,.2f}")

with col2:
    st.markdown("#### Taxa Média Ponderada")
    taxa_stats = {
        'Média': df_filtered['TaxaMediaPonderada'].mean(),
        'Mediana': df_filtered['TaxaMediaPonderada'].median(),
        'Desvio Padrão': df_filtered['TaxaMediaPonderada'].std(),
        'Coef. de Variação': df_filtered['TaxaMediaPonderada'].std() / df_filtered['TaxaMediaPonderada'].mean(),
        'Assimetria': df_filtered['TaxaMediaPonderada'].skew(),
        'Curtose': df_filtered['TaxaMediaPonderada'].kurtosis()
    }
    
    for metric, value in taxa_stats.items():
        if metric in ['Média', 'Mediana', 'Desvio Padrão', 'Coef. de Variação']:
            st.metric(metric, f"{value:.2%}")
        else:
            st.metric(metric, f"{value:.2f}")

# 2. Análise de Concentração
st.markdown("### Análise de Concentração")

# Cálculo do Índice de Gini para Volume
def gini(array):
    array = np.array(array)
    if np.amin(array) < 0:
        array -= np.amin(array)
    array += 0.0000001
    array = np.sort(array)
    index = np.arange(1, array.shape[0] + 1)
    n = array.shape[0]
    return ((np.sum((2 * index - n - 1) * array)) / (n * np.sum(array)))

gini_volume = gini(df_filtered['VolumeMediaMensal'])
top_10_volume = df_filtered['VolumeMediaMensal'].nlargest(int(len(df_filtered)*0.1)).sum() / df_filtered['VolumeMediaMensal'].sum()
top_20_volume = df_filtered['VolumeMediaMensal'].nlargest(int(len(df_filtered)*0.2)).sum() / df_filtered['VolumeMediaMensal'].sum()

col1, col2, col3 = st.columns(3)
col1.metric("Índice de Gini (Volume)", f"{gini_volume:.2f}")
col2.metric("Concentração Top 10%", f"{top_10_volume:.1%}")
col3.metric("Concentração Top 20%", f"{top_20_volume:.1%}")

# 3. Análise de Elasticidade e Sensibilidade
st.markdown("### Análise de Elasticidade Volume-Taxa")

# Análise por quartis de volume
df_filtered['VolumeQuartil'] = pd.qcut(df_filtered['VolumeMediaMensal'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
elasticidade = df_filtered.groupby('VolumeQuartil').agg({
    'VolumeMediaMensal': 'mean',
    'TaxaMediaPonderada': 'mean'
})

# Visualização da elasticidade
fig_elast = px.line(
    elasticidade.reset_index(),
    x='VolumeQuartil',
    y=['VolumeMediaMensal', 'TaxaMediaPonderada'],
    title='Relação Volume-Taxa por Quartil',
    labels={'value': 'Valor', 'VolumeQuartil': 'Quartil de Volume'}
)
st.plotly_chart(fig_elast, use_container_width=True)

# 4. Insights Estratégicos
st.markdown("### Insights Estratégicos")

taxa_media_global = df_filtered['TaxaMediaPonderada'].mean()
vol_medio_global = df_filtered['VolumeMediaMensal'].mean()
spread_taxa = df_filtered['TaxaMediaPonderada'].max() - df_filtered['TaxaMediaPonderada'].min()

st.markdown("""
    <div style='background-color: #f8f9fa; padding: 1.5rem; border-radius: 8px; margin: 1rem 0;'>
        <h4 style='color: #2E4057; margin-bottom: 1rem;'>Análise de Risco-Retorno</h4>
        <ul style='color: #666; margin-left: 1rem;'>
""" + f"""
            <li>O spread total de taxas é de {spread_taxa:.2%}, indicando a amplitude de negociação</li>
            <li>A assimetria de {taxa_stats['Assimetria']:.2f} na distribuição das taxas indica {'uma concentração em taxas mais baixas' if taxa_stats['Assimetria'] < 0 else 'uma tendência para taxas mais altas'}</li>
            <li>O coeficiente de variação de {taxa_stats['Coef. de Variação']:.2%} nas taxas sugere {'alta' if taxa_stats['Coef. de Variação'] > 0.5 else 'moderada'} dispersão relativa</li>
            <li>A concentração de volume nos top 10% dos clientes ({top_10_volume:.1%}) indica {'alta' if top_10_volume > 0.5 else 'moderada'} concentração de risco</li>
        </ul>
    </div>
""", unsafe_allow_html=True)

# 5. Mapa de Calor de Densidade
fig_density = px.density_heatmap(
    df_filtered,
    x='VolumeMediaMensal',
    y='TaxaMediaPonderada',
    title='Mapa de Densidade Volume-Taxa',
    labels={
        'VolumeMediaMensal': 'Volume Médio Mensal (R$)',
        'TaxaMediaPonderada': 'Taxa Média Ponderada'
    }
)
st.plotly_chart(fig_density, use_container_width=True)

# Continua com o footer original
# ... resto do código original ...

st.markdown("## 📊 Análises Estatísticas Avançadas")

# 1. Testes de Normalidade
st.markdown("### 1. Testes de Normalidade")
col1, col2 = st.columns(2)

with col1:
    st.markdown("**Volume Médio**")
    shapiro_vol = stats.shapiro(df_filtered['VolumeMediaMensal'])
    st.metric("Shapiro-Wilk p-value", f"{shapiro_vol.pvalue:.4f}")
    st.markdown(f"{'✅ Normal' if shapiro_vol.pvalue > 0.05 else '❌ Não Normal'}")

with col2:
    st.markdown("**Taxa Média Ponderada**")
    shapiro_taxa = stats.shapiro(df_filtered['TaxaMediaPonderada'])
    st.metric("Shapiro-Wilk p-value", f"{shapiro_taxa.pvalue:.4f}")
    st.markdown(f"{'✅ Normal' if shapiro_taxa.pvalue > 0.05 else '❌ Não Normal'}")

# 2. Análise de Outliers Multivariada
st.markdown("### 2. Análise de Outliers Multivariada")

# Mahalanobis Distance
def mahalanobis_distance(data):
    covariance_matrix = np.cov(data, rowvar=False)
    inv_covariance_matrix = np.linalg.inv(covariance_matrix)
    mean = np.mean(data, axis=0)
    diff = data - mean
    dist = np.sqrt(np.sum(np.dot(diff, inv_covariance_matrix) * diff, axis=1))
    return dist

X = df_filtered[['VolumeMediaMensal', 'TaxaMediaPonderada']].values
distances = mahalanobis_distance(X)
threshold = np.percentile(distances, 97.5)
outliers = distances > threshold

st.markdown(f"**Outliers Identificados:** {sum(outliers)} ({(sum(outliers)/len(df_filtered)*100):.1f}%)")

# Visualização dos outliers
fig_outliers = px.scatter(
    df_filtered,
    x='VolumeMediaMensal',
    y='TaxaMediaPonderada',
    color=outliers,
    title='Outliers Multivariados (Distância de Mahalanobis)',
    labels={'color': 'É Outlier'}
)
st.plotly_chart(fig_outliers, use_container_width=True)

# 3. Análise de Copula
st.markdown("### 3. Análise de Dependência (Copula)")

# Transformação para ranks uniformes
u1 = stats.rankdata(df_filtered['VolumeMediaMensal']) / (len(df_filtered) + 1)
u2 = stats.rankdata(df_filtered['TaxaMediaPonderada']) / (len(df_filtered) + 1)

fig_copula = px.scatter(
    x=u1, 
    y=u2,
    title='Copula Empírica Volume-Taxa',
    labels={'x': 'Volume (rank)', 'y': 'Taxa (rank)'}
)
st.plotly_chart(fig_copula, use_container_width=True)

# 4. Machine Learning Avançado
st.markdown("## 🤖 Modelos de Machine Learning")

# Preparação dos dados
X = df_filtered[['VolumeMediaMensal', 'TaxaMediaPonderada']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 1. Clustering Hierárquico
st.markdown("### 1. Clustering Hierárquico")
n_clusters = st.slider("Número de Clusters", 2, 8, 4)

from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering

# Criar linkage matrix
linkage_matrix = linkage(X_scaled, method='ward')


# Aplicar clustering
hc = AgglomerativeClustering(n_clusters=n_clusters)
df_filtered['Cluster_Hierarquico'] = hc.fit_predict(X_scaled)

# 2. Análise de Componentes Principais (PCA)
st.markdown("### 2. Análise de Componentes Principais")

from sklearn.decomposition import PCA
pca = PCA()
pca_result = pca.fit_transform(X_scaled)

explained_variance = pca.explained_variance_ratio_
st.markdown(f"**Variância Explicada:**")
for i, var in enumerate(explained_variance):
    st.markdown(f"PC{i+1}: {var:.2%}")

# 3. Isolation Forest para Detecção de Anomalias
st.markdown("### 3. Detecção de Anomalias (Isolation Forest)")

from sklearn.ensemble import IsolationForest
iso_forest = IsolationForest(contamination=0.1, random_state=42)
anomalies = iso_forest.fit_predict(X_scaled)
df_filtered['Is_Anomaly'] = anomalies == -1

# Visualizaão das anomalias
fig_anomalies = px.scatter(
    df_filtered,
    x='VolumeMediaMensal',
    y='TaxaMediaPonderada',
    color='Is_Anomaly',
    title='Detecção de Anomalias (Isolation Forest)',
    labels={'Is_Anomaly': 'É Anomalia'}
)
st.plotly_chart(fig_anomalies, use_container_width=True)

# 4. Análise de Tendências
st.markdown("### 4. Análise de Tendências e Padrões")

# Regressão Polinomial
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

X_vol = df_filtered['VolumeMediaMensal'].values.reshape(-1, 1)
y_taxa = df_filtered['TaxaMediaPonderada'].values

degree = st.slider("Grau do Polinômio", 1, 5, 2)
polyreg = make_pipeline(PolynomialFeatures(degree), LinearRegression())
polyreg.fit(X_vol, y_taxa)

# Plot da regressão polinomial
X_plot = np.linspace(X_vol.min(), X_vol.max(), 100).reshape(-1, 1)
y_plot = polyreg.predict(X_plot)

fig_trend = go.Figure()
fig_trend.add_trace(go.Scatter(x=df_filtered['VolumeMediaMensal'], y=df_filtered['TaxaMediaPonderada'], 
                              mode='markers', name='Dados Reais'))
fig_trend.add_trace(go.Scatter(x=X_plot.ravel(), y=y_plot, name='Tendência Polinomial'))
fig_trend.update_layout(title='Análise de Tendência Volume-Taxa')
st.plotly_chart(fig_trend, use_container_width=True)

# 5. Métricas de Performance
st.markdown("### 5. Métricas de Performance do Modelo")

r2 = r2_score(y_taxa, polyreg.predict(X_vol))
mse = mean_squared_error(y_taxa, polyreg.predict(X_vol))

col1, col2 = st.columns(2)
with col1:
    st.metric("R² Score", f"{r2:.4f}")
with col2:
    st.metric("MSE", f"{mse:.4f}")

# ... resto do código continua igual ...

# Sugestão: Usar SessionState para persistir dados entre interações
if 'df' not in st.session_state:
    st.session_state.df = None

# Adicionar métricas de conversão
st.markdown("### 📊 Análise de Conversão")

# Calcular métricas de conversão
df_filtered['VolumeConvertido'] = df_filtered['VolumeMediaMensal'] * conversion_rate
total_volume_original = df_filtered['VolumeMediaMensal'].sum()
total_volume_convertido = df_filtered['VolumeConvertido'].sum()

# Métricas principais de conversão
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Volume Total Original",
        f"R$ {total_volume_original:,.2f}",
        help="Volume mensal total antes da taxa de conversão"
    )

with col2:
    st.metric(
        "Volume Total Convertido",
        f"R$ {total_volume_convertido:,.2f}",
        delta=f"{conversion_rate:.1%}",
        help="Volume mensal total após aplicar a taxa de conversão"
    )

with col3:
    projecao_anual = total_volume_convertido * 12
    st.metric(
        "Projeção Anual",
        f"R$ {projecao_anual:,.2f}",
        help="Projeção do volume convertido para 12 meses"
    )

# Análise por banda
st.markdown("#### Análise de Conversão por Banda")

conversao_banda = df_filtered.groupby('BandaCliente').agg({
    'VolumeMediaMensal': 'sum',
    'VolumeConvertido': 'sum',
    'BandaCliente': 'count'
}).rename(columns={'BandaCliente': 'Quantidade'})

conversao_banda['PercentualConversao'] = conversion_rate
conversao_banda['DiferencaVolume'] = conversao_banda['VolumeConvertido'] - conversao_banda['VolumeMediaMensal']

# Visualização da conversão por banda
fig_conversao = go.Figure()

fig_conversao.add_trace(go.Bar(
    name='Volume Original',
    x=conversao_banda.index,
    y=conversao_banda['VolumeMediaMensal'],
    text=conversao_banda['VolumeMediaMensal'].apply(lambda x: f'R$ {x:,.2f}'),
    textposition='auto',
))

fig_conversao.add_trace(go.Bar(
    name='Volume Convertido',
    x=conversao_banda.index,
    y=conversao_banda['VolumeConvertido'],
    text=conversao_banda['VolumeConvertido'].apply(lambda x: f'R$ {x:,.2f}'),
    textposition='auto',
))

fig_conversao.update_layout(
    title=f'Comparação de Volume por Banda (Taxa de Conversão: {conversion_rate:.1%})',
    barmode='group',
    height=500
)

st.plotly_chart(fig_conversao, use_container_width=True)

# Tabela detalhada de conversão
st.markdown("#### Detalhamento da Conversão")

conversao_detalhada = pd.DataFrame({
    'Banda': conversao_banda.index,
    'Volume Original': conversao_banda['VolumeMediaMensal'],
    'Volume Convertido': conversao_banda['VolumeConvertido'],
    'Diferença': conversao_banda['DiferencaVolume'],
    'Quantidade Clientes': conversao_banda['Quantidade'],
    'Taxa de Conversão': conversao_banda['PercentualConversao']
})

st.dataframe(
    conversao_detalhada.style.format({
        'Volume Original': 'R$ {:,.2f}',
        'Volume Convertido': 'R$ {:,.2f}',
        'Diferença': 'R$ {:,.2f}',
        'Taxa de Conversão': '{:.1%}'
    })
)

# Análise de sensibilidade
st.markdown("#### 📈 Análise de Sensibilidade da Conversão")

taxas_simulacao = np.linspace(0.1, 1.0, 10)
volumes_simulados = [total_volume_original * taxa for taxa in taxas_simulacao]

fig_sensibilidade = go.Figure()

fig_sensibilidade.add_trace(go.Scatter(
    x=taxas_simulacao * 100,
    y=volumes_simulados,
    mode='lines+markers',
    name='Volume Projetado',
    text=[f'R$ {v:,.2f}' for v in volumes_simulados],
    hovertemplate='Taxa: %{x:.1f}%<br>Volume: %{text}'
))

fig_sensibilidade.add_vline(
    x=conversion_rate * 100,
    line_dash="dash",
    line_color="red",
    annotation_text=f"Taxa Atual: {conversion_rate:.1%}"
)

fig_sensibilidade.update_layout(
    title='Simulação de Volumes por Taxa de Conversão',
    xaxis_title='Taxa de Conversão (%)',
    yaxis_title='Volume Projetado (R$)',
    height=500
)

st.plotly_chart(fig_sensibilidade, use_container_width=True)

# Métricas de impacto
st.markdown("#### 💡 Insights de Conversão")

col1, col2 = st.columns(2)

with col1:
    st.info("""
    **Impacto da Conversão**
    - Volume médio por cliente antes: R$ {:.2f}
    - Volume médio por cliente após: R$ {:.2f}
    - Diferença média por cliente: R$ {:.2f}
    """.format(
        df_filtered['VolumeMediaMensal'].mean(),
        df_filtered['VolumeConvertido'].mean(),
        df_filtered['VolumeConvertido'].mean() - df_filtered['VolumeMediaMensal'].mean()
    ))

with col2:
    st.info("""
    **Distribuição do Impacto**
    - Maior impacto absoluto: R$ {:.2f}
    - Menor impacto absoluto: R$ {:.2f}
    - Desvio padrão do impacto: R$ {:.2f}
    """.format(
        (df_filtered['VolumeConvertido'] - df_filtered['VolumeMediaMensal']).max(),
        (df_filtered['VolumeConvertido'] - df_filtered['VolumeMediaMensal']).min(),
        (df_filtered['VolumeConvertido'] - df_filtered['VolumeMediaMensal']).std()
    ))

# Adicionar botão para download dos dados de conversão
csv_conversao = conversao_detalhada.to_csv(index=False).encode('utf-8')
st.download_button(
    label="📥 Download da Análise de Conversão",
    data=csv_conversao,
    file_name='analise_conversao.csv',
    mime='text/csv',
    key='download_conversao'
)

# 1. VISÃO GERAL E CONTEXTO
st.markdown("""
    <div style='background-color: #f8f9fa; padding: 1.5rem; border-radius: 8px; margin: 2rem 0;'>
        <h1 style='color: #2E4057; margin-bottom: 1rem;'>Análise de Performance e Conversão</h1>
        <p style='color: #666; font-size: 1.1rem;'>
            Dashboard analítico para avaliação de volumes, taxas e potencial de conversão da carteira.
        </p>
    </div>
""", unsafe_allow_html=True)

# 2. MÉTRICAS PRINCIPAIS (KPIs)
st.markdown("### 📊 Indicadores Chave")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Volume Total",
        f"R$ {df_filtered['VolumeMediaMensal'].sum():,.2f}",
        help="Volume mensal total da carteira"
    )

with col2:
    st.metric(
        "Taxa Média",
        f"{df_filtered['TaxaMediaPonderada'].mean():.2%}",
        help="Taxa média ponderada da carteira"
    )

with col3:
    st.metric(
        "Clientes",
        f"{len(df_filtered):,}",
        help="Número total de clientes"
    )

with col4:
    st.metric(
        "Taxa Alvo",
        f"{target_rate:.2%}",
        help="Taxa objetivo definida"
    )

# 3. ANÁLISE DE CONVERSÃO E POTENCIAL
st.markdown("### 💹 Potencial de Conversão")
# (Inserir aqui o código de análise de conversão que já temos)

# 4. DISTRIBUIÇÃO E SEGMENTAÇÃO
st.markdown("### 🎯 Segmentação da Base")
# (Manter aqui o código do scatter plot e análise por categoria)

# 5. ANÁLISE DETALHADA POR BANDA
st.markdown("### 📈 Performance por Banda")
# (Manter análises de volume por banda e estatísticas detalhadas)

# 6. INSIGHTS E OPORTUNIDADES
st.markdown("### 💡 Insights Principais")
col1, col2 = st.columns(2)

with col1:
    st.info("""
    **Oportunidades de Volume**
    - Volume potencial de conversão: R$ {:.2f}
    - Projeção anual com conversão: R$ {:.2f}
    - Impacto médio por cliente: R$ {:.2f}
    """.format(
        total_volume_convertido - total_volume_original,
        projecao_anual,
        (df_filtered['VolumeConvertido'].mean() - df_filtered['VolumeMediaMensal'].mean()
    )))
with col2:
    st.info("""
    **Análise de Concentração**
    - Top 10% dos clientes: {:.1%} do volume
    - Índice de Gini: {:.2f}
    - Coeficiente de variação: {:.2%}
    """.format(
        top_10_volume,
        gini_volume,
        df_filtered['VolumeMediaMensal'].std() / df_filtered['VolumeMediaMensal'].mean()
    ))

# 7. ANÁLISES AVANÇADAS
st.markdown("### 🔍 Análises Avançadas")
tabs = st.tabs(["Machine Learning", "Estatísticas", "Correlações"])

with tabs[0]:
    # (Manter código de clustering e análises ML)
    st.markdown("#### Segmentação Avançada de Clientes")
    
    # Adicionar visualização do clustering
    col1, col2 = st.columns(2)
    
    with col1:
        # Métricas dos clusters
        st.markdown("**Características dos Clusters**")
        cluster_stats = df_filtered.groupby('Cluster').agg({
            'VolumeMediaMensal': ['mean', 'count'],
            'TaxaMediaPonderada': 'mean',
            'TicketMedio': 'mean'
        }).round(2)
        
        cluster_stats.columns = ['Volume Médio', 'Quantidade', 'Taxa Média', 'Ticket Médio']
        st.dataframe(
            cluster_stats.style.format({
                'Volume Médio': 'R$ {:,.2f}',
                'Taxa Média': '{:.2%}',
                'Ticket Médio': 'R$ {:,.2f}'
            })
        )
    
    with col2:
        # Distribuição dos clusters
        st.markdown("**Distribuição dos Segmentos**")
        fig_dist = px.pie(
            df_filtered, 
            names='Cluster',
            title='Distribuição dos Clientes por Segmento'
        )
        st.plotly_chart(fig_dist, use_container_width=True)
    
    # Visualização 2D dos clusters
    st.markdown("**Visualização dos Segmentos**")
    fig_clusters = px.scatter(
        df_filtered,
        x='VolumeMediaMensal',
        y='TaxaMediaPonderada',
        color='Cluster',
        size='TicketMedio',
        title='Segmentação de Clientes',
        labels={
            'VolumeMediaMensal': 'Volume Médio Mensal',
            'TaxaMediaPonderada': 'Taxa Média Ponderada',
            'TicketMedio': 'Ticket Médio'
        }
    )
    st.plotly_chart(fig_clusters, use_container_width=True)
    
    # Análise detalhada dos clusters
    st.markdown("**Análise Detalhada dos Segmentos**")
    selected_cluster = st.selectbox(
        "Selecione um segmento para análise detalhada",
        sorted(df_filtered['Cluster'].unique())
    )
    
    cluster_detail = df_filtered[df_filtered['Cluster'] == selected_cluster]
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Volume Médio",
            f"R$ {cluster_detail['VolumeMediaMensal'].mean():,.2f}"
        )
    with col2:
        st.metric(
            "Taxa Média",
            f"{cluster_detail['TaxaMediaPonderada'].mean():.2%}"
        )
    with col3:
        st.metric(
            "Ticket Médio",
            f"R$ {cluster_detail['TicketMedio'].mean():,.2f}"
        )

with tabs[1]:
    # (Manter análises estatísticas)
    st.markdown("### 📊 Análise Estatística Detalhada")
    
    # Estatísticas descritivas por banda
    stats_df = df_filtered.groupby('BandaCliente').agg({
        'VolumeMediaMensal': ['count', 'mean', 'std', 'min', 'max'],
        'TaxaMediaPonderada': ['mean', 'std', 'min', 'max'],
        'TicketMedio': ['mean', 'std']
    }).round(4)
    
    # Renomear as colunas para melhor visualização
    stats_df.columns = [
        'Quantidade', 'Volume Médio', 'Desvio Volume', 'Volume Min', 'Volume Max',
        'Taxa Média', 'Desvio Taxa', 'Taxa Min', 'Taxa Max',
        'Ticket Médio', 'Desvio Ticket'
    ]
    
    # Formatar os valores para exibição
    st.dataframe(
        stats_df.style.format({
            'Volume Médio': 'R$ {:,.2f}',
            'Desvio Volume': 'R$ {:,.2f}',
            'Volume Min': 'R$ {:,.2f}',
            'Volume Max': 'R$ {:,.2f}',
            'Taxa Média': '{:.2%}',
            'Desvio Taxa': '{:.2%}',
            'Taxa Min': '{:.2%}',
            'Taxa Max': '{:.2%}',
            'Ticket Médio': 'R$ {:,.2f}',
            'Desvio Ticket': 'R$ {:,.2f}'
        })
    )
    
    # Testes de normalidade
    st.markdown("#### Testes de Normalidade")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Volume Médio Mensal**")
        shapiro_vol = stats.shapiro(df_filtered['VolumeMediaMensal'])
        st.metric(
            "Shapiro-Wilk p-value", 
            f"{shapiro_vol.pvalue:.4f}",
            help="Valores > 0.05 indicam distribuição normal"
        )
        
    with col2:
        st.markdown("**Taxa Média Ponderada**")
        shapiro_taxa = stats.shapiro(df_filtered['TaxaMediaPonderada'])
        st.metric(
            "Shapiro-Wilk p-value", 
            f"{shapiro_taxa.pvalue:.4f}",
            help="Valores > 0.05 indicam distribuição normal"
        )

with tabs[2]:
    st.markdown("### 🔄 Análise de Correlações")
    
    # Calcular matriz de correlação
    corr_vars = ['VolumeMediaMensal', 'TaxaMediaPonderada', 'TicketMedio', 'NumeroOperacoes']
    corr_matrix = df_filtered[corr_vars].corr()
    
    # Criar heatmap de correlação
    fig_corr = go.Figure(data=go.Heatmap(
        z=corr_matrix,
        x=corr_vars,
        y=corr_vars,
        text=corr_matrix.round(3),
        texttemplate='%{text}',
        textfont={"size": 10},
        hoverongaps=False,
        colorscale='RdBu',
        zmin=-1,
        zmax=1
    ))
    
    fig_corr.update_layout(
        title='Matriz de Correlação',
        height=500,
        width=700
    )
    
    st.plotly_chart(fig_corr, use_container_width=True)
    
    # Análise detalhada das correlações
    st.markdown("#### Correlações Significativas")
    
    # Função para interpretar a correlação
    def interpretar_correlacao(corr):
        if abs(corr) < 0.3:
            return "Fraca"
        elif abs(corr) < 0.7:
            return "Moderada"
        else:
            return "Forte"
    
    # Criar DataFrame com interpretações
    correlacoes = []
    for i in range(len(corr_vars)):
        for j in range(i+1, len(corr_vars)):
            corr = corr_matrix.iloc[i,j]
            correlacoes.append({
                'Variável 1': corr_vars[i],
                'Variável 2': corr_vars[j],
                'Correlação': corr,
                'Intensidade': interpretar_correlacao(corr)
            })
    
    corr_df = pd.DataFrame(correlacoes)
    
    # Exibir tabela de correlações
    st.dataframe(
        corr_df.style.format({
            'Correlação': '{:.3f}'
        }).background_gradient(
            subset=['Correlação'],
            cmap='RdBu',
            vmin=-1,
            vmax=1
        )
    )
    
    # Scatter plots para correlações mais relevantes
    st.markdown("#### Visualização das Principais Correlações")
    
    # Encontrar a correlação mais forte
    strongest_corr = corr_df.iloc[corr_df['Correlação'].abs().idxmax()]
    
    fig_scatter = px.scatter(
        df_filtered,
        x=strongest_corr['Variável 1'],
        y=strongest_corr['Variável 2'],
        color='BandaCliente',
        title=f'Correlação entre {strongest_corr["Variável 1"]} e {strongest_corr["Variável 2"]}',
        trendline="ols"
    )
    
    st.plotly_chart(fig_scatter, use_container_width=True)

# 8. SIMULAÇÕES E CENÁRIOS
st.markdown("### 🎮 Simulador de Cenários")
# (Manter análises de sensibilidade e simulações)

# 9. EXPORTAÇÃO E DOCUMENTAÇÃO
st.markdown("### 📑 Documentação e Exportação")
col1, col2 = st.columns(2)

with col1:
    # Botão de download da análise
    st.download_button(
        label="Download da Análise Completa",
        data=analysis_df.to_csv().encode('utf-8'),
        file_name='analise_completa.csv',
        mime='text/csv'
    )

with col2:
    # Documentação e metodologia
    st.info("""
    **Metodologia**
    - Taxa de conversão aplicada: {:.1%}
    - Taxa alvo definida: {:.1%}
    - Período de análise: Mensal
    """.format(conversion_rate, target_rate))

# 10. NOTAS E CONSIDERAÇÕES
st.markdown("""
    <div style='background-color: #f8f9fa; padding: 1rem; border-radius: 8px; margin-top: 2rem;'>
        <h4 style='color: #2E4057;'>Notas Importantes</h4>
        <ul style='color: #666;'>
            <li>Valores baseados em médias mensais</li>
            <li>Projeções consideram cenário linear de conversão</li>
            <li>Análises estatísticas excluem outliers quando aplicável</li>
        </ul>
    </div>
""", unsafe_allow_html=True)
# Calcular métricas de referência
volume_medio_atual = df_filtered['VolumeMediaMensal'].mean()
taxa_media_atual = df_filtered['TaxaMediaPonderada'].mean()

# Seção Otimizada de Análise de Abertura de Mercado
st.markdown("""
    <div style='background-color: #f8f9fa; padding: 1.5rem; border-radius: 8px; margin: 2rem 0;'>
        <h2 style='color: #2E4057; margin-bottom: 1rem;'>🚀 Simulador de Expansão de Mercado</h2>
        <p style='color: #666; font-size: 1.1rem;'>
            Ferramenta avançada para simulação de cenários de expansão, projeções e análise de metas.
        </p>
    </div>
""", unsafe_allow_html=True)

# Tabs para diferentes análises
tab_cenarios, tab_projecao, tab_metas = st.tabs(["Cenários de Expansão", "Projeção 24 Meses", "Análise de Metas"])

with tab_cenarios:
    st.markdown("### 📊 Simulador de Cenários")
    
    # Controles interativos para cenários
    col1, col2, col3 = st.columns(3)
    
    with col1:
        volume_min = st.number_input(
            "Volume Mínimo (Milhões R$)",
            min_value=1.0,
            max_value=50.0,
            value=5.0,
            step=1.0,
            help="Volume inicial para simulação"
        )
    
    with col2:
        volume_max = st.number_input(
            "Volume Máximo (Milhões R$)",
            min_value=volume_min,
            max_value=100.0,
            value=50.0,
            step=5.0,
            help="Volume máximo para simulação"
        )
    
    with col3:
        incremento = st.number_input(
            "Incremento (Milhões R$)",
            min_value=1.0,
            max_value=10.0,
            value=5.0,
            step=1.0,
            help="Incremento entre faixas"
        )

    # Criar sequência de volumes personalizada
    volumes_base = np.arange(volume_min * 1e6, volume_max * 1e6 + 1, incremento * 1e6)
    
    # DataFrame de simulação atualizado
    simulacao_df = pd.DataFrame({
        'Volume_Base': volumes_base,
        'Volume_Mensal': volumes_base * conversion_rate,
        'Volume_Anual': volumes_base * conversion_rate * 12,
        'Clientes_Estimados': (volumes_base / volume_medio_atual).round(0),
        'Receita_Mensal': volumes_base * conversion_rate * taxa_media_atual,
        'Receita_Anual': volumes_base * conversion_rate * taxa_media_atual * 12
    })

    # Visualização interativa com seletor
    metrica_selecionada = st.selectbox(
        "Selecione a Métrica para Visualização",
        ["Volume Base vs Convertido", "Receita Mensal vs Anual", "Volume vs Clientes"],
        key="metrica_cenarios"
    )

    if metrica_selecionada == "Volume Base vs Convertido":
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name='Volume Base',
            x=[f'R${v/1e6:.0f}M' for v in volumes_base],
            y=simulacao_df['Volume_Base'],
            text=simulacao_df['Volume_Base'].apply(lambda x: f'R$ {x:,.0f}'),
            textposition='auto',
        ))
        fig.add_trace(go.Bar(
            name='Volume Convertido',
            x=[f'R${v/1e6:.0f}M' for v in volumes_base],
            y=simulacao_df['Volume_Mensal'],
            text=simulacao_df['Volume_Mensal'].apply(lambda x: f'R$ {x:,.0f}'),
            textposition='auto',
        ))
        fig.update_layout(
            title='Comparativo de Volumes por Faixa',
            barmode='group'
        )
    
    elif metrica_selecionada == "Receita Mensal vs Anual":
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            name='Receita Mensal',
            x=[f'R${v/1e6:.0f}M' for v in volumes_base],
            y=simulacao_df['Receita_Mensal'],
            mode='lines+markers',
            text=simulacao_df['Receita_Mensal'].apply(lambda x: f'R$ {x:,.2f}'),
        ))
        fig.add_trace(go.Scatter(
            name='Receita Anual',
            x=[f'R${v/1e6:.0f}M' for v in volumes_base],
            y=simulacao_df['Receita_Anual'],
            mode='lines+markers',
            text=simulacao_df['Receita_Anual'].apply(lambda x: f'R$ {x:,.2f}'),
        ))
        fig.update_layout(title='Projeção de Receitas')
    
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            name='Volume Mensal',
            x=simulacao_df['Clientes_Estimados'],
            y=simulacao_df['Volume_Mensal'],
            mode='markers',
            text=[f'R${v/1e6:.1f}M' for v in volumes_base],
            marker=dict(
                size=10,
                color=simulacao_df['Volume_Mensal'],
                colorscale='Viridis',
                showscale=True
            )
        ))
        fig.update_layout(
            title='Relação Volume vs Clientes',
            xaxis_title='Número de Clientes',
            yaxis_title='Volume Mensal (R$)'
        )

    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

    # Métricas de cenário
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            "Volume Médio",
            f"R$ {simulacao_df['Volume_Mensal'].mean():,.2f}",
            help="Média do volume mensal convertido"
        )
    with col2:
        st.metric(
            "Receita Média",
            f"R$ {simulacao_df['Receita_Mensal'].mean():,.2f}",
            help="Média da receita mensal"
        )
    with col3:
        st.metric(
            "Clientes Médio",
            f"{simulacao_df['Clientes_Estimados'].mean():,.0f}",
            help="Média de clientes estimados"
        )
    with col4:
        st.metric(
            "Ticket Médio",
            f"R$ {(simulacao_df['Volume_Mensal'] / simulacao_df['Clientes_Estimados']).mean():,.2f}",
            help="Ticket médio por cliente"
        )

with tab_projecao:
    st.markdown("### 📈 Projeção Temporal")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        meta_inicial = st.number_input(
            "Meta Inicial (R$)",
            min_value=1_000_000.0,
            max_value=100_000_000.0,
            value=10_000_000.0,
            step=1_000_000.0,
            format="%.2f",
            help="Volume inicial para projeção"
        )
    
    with col2:
        incremento_mensal = st.number_input(
            "Crescimento Mensal (%)",
            min_value=0.0,
            max_value=100.0,
            value=5.0,
            step=0.5,
            format="%.1f",
            help="Percentual de crescimento mensal"
        )
    
    with col3:
        meses_projecao = st.slider(
            "Período de Projeção (Meses)",
            min_value=6,
            max_value=36,
            value=24,
            step=6,
            help="Número de meses para projeção"
        )

    # Criar projeção temporal
    meses = range(1, meses_projecao + 1)
    projecao = pd.DataFrame({
        'Mês': meses,
        'Volume_Base': [meta_inicial * (1 + incremento_mensal/100) ** (m-1) for m in meses]
    })
    
    projecao['Volume_Convertido'] = projecao['Volume_Base'] * (conversion_rate)
    projecao['Receita'] = projecao['Volume_Convertido'] * taxa_media_atual
    projecao['Clientes_Estimados'] = (projecao['Volume_Base'] / volume_medio_atual).round(0)
    projecao['Meta_Acumulada'] = projecao['Volume_Convertido'].cumsum()
    
    # Visualização da projeção
    metrica_tempo = st.selectbox(
        "Selecione a Métrica para Análise Temporal",
        ["Volumes", "Receita", "Acumulado"],
        key="metrica_tempo"
    )
    
    fig_tempo = go.Figure()
    
    if metrica_tempo == "Volumes":
        fig_tempo.add_trace(go.Scatter(
            x=projecao['Mês'],
            y=projecao['Volume_Base'],
            name='Volume Base',
            mode='lines+markers',
            line=dict(dash='dot')
        ))
        fig_tempo.add_trace(go.Scatter(
            x=projecao['Mês'],
            y=projecao['Volume_Convertido'],
            name='Volume Convertido',
            mode='lines+markers'
        ))
    elif metrica_tempo == "Receita":
        fig_tempo.add_trace(go.Scatter(
            x=projecao['Mês'],
            y=projecao['Receita'],
            name='Receita Mensal',
            mode='lines+markers',
            fill='tozeroy'
        ))
    else:
        fig_tempo.add_trace(go.Scatter(
            x=projecao['Mês'],
            y=projecao['Meta_Acumulada'],
            name='Volume Acumulado',
            mode='lines+markers',
            fill='tozeroy'
        ))
    
    fig_tempo.update_layout(
        title=f'Projeção {metrica_tempo} - {meses_projecao} Meses',
        height=500,
        **layout_config
    )
    
    st.plotly_chart(fig_tempo, use_container_width=True)

    # Métricas de projeção
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Volume Final",
            f"R$ {projecao['Volume_Convertido'].iloc[-1]:,.2f}",
            delta=f"{((projecao['Volume_Convertido'].iloc[-1] / projecao['Volume_Convertido'].iloc[0]) - 1):.1%}"
        )
    with col2:
        st.metric(
            "Total Acumulado",
            f"R$ {projecao['Meta_Acumulada'].iloc[-1]:,.2f}"
        )
    with col3:
        st.metric(
            "Média Mensal",
            f"R$ {projecao['Volume_Convertido'].mean():,.2f}"
        )

with tab_metas:
    st.markdown("### 🎯 Análise de Metas")
    
    # Seletor de visualização de metas
    tipo_meta = st.radio(
        "Tipo de Análise",
        ["Metas por Período", "Distribuição de Metas", "Comparativo"],
        horizontal=True
    )
    
    if tipo_meta == "Metas por Período":
        # Análise de metas por período
        periodos = ['Mensal', 'Trimestral', 'Semestral', 'Anual']
        metas_periodo = pd.DataFrame({
            'Período': periodos,
            'Meta Base': [
                projecao['Volume_Base'].mean(),
                projecao['Volume_Base'].rolling(3).mean().mean(),
                projecao['Volume_Base'].rolling(6).mean().mean(),
                projecao['Volume_Base'].mean() * 12
            ],
            'Meta Convertida': [
                projecao['Volume_Convertido'].mean(),
                projecao['Volume_Convertido'].rolling(3).mean().mean(),
                projecao['Volume_Convertido'].rolling(6).mean().mean(),
                projecao['Volume_Convertido'].mean() * 12
            ]
        })
        
        st.dataframe(
            metas_periodo.style.format({
                'Meta Base': 'R$ {:,.2f}',
                'Meta Convertida': 'R$ {:,.2f}'
            }),
            height=400
        )
        
    elif tipo_meta == "Distribuição de Metas":
        # Gráfico de distribuição das metas
        fig_dist = go.Figure()
        
        fig_dist.add_trace(go.Box(
            y=projecao['Volume_Convertido'],
            name='Distribuição das Metas',
            boxpoints='all',
            jitter=0.3,
            pointpos=-1.8
        ))
        
        fig_dist.update_layout(
            title='Distribuição das Metas Mensais',
            height=500,
            **layout_config
        )
        
        st.plotly_chart(fig_dist, use_container_width=True)
        
    else:
        # Comparativo de metas
        fig_comp = go.Figure()
        
        fig_comp.add_trace(go.Bar(
            name='Meta Base',
            x=['Mínima', 'Média', 'Máxima'],
            y=[
                projecao['Volume_Base'].min(),
                projecao['Volume_Base'].mean(),
                projecao['Volume_Base'].max()
            ],
            text=[
                f'R$ {projecao["Volume_Base"].min():,.2f}',
                f'R$ {projecao["Volume_Base"].mean():,.2f}',
                f'R$ {projecao["Volume_Base"].max():,.2f}'
            ],
            textposition='auto',
        ))
        
        fig_comp.add_trace(go.Bar(
            name='Meta Convertida',
            x=['Mínima', 'Média', 'Máxima'],
            y=[
                projecao['Volume_Convertido'].min(),
                projecao['Volume_Convertido'].mean(),
                projecao['Volume_Convertido'].max()
            ],
            text=[
                f'R$ {projecao["Volume_Convertido"].min():,.2f}',
                f'R$ {projecao["Volume_Convertido"].mean():,.2f}',
                f'R$ {projecao["Volume_Convertido"].max():,.2f}'
            ],
            textposition='auto',
        ))
        
        fig_comp.update_layout(
            title='Comparativo de Metas',
            barmode='group',
            height=500,
            **layout_config
        )
        
        st.plotly_chart(fig_comp, use_container_width=True)

# Download dos dados
col1, col2 = st.columns(2)
with col1:
    # Botão de download da análise
    st.download_button(
        label="Download da Análise Completa",
        data=analysis_df.to_csv().encode('utf-8'),
        file_name='analise_completa.csv',
        mime='text/csv'
    )

with col2:
    # Documentação e metodologia
    st.info("""
    **Metodologia**
    - Taxa de conversão aplicada: {:.1%}
    - Taxa alvo definida: {:.1%}
    - Período de análise: Mensal
    """.format(conversion_rate, target_rate))

# Seção Otimizada de Análise de Abertura de Mercado
st.markdown("""
    <div style='background-color: #f8f9fa; padding: 1.5rem; border-radius: 8px; margin: 2rem 0;'>
        <h2 style='color: #2E4057; margin-bottom: 1rem;'>Simulador de Expansão de Mercado</h2>
        <p style='color: #666; font-size: 1.1rem;'>
            Ferramenta avançada para simulação de cenários de expansão, projeções e análise de metas.
        </p>
    </div>
""", unsafe_allow_html=True)

# Download dos dados de simulação
col1, col2 = st.columns(2)
with col1:
    st.download_button(
        label="Download Simulação de Cenários",
        data=simulacao_df.to_csv(index=False).encode('utf-8'),
        file_name='simulacao_cenarios.csv',
        mime='text/csv',
        key='download_simulacao'
    )

with col2:
    st.download_button(
        label="Download Projeção Temporal",
        data=projecao.to_csv(index=False).encode('utf-8'),
        file_name='projecao_temporal.csv',
        mime='text/csv',
        key='download_projecao_temporal'
    )

# Seção de Análise de Crescimento
st.markdown("#### 📈 Análise de Crescimento")

col1, col2 = st.columns(2)

with col1:
    st.info(
        """
        **Crescimento Total**
        - Volume inicial: R$ {:,.2f}
        - Volume final: R$ {:,.2f}
        - Crescimento total: {:.1%}
        """.format(
            projecao['Volume_Convertido'].iloc[0],
            projecao['Volume_Convertido'].iloc[-1],
            (projecao['Volume_Convertido'].iloc[-1] / projecao['Volume_Convertido'].iloc[0]) - 1
        )
    )

with col2:
    st.info(
        """
        **Médias e Totais**
        - Volume médio mensal: R$ {:,.2f}
        - Total acumulado: R$ {:,.2f}
        - Média de clientes: {:,.0f}
        """.format(
            projecao['Volume_Convertido'].mean(),
            projecao['Meta_Acumulada'].iloc[-1],
            projecao['Clientes_Estimados'].mean()
        )
    )



