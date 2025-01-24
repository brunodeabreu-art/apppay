import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np
import calendar
from plotly.subplots import make_subplots
from prophet import Prophet  # Para previsões avançadas
import tensorflow as tf  # Para análise preditiva
from sklearn.cluster import KMeans  # Para clustering
from scipy import stats  # Para análises estatísticas
import plotly.figure_factory as ff  # Para visualizações avançadas
import networkx as nx  # Para análise de rede/dependências
from textblob import TextBlob  # Para análise de sentimento
import pycaret.regression as pcr  # Para AutoML
import altair as alt
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool
import holoviews as hv
from holoviews import opts
import seaborn as sns
from scipy.stats import gaussian_kde
from PIL import Image

# Configuração da página
st.set_page_config(
    page_title="Cockpit @ CEO ",
    layout="wide", 
    initial_sidebar_state="expanded"
)

# Estilo personalizado
st.markdown("""
<style>
    /* Tema geral */
    .main {
        background-color: #f8f9fa;
    }
    
    /* Cards de métricas */
    .metric-card {
        background: linear-gradient(145deg, #ffffff, #f0f2f6);
        border-radius: 15px;
        padding: 20px;
        box-shadow: 5px 5px 15px rgba(0,0,0,0.08);
        margin: 10px 0;
        transition: transform 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    /* Indicadores de risco */
    .high-risk { 
        color: #1e3d59; 
        font-weight: bold;
    }
    .medium-risk { 
        color: #2b5876;
        font-weight: bold;
    }
    .low-risk { 
        color: #4682B4;
        font-weight: bold;
    }
    
    /* Headers e títulos */
    .metric-title { 
        font-size: 1.2em; 
        font-weight: bold;
        color: #1e3d59;
    }
    .section-header {
        background: linear-gradient(90deg, #1e3d59, #2b5876);
        color: white;
        padding: 10px 20px;
        border-radius: 10px;
        margin: 20px 0;
    }
    
    /* Progress bars */
    .stProgress > div > div > div > div {
        background-color: #1e3d59;
    }
    
    /* Departament badges */
    .dept-badge {
        padding: 5px 10px;
        border-radius: 15px;
        font-size: 12px;
        font-weight: bold;
    }
    
    /* Custom tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 5px;
        padding: 10px 20px;
        color: #1e3d59;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1e3d59;
        color: white;
    }
    .small-font {
        font-size: 7px;
        padding: 2px;
    }
    
    /* Ajuste para o logo */
    .sidebar-logo {
        margin-bottom: 20px;
        text-align: center;
    }
    
    .sidebar-logo img {
        max-width: 80%;
        height: auto;
    }
</style>
""", unsafe_allow_html=True)

# Cores personalizadas para gráficos (nova paleta mais sóbria)
CUSTOM_COLORS = {
    'primary': '#2C3E50',    # Azul escuro
    'secondary': '#34495E',  # Azul acinzentado
    'tertiary': '#7F8C8D',  # Cinza
    'quaternary': '#95A5A6', # Cinza claro
    'quinary': '#BDC3C7'    # Cinza muito claro
}

# Configurações de tema para gráficos Plotly
PLOTLY_THEME = {
    'layout': {
        'plot_bgcolor': 'rgba(0,0,0,0)',
        'paper_bgcolor': 'rgba(0,0,0,0)',
        'font': {'color': '#2C3E50'},
        'title': {'font': {'size': 24, 'color': '#2C3E50'}},
        'margin': {'t': 40, 'b': 40, 'l': 40, 'r': 40},
        'colorway': ['#2C3E50', '#34495E', '#7F8C8D', '#95A5A6', '#BDC3C7']
    }
}

# Dados de exemplo expandidos
DEPARTMENTS = {
    "marketing": "Marketing",
    "sales": "Vendas", 
    "product": "Produto",
    "support": "Suporte",
    "tech_green": "Tech Green",
    "tech_blue": "Tech Blue",
    "data": "Dados",
    "kam": "KAM",
    "implementation": "Implantação"
}

# Geração de dados simulados para 20 projetos
np.random.seed(42)

def gerar_projetos_simulados(n_projetos=20):
    projetos = []
    nomes_projetos = [
        "Automação de Lead Scoring", "Portal de Autoatendimento", "Integração HL7 FHIR",
        "Analytics Platform 2.0", "Expansão Regional Sul", "CRM Healthcare",
        "Mobile App v2", "Chatbot IA", "Gestão de Prontuários", "Telemedicina Plus",
        "Marketplace Saúde", "Business Intelligence", "API Gateway", "Customer Success",
        "Programa de Fidelidade", "Gestão de Parceiros", "Automação de Marketing",
        "Sistema de Billing", "Portal do Paciente", "Integração Labs"
    ]
    
    for i in range(n_projetos):
        inicio = datetime.now() + timedelta(days=np.random.randint(-60, 60))
        duracao = np.random.randint(90, 365)
        orcamento = np.random.randint(100000, 1000000)
        progresso = np.random.randint(0, 100)
        
        projeto = {
            "id": i + 1,
            "nome": nomes_projetos[i],
            "departamento": np.random.choice(list(DEPARTMENTS.keys())),
            "inicio": inicio,
            "duracao": duracao,
            "orcamento": orcamento,
            "gasto": orcamento * np.random.uniform(0.6, 1.2),
            "progresso": progresso,
            "status": np.random.choice(["Em Dia", "Atrasado", "Crítico"], p=[0.6, 0.3, 0.1]),
            "prioridade": np.random.choice(["Alta", "Média", "Baixa"]),
            "risco": np.random.choice(["Alto", "Médio", "Baixo"]),
            "recursos_alocados": np.random.randint(3, 15),
            "velocidade_sprint": np.random.uniform(60, 100),
            "bugs_criticos": np.random.randint(0, 10),
            "satisfacao_cliente": np.random.uniform(7, 10),
            "roi_esperado": np.random.uniform(1.5, 4.0)
        }
        
        projetos.append(projeto)
    
    return pd.DataFrame(projetos)

PROJECTS = gerar_projetos_simulados(20)
df = pd.DataFrame(PROJECTS)

# Funções auxiliares
def calcular_metricas_estrategicas(df):
    """Calcula métricas estratégicas para o dashboard executivo"""
    return {
        "burn_rate_mensal": df['gasto'].sum() / 6,
        "roi_medio": df['roi_esperado'].mean(),
        "projetos_estrategicos": len(df[df['impacto_estrategico'] == 'Alto']),
        "health_score": calcular_health_score(df),
        "velocidade_media": df['velocidade_sprint'].mean(),
        "satisfacao_media": df['satisfacao_cliente'].mean(),
        "tempo_medio_mercado": df['time_to_market'].mean()
    }

def calcular_health_score(df):
    """Calcula o health score geral do portfólio"""
    progresso_medio = df['progresso'].mean() / 100
    orcamento_saude = 1 - (df['gasto'].sum() / df['orcamento'].sum())
    risco_score = len(df[df['risco'] == 'Baixo']) / len(df)
    return (progresso_medio * 0.4 + orcamento_saude * 0.4 + risco_score * 0.2)

def gerar_previsao_financeira(df):
    """Gera previsão financeira para os próximos 12 meses"""
    meses = pd.date_range(start=datetime.now(), periods=12, freq='M')
    previsao = pd.DataFrame({
        'mes': meses,
        'receita': np.random.normal(1000000, 100000, 12).cumsum(),
        'custos': np.random.normal(800000, 80000, 12).cumsum(),
        'margem': np.random.normal(200000, 20000, 12).cumsum()
    })
    return previsao

def gerar_analise_departamental(df):
    """Gera análises comparativas entre departamentos"""
    dept_analysis = df.groupby('departamento').agg({
        'orcamento': 'sum',
        'gasto': 'sum',
        'progresso': 'mean',
        'roi_esperado': 'mean',
        'satisfacao_cliente': 'mean',
        'velocidade_sprint': 'mean'
    }).reset_index()
    
    dept_analysis['eficiencia_custo'] = (dept_analysis['orcamento'] - dept_analysis['gasto']) / dept_analysis['orcamento']
    return dept_analysis

def update_fig_layout(fig):
    """Atualiza o layout do gráfico com o tema padrão"""
    fig.update_layout(
        **PLOTLY_THEME['layout'],
        coloraxis_colorscale=[
            [0, '#2C3E50'],    # Mais escuro
            [0.25, '#34495E'],
            [0.5, '#7F8C8D'],
            [0.75, '#95A5A6'],
            [1, '#BDC3C7']     # Mais claro
        ]
    )
    return fig

def criar_matriz_riscos(df):
    """Cria matriz de riscos baseada em complexidade e impacto"""
    fig = px.scatter(df,
        x='complexidade',
        y='impacto_estrategico',
        size='orcamento',
        color='risco',
        hover_data=['nome', 'progresso', 'gasto'],
        title='Matriz de Riscos e Complexidade',
        color_discrete_map={
            'Alto': '#2C3E50',
            'Médio': '#7F8C8D',
            'Baixo': '#BDC3C7'
        }
    )
    return update_fig_layout(fig)

def render_advanced_filters(df):
    """Renderiza filtros avançados para o dashboard"""
    with st.sidebar:
        st.header("Filtros Avançados")
        
        # Filtro de progresso
        st.subheader("Progresso")
        progresso_range = st.slider(
            label="Faixa de Progresso (%)",
            min_value=int(df['progresso'].min()),
            max_value=int(df['progresso'].max()),
            value=(int(df['progresso'].min()), int(df['progresso'].max())))
        
        # Filtro de ROI
        st.subheader("ROI")
        roi_range = st.slider(
            label="Faixa de ROI Esperado",
            min_value=float(df['roi_esperado'].min()),
            max_value=float(df['roi_esperado'].max()),
            value=(float(df['roi_esperado'].min()), float(df['roi_esperado'].max())))
        
        # Filtro de prioridade
        st.subheader("Prioridade")
        prioridades = st.multiselect(
            label="Selecione as Prioridades",
            options=df['prioridade'].unique(),
            default=df['prioridade'].unique())
        
        
        # Filtro de complexidade
        st.subheader("Complexidade")
        complexidades = st.multiselect(
            label="Selecione as Complexidades",
            options=df['complexidade'].unique(),
            default=df['complexidade'].unique())
        
        
        # Filtro de bugs críticos
        st.subheader("Bugs Críticos")
        bugs_range = st.slider(
            label="Faixa de Bugs Críticos",
            min_value=int(df['bugs_criticos'].min()),
            max_value=int(df['bugs_criticos'].max()),
            value=(int(df['bugs_criticos'].min()), int(df['bugs_criticos'].max())))
        
        # Filtro de velocidade de sprint
        st.subheader("Velocidade de Sprint")
        velocidade_range = st.slider(
            label="Faixa de Velocidade de Sprint",
            min_value=float(df['velocidade_sprint'].min()),
            max_value=float(df['velocidade_sprint'].max()),
            value=(float(df['velocidade_sprint'].min()), float(df['velocidade_sprint'].max())))
        
        
    # Retornar todos os filtros em um dicionário
    return {
        'progresso_range': progresso_range,
        'roi_range': roi_range,
        'prioridade': prioridades,
        'complexidade': complexidades,
        'bugs_range': bugs_range,
        'velocidade_range': velocidade_range
    }

def analise_preditiva_projetos(df):
    """Realiza análise preditiva de métricas chave dos projetos"""
    # Preparar dados para previsão
    df_prophet = df[['inicio', 'progresso']].copy()
    df_prophet.columns = ['ds', 'y']
    
    # Criar e treinar modelo Prophet
    model = Prophet(
        changepoint_prior_scale=0.05,
        seasonality_mode='multiplicative'
    )
    model.fit(df_prophet)
    
    # Fazer previsões
    future = model.make_future_dataframe(periods=90)
    forecast = model.predict(future)
    
    return forecast

def clustering_projetos(df):
    """Realiza clustering dos projetos baseado em múltiplas dimensões"""
    # Preparar features para clustering
    features = df[[
        'progresso', 'roi_esperado', 'time_to_market',
        'recursos_alocados', 'satisfacao_cliente'
    ]].values
    
    # Normalizar dados
    features_norm = stats.zscore(features)
    
    # Aplicar K-means
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(features_norm)
    
    return clusters

def criar_visualizacao_rede(df):
    """Cria visualização de rede usando networkx e plotly"""
    # Criar grafo
    G = nx.Graph()
    
    # Adicionar nós (projetos)
    for _, proj in df.iterrows():
        G.add_node(proj['nome'], 
                  departamento=proj['departamento'],
                  progresso=proj['progresso'])
    
    # Adicionar arestas baseadas em departamentos
    for i, proj1 in df.iterrows():
        for j, proj2 in df.iterrows():
            if i < j and proj1['departamento'] == proj2['departamento']:
                G.add_edge(proj1['nome'], proj2['nome'])
    
    # Calcular layout
    pos = nx.spring_layout(G)
    
    # Criar traces para as arestas
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines'
    )

    # Criar traces para os nós
    node_x = []
    node_y = []
    node_text = []
    node_color = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(f"Projeto: {node}<br>"
                        f"Dept: {G.nodes[node]['departamento']}<br>"
                        f"Progresso: {G.nodes[node]['progresso']}%")
        # Cor baseada no departamento
        node_color.append(hash(G.nodes[node]['departamento']) % 20)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            showscale=True,
            colorscale='Blues',
            size=20,
            color=node_color,
            line_width=2
        )
    )

    # Criar figura
    fig = go.Figure(data=[edge_trace, node_trace],
                   layout=go.Layout(
                       title='Rede de Projetos por Departamento',
                       showlegend=False,
                       hovermode='closest',
                       margin=dict(b=20,l=5,r=5,t=40),
                       xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                   ))
    
    return fig

def render_analise_avancada(df):
    """Renderiza análises avançadas no dashboard"""
    st.markdown('<div class="section-header">🔬 Análise Avançada</div>', 
                unsafe_allow_html=True)
    
    # Adicionar tabs para diferentes análises
    tab1, tab2, tab3 = st.tabs(["Previsões", "Clustering", "Rede de Projetos"])
    
    with tab1:
        # Previsões com Prophet
        forecast = analise_preditiva_projetos(df)
        fig_forecast = go.Figure()
        fig_forecast.add_trace(go.Scatter(
            x=forecast['ds'],
            y=forecast['yhat'],
            name='Previsão',
            mode='lines',
            line_color='#1e3d59'
        ))
        fig_forecast.add_trace(go.Scatter(
            x=forecast['ds'],
            y=forecast['yhat_upper'],
            fill=None,
            mode='lines',
            line_color='#2b5876',
            name='Intervalo Superior'
        ))
        fig_forecast.add_trace(go.Scatter(
            x=forecast['ds'],
            y=forecast['yhat_lower'],
            fill='tonexty',
            mode='lines',
            line_color='#4682B4',
            name='Intervalo Inferior'
        ))
        st.plotly_chart(fig_forecast)
    
    with tab2:
        # Clustering
        clusters = clustering_projetos(df)
        df['cluster'] = clusters
        fig_cluster = px.scatter_3d(
            df,
            x='progresso',
            y='roi_esperado',
            z='satisfacao_cliente',
            color='cluster',
            hover_data=['nome'],
            title='Clustering 3D de Projetos',
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig_cluster)
    
    with tab3:
        # Visualização de rede
        fig_network = criar_visualizacao_rede(df)
        st.plotly_chart(fig_network)

def add_anomaly_detection(df):
    """Adiciona detecção de anomalias nas métricas"""
    # Usar autoencoder para detecção de anomalias
    input_dim = 5
    encoding_dim = 3
    
    autoencoder = tf.keras.Sequential([
        tf.keras.layers.Dense(encoding_dim, activation='relu', 
                            input_shape=(input_dim,)),
        tf.keras.layers.Dense(input_dim, activation='sigmoid')
    ])
    
    autoencoder.compile(optimizer='adam', loss='mse')
    
    # Preparar dados
    features = df[[
        'progresso', 'roi_esperado', 'satisfacao_cliente',
        'velocidade_sprint', 'bugs_criticos'
    ]].values
    
    # Normalizar
    features_norm = stats.zscore(features)
    
    # Treinar
    autoencoder.fit(features_norm, features_norm, 
                   epochs=50, batch_size=32, verbose=0)
    
    # Detectar anomalias
    reconstructed = autoencoder.predict(features_norm)
    mse = np.mean(np.power(features_norm - reconstructed, 2), axis=1)
    threshold = np.percentile(mse, 95)
    
    return mse > threshold

def criar_visualizacoes_avancadas(df):
    """Cria visualizações avançadas dos dados"""
    
    # Calcular métricas por departamento primeiro
    dept_metrics = df.groupby('departamento').agg({
        'progresso': 'mean',
        'bugs_criticos': 'sum',
        'velocidade_sprint': 'mean',
        'satisfacao_cliente': 'mean',
        'roi_esperado': 'mean'
    }).round(2)
    
    # 1. Visão Geral por Departamento
    st.subheader("Visão Geral por Departamento")
    
    col1, col2, col3 = st.columns(3)
    
    for idx, dept in enumerate(df['departamento'].unique()):
        dept_data = df[df['departamento'] == dept]
        
        with col1 if idx % 3 == 0 else col2 if idx % 3 == 1 else col3:
            st.metric(
                dept,
                f"{dept_data['progresso'].mean():.1f}%",
                f"{dept_data['velocidade_sprint'].mean():.1f} vel."
            )
    
    # 2. Análise de Progresso
    st.subheader("Análise de Progresso")
    
    # Criar gráfico de progresso
    fig_progress = go.Figure()
    
    for dept in df['departamento'].unique():
        dept_data = df[df['departamento'] == dept]
        
        fig_progress.add_trace(go.Bar(
            name=dept,
            x=[dept],
            y=[dept_data['progresso'].mean()],
            text=[f"{dept_data['progresso'].mean():.1f}%"],
            textposition='auto',
            marker_color='#1e3d59'
        ))
    
    fig_progress.update_layout(
        title='Progresso Médio por Departamento',
        yaxis_title='Progresso (%)',
        showlegend=False
    )
    
    st.plotly_chart(fig_progress, use_container_width=True)
    
    # 3. Matriz de Desempenho
    st.subheader("Matriz de Desempenho")
    
    fig_matrix = px.scatter(
        df,
        x='velocidade_sprint',
        y='satisfacao_cliente',
        size='bugs_criticos',
        color='departamento',
        hover_data=['nome', 'progresso'],
        title='Velocidade vs Satisfação',
        color_discrete_sequence=px.colors.sequential.Blues
    )
    
    st.plotly_chart(fig_matrix, use_container_width=True)
    
    # 4. Análise de ROI
    st.subheader("Análise de ROI")
    
    fig_roi = px.bar(
        dept_metrics,
        y=dept_metrics.index,
        x='roi_esperado',
        orientation='h',
        title='ROI Esperado por Departamento',
        color_discrete_sequence=['#1e3d59']
    )
    
    fig_roi.update_traces(text=dept_metrics['roi_esperado'].round(2), textposition='auto')
    
    st.plotly_chart(fig_roi, use_container_width=True)
    
    # 5. Insights
    st.subheader("Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🎯 Performance")
        
        # Encontrar departamentos com melhor desempenho
        best_dept = dept_metrics['velocidade_sprint'].idxmax()
        best_roi = dept_metrics['roi_esperado'].idxmax()
        
        st.markdown(f"""
        - Maior velocidade: **{best_dept}** ({dept_metrics.loc[best_dept, 'velocidade_sprint']:.1f})
        - Melhor ROI: **{best_roi}** ({dept_metrics.loc[best_roi, 'roi_esperado']:.2f}x)
        - Satisfação média: **{df['satisfacao_cliente'].mean():.1f}/10**
        """)
        
    with col2:
        st.markdown("#### ⚠️ Atenção Necessária")
        
        # Identificar departamentos que precisam de atenção
        low_progress = dept_metrics[dept_metrics['progresso'] < 70].index.tolist()
        high_bugs = dept_metrics[dept_metrics['bugs_criticos'] > dept_metrics['bugs_criticos'].mean()].index.tolist()
        
        if low_progress:
            st.markdown(f"- Progresso baixo: **{', '.join(low_progress)}**")
        if high_bugs:
            st.markdown(f"- Muitos bugs: **{', '.join(high_bugs)}**")

def render_summary(df):
    """Renderiza o sumário executivo com UI/UX avançada"""
    
    # Estilo personalizado aprimorado
    st.markdown("""
        <style>
        /* Estilos gerais */
        .summary-header {
            font-size: 24px;
            font-weight: 600;
            color: #0068c9;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #f0f2f6;
        }
        
        /* KPI Cards Animados */
        .kpi-card {
            background: white;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            transition: all 0.3s ease;
            cursor: pointer;
            position: relative;
            overflow: hidden;
        }
        
        .kpi-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 4px 15px rgba(0,0,0,0.15);
        }
        
        .kpi-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 4px;
            background: linear-gradient(90deg, #0068c9, #00a8e8);
            opacity: 0;
            transition: opacity 0.3s ease;
        }
        
        .kpi-card:hover::before {
            opacity: 1;
        }
        
        /* Estilos para métricas */
        .metric-value {
            font-size: 24px;
            font-weight: 700;
            color: #0068c9;
            margin: 10px 0;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .metric-delta {
            font-size: 14px;
            padding: 4px 8px;
            border-radius: 12px;
            font-weight: 500;
        }
        
        .metric-delta.positive {
            background-color: #e6f4ea;
            color: #137333;
        }
        
        .metric-delta.negative {
            background-color: #fce8e6;
            color: #c5221f;
        }
        
        .metric-label {
            font-size: 14px;
            color: #5f6368;
            margin-bottom: 5px;
        }
        
        /* Animação de pulso para alertas */
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.05); }
            100% { transform: scale(1); }
        }
        
        .alert-pulse {
            animation: pulse 2s infinite;
        }
        
        /* Tooltips personalizados */
        .tooltip {
            position: relative;
            display: inline-block;
        }
        
        .tooltip .tooltiptext {
            visibility: hidden;
            width: 200px;
            background-color: #333;
            color: white;
            text-align: center;
            border-radius: 6px;
            padding: 8px;
            position: absolute;
            z-index: 1;
            bottom: 125%;
            left: 50%;
            transform: translateX(-50%);
            opacity: 0;
            transition: opacity 0.3s;
        }
        
        .tooltip:hover .tooltiptext {
            visibility: visible;
            opacity: 1;
        }
        </style>
    """, unsafe_allow_html=True)

    # Cabeçalho
    st.markdown('<div class="summary-header"></div>', unsafe_allow_html=True)

    # KPIs Principais com interatividade
    col1, col2, col3, col4 = st.columns(4)
    
    # Função auxiliar para gerar HTML do KPI
    def create_kpi_card(label, value, delta, icon, tooltip):
        """Cria um card KPI com estilo personalizado"""
        # Remove o símbolo % ou x do delta para conversão
        delta_value = float(delta.replace('%', '').replace('x', ''))
        delta_class = "positive" if delta_value >= 0 else "negative"
        delta_icon = "↑" if delta_class == "positive" else "↓"
        
        return f"""
        <div class="kpi-card tooltip">
            <div class="metric-label">{icon} {label}</div>
            <div class="metric-value">
                {value}
                <span class="metric-delta {delta_class}">
                    {delta_icon} {delta}
                </span>
            </div>
            <span class="tooltiptext">{tooltip}</span>
        </div>
        """

    with col1:
        progresso_medio = df['progresso'].mean()
        delta_progresso = progresso_medio - 75
        st.markdown(
            create_kpi_card(
                "Progresso Geral",
                f"{progresso_medio:.1f}%",
                f"{delta_progresso:.1f}%",
                "🎯",
                "Progresso médio de todos os projetos vs. meta de 75%"
            ),
            unsafe_allow_html=True
        )

    with col2:
        projetos_atrasados = len(df[df['status'] == 'Atrasado'])
        perc_atrasados = (projetos_atrasados/len(df))*100
        st.markdown(
            create_kpi_card(
                "Projetos Atrasados",
                str(projetos_atrasados),
                f"{perc_atrasados:.1f}%",
                "⚠️",
                "Número total de projetos atrasados e percentual do portfolio"
            ),
            unsafe_allow_html=True
        )

    with col3:
        roi_medio = df['roi_esperado'].mean()
        delta_roi = roi_medio - 2
        st.markdown(
            create_kpi_card(
                "ROI Médio",
                f"{roi_medio:.2f}x",
                f"{delta_roi:.2f}x",
                "💰",
                "Retorno sobre investimento médio vs. target de 2x"
            ),
            unsafe_allow_html=True
        )

    with col4:
        satisfacao = df['satisfacao_cliente'].mean()
        delta_satisfacao = satisfacao - 8.5
        st.markdown(
            create_kpi_card(
                "Satisfação Cliente",
                f"{satisfacao:.1f}/10",
                f"{delta_satisfacao:.1f}",
                "⭐",
                "Índice médio de satisfação dos clientes vs. meta de 8.5"
            ),
            unsafe_allow_html=True
        )

    # 2. Gráficos Principais
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.subheader("Status dos Projetos")
        
        # Gráfico de status
        status_count = df['status'].value_counts()
        fig_status = px.pie(
            values=status_count.values,
            names=status_count.index,
            title="Distribuição de Status",
            hole=0.4,
            color_discrete_sequence=px.colors.sequential.Blues
        )
        fig_status.update_layout(height=300)
        st.plotly_chart(fig_status, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.subheader("Progresso por Departamento")
        
        # Gráfico de progresso
        dept_progress = df.groupby('departamento')['progresso'].mean().sort_values(ascending=True)
        fig_progress = px.bar(
            x=dept_progress.values,
            y=dept_progress.index,
            orientation='h',
            title="Progresso Médio (%)",
            color=dept_progress.values,
            color_continuous_scale='Blues'
        )
        fig_progress.update_layout(height=300)
        st.plotly_chart(fig_progress, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # 3. Tabela de Destaques
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.subheader("Projetos em Destaque")
    
    destaques = df.nlargest(5, 'roi_esperado')[
        ['nome', 'departamento', 'progresso', 'roi_esperado', 'status']
    ]
    
    # Formatação da tabela
    destaques['progresso'] = destaques['progresso'].apply(lambda x: f"{x:.1f}%")
    destaques['roi_esperado'] = destaques['roi_esperado'].apply(lambda x: f"{x:.2f}x")
    
    st.dataframe(
        destaques,
        column_config={
            "nome": "Projeto",
            "departamento": "Departamento",
            "progresso": "Progresso",
            "roi_esperado": "ROI Esperado",
            "status": "Status"
        },
        hide_index=True,
        use_container_width=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # 4. Insights e Alertas
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.subheader("🎯 Principais Insights")
        
        # Calcular insights
        melhor_dept = df.groupby('departamento')['roi_esperado'].mean().idxmax()
        maior_atraso = df[df['status'] == 'Atrasado']['departamento'].mode().iloc[0]
        st.markdown(f"""
        - Melhor ROI: **{melhor_dept}** ({df[df['departamento'] == melhor_dept]['roi_esperado'].mean():.2f}x)
        - Dept. mais atrasado: **{maior_atraso}**
        - Progresso geral: **{progresso_medio:.1f}%**
        - Satisfação média: **{satisfacao:.1f}/10**
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.subheader("⚠️ Alertas Principais")
        
        # Identificar alertas
        atrasados = df[df['status'] == 'Atrasado']['departamento'].unique()
        baixo_roi = df[df['roi_esperado'] < 2]['departamento'].unique()
        
        if len(atrasados) > 0:
            st.warning(f"Atrasos em: **{', '.join(atrasados)}**")
        if len(baixo_roi) > 0:
            st.warning(f"ROI baixo em: **{', '.join(baixo_roi)}**")
        if progresso_medio < 75:
            st.warning(f"Progresso geral abaixo da meta: **{progresso_medio:.1f}%**")
        st.markdown('</div>', unsafe_allow_html=True)

def render_visao_geral(df):
    """Renderiza a visão geral do dashboard"""
    st.markdown('<div class="section-header">📊 Visão Geral</div>', 
                unsafe_allow_html=True)
    
    # 1. KPIs Principais
    st.subheader("KPIs Principais")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_projetos = len(df)
        projetos_atrasados = len(df[df['status'] == 'Atrasado'])
        st.metric(
            "Total de Projetos",
            f"{total_projetos}",
            f"{projetos_atrasados} atrasados"
        )
        
    with col2:
        orcamento_total = df['orcamento'].sum()
        gasto_total = df['gasto'].sum()
        st.metric(
            "Orçamento Total",
            f"R$ {orcamento_total:,.2f}",
            f"{((gasto_total/orcamento_total - 1) * 100):.1f}% vs planejado"
        )
        
    with col3:
        progresso_medio = df['progresso'].mean()
        st.metric(
            "Progresso Médio",
            f"{progresso_medio:.1f}%",
            f"{(progresso_medio - 100):.1f}% para conclusão"
        )
        
    with col4:
        roi_medio = df['roi_esperado'].mean()
        st.metric(
            "ROI Médio Esperado",
            f"{roi_medio:.2f}x",
            f"{(roi_medio - 2):.2f}x vs target"
        )
    
    # 2. Distribuição de Status
    st.subheader("Distribuição de Status")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Status por departamento
        fig_status = px.bar(
            df.groupby(['departamento', 'status']).size().reset_index(name='count'),
            x='departamento',
            y='count',
            color='status',
            title='Status por Departamento',
            barmode='group'
        )
        st.plotly_chart(fig_status)
        
    with col2:
        # Distribuição de prioridade
        fig_prioridade = px.pie(
            df,
            names='prioridade',
            title='Distribuição de Prioridade',
            color='prioridade',
            color_discrete_map={
                'Alta': '#ff6b6b',
                'Média': '#4ecdc4',
                'Baixa': '#45b7d1'
            }
        )
        st.plotly_chart(fig_prioridade)
    
    # 3. Progresso dos Projetos
    st.subheader("Progresso dos Projetos")
    
    # Ordenar projetos por progresso
    df_progresso = df.sort_values('progresso', ascending=True).tail(10)
    
    fig_progresso = px.bar(
        df_progresso,
        x='progresso',
        y='nome',
        orientation='h',
        title='Top 10 Projetos por Progresso',
        color='status'
    )
    
    fig_progresso.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig_progresso, use_container_width=True)
    
    # 4. Análise de Recursos
    st.subheader("Análise de Recursos")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Recursos alocados por departamento
        fig_recursos = px.box(
            df,
            x='departamento',
            y='recursos_alocados',
            title='Distribuição de Recursos por Departamento'
        )
        st.plotly_chart(fig_recursos)
        
    with col2:
        # Relação entre recursos e progresso
        fig_rel = px.scatter(
            df,
            x='recursos_alocados',
            y='progresso',
            color='departamento',
            size='orcamento',
            title='Recursos vs Progresso',
            hover_data=['nome']
        )
        st.plotly_chart(fig_rel)
    
    # 5. Insights e Recomendações
    st.subheader("Insights e Recomendações")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 💡 Principais Insights")
        
        # Calcular métricas relevantes
        dept_performance = df.groupby('departamento')['progresso'].mean().sort_values(ascending=False)
        best_dept = dept_performance.index[0]
        worst_dept = dept_performance.index[-1]
        
        st.markdown(f"""
        - Melhor departamento: **{best_dept}** ({dept_performance[best_dept]:.1f}% progresso)
        - Departamento crítico: **{worst_dept}** ({dept_performance[worst_dept]:.1f}% progresso)
        - Projetos em risco: **{len(df[df['status'] == 'Crítico'])}**
        - Eficiência orçamentária: **{((orcamento_total - gasto_total)/orcamento_total * 100):.1f}%**
        """)
        
    with col2:
        st.markdown("#### 🎯 Recomendações")
        
        # Identificar departamentos que precisam de atenção
        baixo_progresso = df[df['progresso'] < df['progresso'].mean()]['departamento'].unique()
        alto_risco = df[df['risco'] == 'Alto']['departamento'].unique()
        
        st.markdown("**Ações Prioritárias:**")
        
        if len(baixo_progresso) > 0:
            st.markdown(f"- Acelerar entregas em: **{', '.join(baixo_progresso)}**")
            
        if len(alto_risco) > 0:
            st.markdown(f"- Mitigar riscos em: **{', '.join(alto_risco)}**")
        
        st.markdown("""
        **Melhorias Gerais:**
        - Otimizar alocação de recursos
        - Revisar processos de gestão
        - Implementar early warnings
        - Melhorar comunicação entre times
        """)

def render_executive_summary(df, df_produtos):
    """Renderiza sumário executivo expandido"""
    st.markdown('<div class="section-header">📊 Executive Summary</div>', 
                unsafe_allow_html=True)
    
    # 1. KPIs Principais em Cards Interativos
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>Portfolio Health</h3>
            <h2 style="color: #00cc96">85%</h2>
            <p>↑ 5% vs último mês</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>ROI Médio</h3>
            <h2 style="color: #1f77b4">2.3x</h2>
            <p>↑ 0.2x vs target</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>Time to Market</h3>
            <h2 style="color: #ff7f0e">45 dias</h2>
            <p>↓ 15% vs baseline</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col4:
        st.markdown("""
        <div class="metric-card">
            <h3>Customer Satisfaction</h3>
            <h2 style="color: #2ca02c">8.7</h2>
            <p> 0.5 vs target</p>
        </div>
        """, unsafe_allow_html=True)

    # 2. Visão Geral do Portfolio
    st.subheader("Visão Geral do Portfolio")
    
    # Criar três colunas para métricas detalhadas
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Gráfico de Distribuição de Status
        status_dist = df['status'].value_counts()
        fig_status = px.pie(
            values=status_dist.values,
            names=status_dist.index,
            title="Distribuição de Status",
            hole=0.4
        )
        st.plotly_chart(fig_status)
        
    with col2:
        # Gráfico de Progresso por Departamento
        fig_prog = px.bar(
            df.groupby('departamento')['progresso'].mean().reset_index(),
            x='departamento',
            y='progresso',
            title="Progresso Médio por Departamento"
        )
        st.plotly_chart(fig_prog)
        
    with col3:
        # Gráfico de ROI vs Investimento
        fig_roi = px.scatter(
            df,
            x='orcamento',
            y='roi_esperado',
            size='progresso',
            color='departamento',
            title="ROI vs Investimento"
        )
        st.plotly_chart(fig_roi)

    # 3. Métricas de Negócio
    st.subheader("Métricas de Negócio")
    
    # Criar gauge charts para métricas principais
    fig_gauges = make_subplots(
        rows=1, cols=3,
        specs=[[{'type': 'indicator'}, {'type': 'indicator'}, {'type': 'indicator'}]]
    )
    
    fig_gauges.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=df['nps_score'].mean(),
            title={'text': "NPS Score"},
            gauge={'axis': {'range': [0, 10]},
                  'steps': [
                      {'range': [0, 6], 'color': "lightgray"},
                      {'range': [6, 8], 'color': "gray"},
                      {'range': [8, 10], 'color': "darkgreen"}
                  ],
                  'threshold': {
                      'line': {'color': "red", 'width': 4},
                      'thickness': 0.75,
                      'value': 8.5
                  }}
        ),
        row=1, col=1
    )
    
    fig_gauges.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=df['taxa_conversao'].mean(),
            title={'text': "Taxa de Conversão (%)"},
            gauge={'axis': {'range': [0, 20]},
                  'steps': [
                      {'range': [0, 5], 'color': "lightgray"},
                      {'range': [5, 10], 'color': "gray"},
                      {'range': [10, 20], 'color': "darkgreen"}
                  ]}
        ),
        row=1, col=2
    )
    
    fig_gauges.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=df['churn_rate'].mean(),
            title={'text': "Churn Rate (%)"},
            gauge={'axis': {'range': [0, 10]},
                  'steps': [
                      {'range': [0, 2], 'color': "darkgreen"},
                      {'range': [2, 5], 'color': "gray"},
                      {'range': [5, 10], 'color': "lightgray"}
                  ]}
        ),
        row=1, col=3
    )
    
    fig_gauges.update_layout(height=250)
    st.plotly_chart(fig_gauges)

    # 4. Análise de Tendências
    st.subheader("Anlise de Tendências")
    
    # Criar dados de tendência simulados
    dates = pd.date_range(start='2024-01-01', periods=12, freq='M')
    trends = pd.DataFrame({
        'data': dates,
        'revenue_growth': np.random.normal(10, 2, 12).cumsum(),
        'cost_reduction': np.random.normal(5, 1, 12).cumsum(),
        'customer_satisfaction': np.random.normal(8.5, 0.2, 12)
    })
    
    fig_trends = go.Figure()
    
    fig_trends.add_trace(go.Scatter(
        x=trends['data'],
        y=trends['revenue_growth'],
        name='Crescimento Receita (%)',
        mode='lines+markers'
    ))
    
    fig_trends.add_trace(go.Scatter(
        x=trends['data'],
        y=trends['cost_reduction'],
        name='Redução de Custos (%)',
        mode='lines+markers'
    ))
    
    fig_trends.add_trace(go.Scatter(
        x=trends['data'],
        y=trends['customer_satisfaction'],
        name='Satisfação Cliente',
        mode='lines+markers',
        yaxis='y2'
    ))
    
    fig_trends.update_layout(
        title='Tendências Principais',
        yaxis=dict(title='Percentual (%)'),
        yaxis2=dict(title='Satisfação', overlaying='y', side='right')
    )
    
    st.plotly_chart(fig_trends)

    # 5. Mapa de Calor de Correlações
    st.subheader("Correlações entre Métricas")
    
    correlation_metrics = [
        'progresso', 'roi_esperado', 'satisfacao_cliente',
        'nps_score', 'taxa_conversao', 'churn_rate'
    ]
    
    corr_matrix = df[correlation_metrics].corr()
    
    fig_corr = go.Figure(data=go.Heatmap(
        z=corr_matrix,
        x=correlation_metrics,
        y=correlation_metrics,
        text=np.round(corr_matrix, 2),
        texttemplate='%{text}',
        textfont={"size": 10},
        hoverongaps=False,
        colorscale='RdBu',
        zmin=-1,
        zmax=1
    ))
    
    fig_corr.update_layout(
        title='Correlações entre Métricas Principais',
        xaxis_title="Métricas",
        yaxis_title="Métricas",
        height=500
    )
    
    # Rotacionar labels do eixo x para melhor legibilidade
    fig_corr.update_xaxes(tickangle=45)
    
    st.plotly_chart(fig_corr, use_container_width=True)
    
    # Adicionar insights baseados nas correlações
    st.markdown("### 📊 Insights das Correlações")
    
    # Encontrar correlações mais fortes (positivas e negativas)
    correlations = corr_matrix.unstack()
    correlations = correlations[correlations != 1.0]  # Remover autocorrelaões
    top_correlations = correlations.abs().nlargest(5)
    
    st.markdown("#### Correlações mais significativas:")
    for idx, corr in top_correlations.items():
        metric1, metric2 = idx
        if metric1 != metric2:  # Evitar autocorrelações
            correlation_type = "positiva" if corr > 0 else "negativa"
            st.markdown(f"- **{metric1}** tem correlação {correlation_type} forte ({corr:.2f}) com **{metric2}**")
    
    # Adicionar recomendações baseadas nas correlações
    st.markdown("""
    #### Recomendações baseadas nas correlações:
    1. 📈 Focar em métricas com correlações positivas fortes para efeito multiplicador
    2. 🎯 Monitorar métricas com correlações negativas para balancear trade-offs
    3. 🔄 Usar correlações para prever impactos em cascata de mudanças
    """)

    # Adicionar seção de produtos
    st.subheader("Portfolio de Produtos")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Top produtos por receita
        latest_data = df_produtos[df_produtos['data'] == df_produtos['data'].max()]
        top_products = latest_data.nlargest(3, 'receita')
        
        st.markdown("#### 🏆 Top Produtos por Receita")
        for _, prod in top_products.iterrows():
            st.markdown(f"""
            **{prod['produto']}**  
            Receita: R$ {prod['receita']:,.2f}  
            Crescimento: {prod['crescimento']}%
            """)
            
    with col2:
        # Matriz BCG simplificada
        fig_bcg_simple = criar_matriz_bcg(df_produtos)
        fig_bcg_simple.update_layout(height=300)
        st.plotly_chart(fig_bcg_simple)
        
    with col3:
        # Mtricas agregadas
        st.markdown("#### 📊 Métricas Agregadas")
        st.markdown(f"""
        **Total Usuários Ativos:** {latest_data['usuarios_ativos'].sum():,}  
        **NPS Médio:** {latest_data['nps'].mean():.1f}  
        **Churn Médio:** {latest_data['churn'].mean():.1f}%  
        **LTV/CAC Médio:** {(latest_data['ltv'] / latest_data['cac']).mean():.1f}x
        """)

def render_analise_departamental(df):
    """Renderiza análise detalhada por departamento"""
    st.markdown('<div class="section-header">🏢 Análise Departamental</div>', 
                unsafe_allow_html=True)
    
    # Calcular métricas por departamento
    dept_analysis = df.groupby('departamento').agg({
        'orcamento': 'sum',
        'gasto': 'sum',
        'progresso': 'mean',
        'roi_esperado': 'mean',
        'satisfacao_cliente': 'mean',
        'velocidade_sprint': 'mean',
        'bugs_criticos': 'sum',
        'recursos_alocados': 'sum',
        'nps_score': 'mean',
        'taxa_conversao': 'mean'
    }).reset_index()
    
    # Calcular métricas derivadas
    dept_analysis['eficiencia_custo'] = (dept_analysis['orcamento'] - dept_analysis['gasto']) / dept_analysis['orcamento']
    dept_analysis['produtividade'] = dept_analysis['progresso'] / dept_analysis['recursos_alocados']
    
    # 1. Visão Geral dos Departamentos
    st.subheader("Visão Geral dos Departamentos")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Gráfico de Orçamento vs Gasto
        fig_budget = go.Figure()
        fig_budget.add_trace(go.Bar(
            name='Orçamento',
            x=dept_analysis['departamento'],
            y=dept_analysis['orcamento'],
            marker_color='lightblue'
        ))
        fig_budget.add_trace(go.Bar(
            name='Gasto',
            x=dept_analysis['departamento'],
            y=dept_analysis['gasto'],
            marker_color='darkblue'
        ))
        fig_budget.update_layout(
            title='Orçamento vs Gasto por Departamento',
            barmode='group'
        )
        st.plotly_chart(fig_budget)
    
    with col2:
        # Gráfico de Radar de Métricas
        categories = ['Progresso', 'ROI', 'Satisfação', 'Velocidade', 'NPS']
        fig_radar = go.Figure()
        
        for dept in dept_analysis['departamento']:
            dept_data = dept_analysis[dept_analysis['departamento'] == dept]
            fig_radar.add_trace(go.Scatterpolar(
                r=[
                    dept_data['progresso'].values[0]/100,
                    dept_data['roi_esperado'].values[0]/4,
                    dept_data['satisfacao_cliente'].values[0]/10,
                    dept_data['velocidade_sprint'].values[0]/100,
                    dept_data['nps_score'].values[0]/10
                ],
                theta=categories,
                fill='toself',
                name=dept
            ))
        
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            title='Métricas Principais por Departamento'
        )
        st.plotly_chart(fig_radar)
    
    # 2. Análise de Eficiência
    st.subheader("Análise de Eficiência")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Eficiência de Custo
        fig_eff = px.bar(
            dept_analysis,
            x='departamento',
            y='eficiencia_custo',
            title='Eficiência de Custo',
            color='eficiencia_custo',
            color_continuous_scale='RdYlGn'
        )
        st.plotly_chart(fig_eff)
        
    with col2:
        # Produtividade
        fig_prod = px.bar(
            dept_analysis,
            x='departamento',
            y='produtividade',
            title='Produtividade (Progresso/Recursos)',
            color='produtividade',
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig_prod)
        
    with col3:
        # Qualidade
        fig_qual = px.scatter(
            dept_analysis,
            x='velocidade_sprint',
            y='bugs_criticos',
            size='recursos_alocados',
            color='departamento',
            title='Velocidade vs Bugs',
            hover_data=['satisfacao_cliente']
        )
        st.plotly_chart(fig_qual)
    
    # 3. Métricas de Negócio
    st.subheader("Métricas de Negócio por Departamento")
    
    # Criar um heatmap de métricas de negócio
    business_metrics = [
        'taxa_conversao', 'nps_score', 'satisfacao_cliente',
        'roi_esperado', 'eficiencia_custo'
    ]
    
    fig_heatmap = go.Figure(data=go.Heatmap(
        z=dept_analysis[business_metrics].values,
        x=business_metrics,
        y=dept_analysis['departamento'],
        colorscale='Viridis',
        text=np.round(dept_analysis[business_metrics].values, 2),
        texttemplate='%{text}',
        textfont={"size": 10}
    ))
    
    fig_heatmap.update_layout(
        title='Heatmap de Métricas de Negócio',
        height=400
    )
    
    st.plotly_chart(fig_heatmap)
    
    # 4. Insights e Recomendações
    st.subheader("Insights e Recomendações")
    
    # Identificar top performers
    top_roi = dept_analysis.nlargest(3, 'roi_esperado')['departamento'].tolist()
    top_prod = dept_analysis.nlargest(3, 'produtividade')['departamento'].tolist()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🏆 Top Performers")
        st.markdown(f"""
        **Maior ROI:**
        - {', '.join(top_roi)}
        
        **Maior Produtividade:**
        - {', '.join(top_prod)}
        """)
        
    with col2:
        st.markdown("#### 📈 Oportunidades de Melhoria")
        # Identificar departamentos com métricas abaixo da média
        below_avg = []
        for metric in ['roi_esperado', 'eficiencia_custo', 'produtividade']:
            mean_value = dept_analysis[metric].mean()
            below_avg.extend(
                dept_analysis[dept_analysis[metric] < mean_value]['departamento'].tolist()
            )
        
        # Contar ocorrências para identificar departamentos mais necessitados
        from collections import Counter
        improvement_needs = Counter(below_avg)
        
        for dept, count in improvement_needs.most_common(3):
            st.markdown(f"- **{dept}**: Melhorar {count} métricas principais")
    
    # 5. Análise de Tendências
    st.subheader("Tendências por Departamento")
    
    # Simular dados de tendência
    dates = pd.date_range(start='2024-01-01', periods=6, freq='M')
    trend_data = pd.DataFrame()
    
    for dept in df['departamento'].unique():
        base_progress = np.random.uniform(60, 80)
        progress_trend = base_progress + np.random.normal(5, 2, len(dates)).cumsum()
        
        dept_trend = pd.DataFrame({
            'data': dates,
            'departamento': dept,
            'progresso': progress_trend
        })
        trend_data = pd.concat([trend_data, dept_trend])
    
    fig_trend = px.line(
        trend_data,
        x='data',
        y='progresso',
        color='departamento',
        title='Tendência de Progresso por Departamento'
    )
    
    st.plotly_chart(fig_trend)

# Adicionar dados simulados de produtos
def gerar_dados_produtos():
    """Gera dados simulados para análise de produtos"""
    produtos = {
        "Mensalidade": {
            "receita_mensal": np.random.uniform(800000, 1200000, 12),
            "crescimento": 15,
            "market_share": 35,
            "market_growth": 12,
            "churn": 2.5,
            "cac": 800,
            "ltv": 12000,
            "nps": 8.5,
            "usuarios_ativos": 5000,
            "tickets_suporte": 450,
            "conversion_rate": 12
        },
        "Agende+": {
            "receita_mensal": np.random.uniform(300000, 500000, 12),
            "crescimento": 25,
            "market_share": 20,
            "market_growth": 18,
            "churn": 3.2,
            "cac": 600,
            "ltv": 8000,
            "nps": 8.2,
            "usuarios_ativos": 3000,
            "tickets_suporte": 280,
            "conversion_rate": 15
        },
        "ChatBot": {
            "receita_mensal": np.random.uniform(150000, 250000, 12),
            "crescimento": 45,
            "market_share": 15,
            "market_growth": 30,
            "churn": 4.0,
            "cac": 400,
            "ltv": 5000,
            "nps": 7.8,
            "usuarios_ativos": 2000,
            "tickets_suporte": 150,
            "conversion_rate": 18
        },
        "SignBox": {
            "receita_mensal": np.random.uniform(100000, 180000, 12),
            "crescimento": 35,
            "market_share": 12,
            "market_growth": 25,
            "churn": 3.8,
            "cac": 350,
            "ltv": 4500,
            "nps": 8.0,
            "usuarios_ativos": 1500,
            "tickets_suporte": 120,
            "conversion_rate": 14
        },
        "AmigoPay": {
            "receita_mensal": np.random.uniform(200000, 350000, 12),
            "crescimento": 40,
            "market_share": 18,
            "market_growth": 28,
            "churn": 3.5,
            "cac": 500,
            "ltv": 7000,
            "nps": 7.9,
            "usuarios_ativos": 2500,
            "tickets_suporte": 200,
            "conversion_rate": 16
        },
        "Contabilidade": {
            "receita_mensal": np.random.uniform(400000, 600000, 12),
            "crescimento": 20,
            "market_share": 25,
            "market_growth": 15,
            "churn": 2.8,
            "cac": 900,
            "ltv": 15000,
            "nps": 8.3,
            "usuarios_ativos": 1800,
            "tickets_suporte": 180,
            "conversion_rate": 10
        },
        "AmigoOne": {
            "receita_mensal": np.random.uniform(250000, 400000, 12),
            "crescimento": 30,
            "market_share": 22,
            "market_growth": 20,
            "churn": 3.0,
            "cac": 700,
            "ltv": 10000,
            "nps": 8.1,
            "usuarios_ativos": 2200,
            "tickets_suporte": 220,
            "conversion_rate": 13
        }
    }
    
    # Criar DataFrame com dados mensais
    dates = pd.date_range(start='2024-01-01', periods=12, freq='M')
    df_produtos = pd.DataFrame()
    
    for produto, metricas in produtos.items():
        df_temp = pd.DataFrame({
            'data': dates,
            'produto': produto,
            'receita': metricas['receita_mensal'],
            'crescimento': metricas['crescimento'],
            'market_share': metricas['market_share'],
            'market_growth': metricas['market_growth'],
            'churn': metricas['churn'],
            'cac': metricas['cac'],
            'ltv': metricas['ltv'],
            'nps': metricas['nps'],
            'usuarios_ativos': metricas['usuarios_ativos'],
            'tickets_suporte': metricas['tickets_suporte'],
            'conversion_rate': metricas['conversion_rate']
        })
        df_produtos = pd.concat([df_produtos, df_temp])
    
    return df_produtos.reset_index(drop=True)

def criar_matriz_bcg(df):
    """Cria matriz BCG dos produtos"""
    # Calcular médias por produto
    produto_metrics = df.groupby('produto').agg({
        'market_share': 'mean',
        'market_growth': 'mean',
        'receita': 'mean'
    }).reset_index()
    
    fig = go.Figure()
    
    # Adicionar quadrantes
    fig.add_shape(type="rect",
        x0=0, y0=20,
        x1=25, y1=40,
        line=dict(color="rgba(0,0,0,0.1)"),
        fillcolor="rgba(255,0,0,0.1)",
        name="Pontos de Interrogação"
    )
    
    fig.add_shape(type="rect",
        x0=25, y0=20,
        x1=50, y1=40,
        line=dict(color="rgba(0,0,0,0.1)"),
        fillcolor="rgba(0,255,0,0.1)",
        name="Estrelas"
    )
    
    fig.add_shape(type="rect",
        x0=0, y0=0,
        x1=25, y1=20,
        line=dict(color="rgba(0,0,0,0.1)"),
        fillcolor="rgba(128,128,128,0.1)",
        name="Abacaxis"
    )
    
    fig.add_shape(type="rect",
        x0=25, y0=0,
        x1=50, y1=20,
        line=dict(color="rgba(0,0,0,0.1)"),
        fillcolor="rgba(0,0,255,0.1)",
        name="Vacas Leiteiras"
    )
    
    # Adicionar produtos
    fig.add_trace(go.Scatter(
        x=produto_metrics['market_share'],
        y=produto_metrics['market_growth'],
        mode='markers+text',
        marker=dict(
            size=produto_metrics['receita']/5000,
            sizemode='area',
            sizeref=2.*max(produto_metrics['receita'])/(40.**2),
            sizemin=4
        ),
        text=produto_metrics['produto'],
        textposition="top center"
    ))
    
    fig.update_layout(
        title="Matriz BCG - Portfolio de Produtos",
        xaxis_title="Market Share (%)",
        yaxis_title="Market Growth (%)",
        showlegend=False
    )
    
    return fig

def render_analise_produtos(df):
    """Renderiza análise detalhada de produtos"""
    st.markdown('<div class="section-header">🏷️ Análise de Produtos</div>', 
                unsafe_allow_html=True)
    
    # Processar dados de produtos
    df_produtos = processar_dados_produtos(df)
    
    # Calcular métricas gerais
    ltv_cac_medio = df_produtos['ltv_cac_ratio'].mean()
    
    # 1. KPIs de Produtos
    st.subheader("KPIs de Produtos")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "LTV/CAC Médio",
            f"{ltv_cac_medio:.1f}x",
            f"{(ltv_cac_medio - 3):.1f}x vs target"
        )
        
    with col2:
        churn_medio = df_produtos['churn'].mean()
        st.metric(
            "Churn Médio",
            f"{churn_medio:.1f}%",
            f"{(5 - churn_medio):.1f}% vs target"
        )
        
    with col3:
        nps_medio = df_produtos['nps'].mean()
        st.metric(
            "NPS Médio",
            f"{nps_medio:.1f}",
            f"{(nps_medio - 70):.1f} vs benchmark"
        )
        
    with col4:
        conversao_media = df_produtos['conversao'].mean()
        st.metric(
            "Taxa de Conversão",
            f"{conversao_media:.1f}%",
            f"{(conversao_media - 2.5):.1f}% vs target"
        )
    
    # 2. Análise de Receita
    st.subheader("Análise de Receita")
    
    # Gráfico de receita por produto
    fig_receita = px.bar(
        df_produtos,
        x='produto',
        y='receita',
        color='departamento',
        title='Receita por Produto',
        text=df_produtos['receita'].apply(lambda x: f'R$ {x:,.0f}')
    )
    
    fig_receita.update_traces(textposition='outside')
    st.plotly_chart(fig_receita, use_container_width=True)
    
    # 3. Análise de Crescimento
    st.subheader("Análise de Crescimento")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Crescimento por produto
        fig_crescimento = px.bar(
            df_produtos,
            x='produto',
            y='crescimento',
            color='classificacao_crescimento',
            title='Taxa de Crescimento por Produto',
            text=df_produtos['crescimento'].apply(lambda x: f'{x:.1f}%')
        )
        fig_crescimento.update_traces(textposition='outside')
        st.plotly_chart(fig_crescimento)
        
    with col2:
        # LTV/CAC por produto
        fig_ltv_cac = px.bar(
            df_produtos,
            x='produto',
            y='ltv_cac_ratio',
            color='saude_produto',
            title='LTV/CAC por Produto',
            text=df_produtos['ltv_cac_ratio'].apply(lambda x: f'{x:.1f}x')
        )
        fig_ltv_cac.update_traces(textposition='outside')
        st.plotly_chart(fig_ltv_cac)
    
    # 4. Matriz de Produtos
    st.subheader("Matriz de Produtos")
    
    fig_matriz = px.scatter(
        df_produtos,
        x='ltv_cac_ratio',
        y='crescimento',
        size='receita',
        color='classificacao_receita',
        hover_data=['produto', 'churn', 'nps'],
        title='Matriz de Produtos: LTV/CAC vs Crescimento'
    )
    
    # Adicionar linhas de referência
    fig_matriz.add_hline(y=df_produtos['crescimento'].median(), line_dash="dash")
    fig_matriz.add_vline(x=3, line_dash="dash")  # LTV/CAC target
    
    st.plotly_chart(fig_matriz, use_container_width=True)
    
    # 5. Insights e Recomendações
    st.subheader("Insights e Recomendações")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 💡 Principais Insights")
        
        # Calcular insights
        top_revenue = df_produtos.nlargest(1, 'receita')
        top_growth = df_produtos.nlargest(1, 'crescimento')
        best_retention = df_produtos.nsmallest(1, 'churn')
        
        st.markdown(f"""
        - Produto mais rentável: **{top_revenue['produto'].iloc[0]}** (R$ {top_revenue['receita'].iloc[0]:,.2f})
        - Maior crescimento: **{top_growth['produto'].iloc[0]}** ({top_growth['crescimento'].iloc[0]:.1f}%)
        - Melhor retenção: **{best_retention['produto'].iloc[0]}** (churn {best_retention['churn'].iloc[0]:.1f}%)
        - LTV/CAC médio do portfolio: **{ltv_cac_medio:.1f}x**
        """)
        
    with col2:
        st.markdown("#### 🎯 Recomendações")
        
        # Identificar produtos que precisam de atenção
        baixo_ltv_cac = df_produtos[df_produtos['ltv_cac_ratio'] < 3]['produto'].unique()
        alto_churn = df_produtos[df_produtos['churn'] > df_produtos['churn'].mean()]['produto'].unique()
        
        st.markdown("**Ações Prioritárias:**")
        
        if len(baixo_ltv_cac) > 0:
            st.markdown(f"- Melhorar LTV/CAC em: **{', '.join(baixo_ltv_cac)}**")
            
        if len(alto_churn) > 0:
            st.markdown(f"- Reduzir churn em: **{', '.join(alto_churn)}**")
        
        st.markdown("""
        **Melhorias Gerais:**
        - Otimizar funil de conversão
        - Aumentar retenção de usuários
        - Expandir base de clientes
        - Melhorar experiência do usuário
        """)

def render_performance_financeira(df):
    """Renderiza análise detalhada de performance financeira"""
    st.markdown('<div class="section-header">💰 Performance Financeira</div>', 
                unsafe_allow_html=True)
    
    # 1. KPIs Financeiros
    st.subheader("KPIs Financeiros")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        roi_medio = df['roi_esperado'].mean()
        st.metric(
            "ROI Médio",
            f"{roi_medio:.2f}x",
            f"{(roi_medio - 2):.2f}x vs target"
        )
        
    with col2:
        orcamento_total = df['orcamento'].sum()
        st.metric(
            "Orçamento Total",
            f"R$ {orcamento_total:,.2f}",
            "12% vs ano anterior"
        )
        
    with col3:
        gasto_total = df['gasto'].sum()
        st.metric(
            "Gasto Total",
            f"R$ {gasto_total:,.2f}",
            f"{((gasto_total/orcamento_total - 1) * 100):.1f}% do orçamento"
        )
        
    with col4:
        eficiencia = (orcamento_total - gasto_total) / orcamento_total * 100
        st.metric(
            "Eficiência de Custos",
            f"{eficiencia:.1f}%",
            "5% vs target"
        )
    
    # 2. Análise de ROI e Investimentos
    st.subheader("ROI e Investimentos")
    
    # Seletor de visualização
    roi_view = st.selectbox(
        "Selecione a visualização",
        ["ROI por Departamento", "Investimento vs Retorno", "Distribuição de Gastos"]
    )
    
    if roi_view == "ROI por Departamento":
        fig = px.box(
            df,
            x='departamento',
            y='roi_esperado',
            color='departamento',
            title='Distribuição de ROI por Departamento',
            points='all'
        )
        
    elif roi_view == "Investimento vs Retorno":
        df['retorno_esperado'] = df['orcamento'] * df['roi_esperado']
        fig = px.scatter(
            df,
            x='orcamento',
            y='retorno_esperado',
            size='recursos_alocados',
            color='departamento',
            title='Investimento vs Retorno Esperado',
            hover_data=['nome', 'roi_esperado']
        )
        
    else:  # Distribuição de Gastos
        fig = px.sunburst(
            df,
            path=['departamento', 'status'],
            values='gasto',
            title='Distribuição Hierárquica de Gastos'
        )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 3. Análise de Custos
    st.subheader("Análise de Custos")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Orçamento vs Gasto por Departamento
        df_custos = df.groupby('departamento').agg({
            'orcamento': 'sum',
            'gasto': 'sum'
        }).reset_index()
        
        df_custos_melted = pd.melt(
            df_custos,
            id_vars=['departamento'],
            value_vars=['orcamento', 'gasto'],
            var_name='tipo',
            value_name='valor'
        )
        
        fig_custos = px.bar(
            df_custos_melted,
            x='departamento',
            y='valor',
            color='tipo',
            title='Orçamento vs Gasto por Departamento',
            barmode='group'
        )
        st.plotly_chart(fig_custos)
        
    with col2:
        # Eficiência de Custos
        df_custos['eficiencia'] = (df_custos['orcamento'] - df_custos['gasto']) / df_custos['orcamento'] * 100
        
        fig_eficiencia = px.bar(
            df_custos,
            x='departamento',
            y='eficiencia',
            title='Eficiência de Custos por Departamento (%)',
            color='eficiencia',
            color_continuous_scale='RdYlGn'
        )
        st.plotly_chart(fig_eficiencia)
    
    # 4. Análise Temporal
    st.subheader("Análise Temporal")
    
    # Simular dados temporais
    dates = pd.date_range(start='2024-01-01', periods=12, freq='M')
    temporal_data = pd.DataFrame()
    
    for dept in df['departamento'].unique():
        base_cost = df[df['departamento'] == dept]['gasto'].mean()
        cost_trend = base_cost + np.random.normal(0, base_cost * 0.1, len(dates)).cumsum()
        roi_trend = np.random.normal(2.5, 0.2, len(dates)).cumsum()
        
        dept_data = pd.DataFrame({
            'data': dates,
            'departamento': dept,
            'gasto': cost_trend,
            'roi': roi_trend
        })
        temporal_data = pd.concat([temporal_data, dept_data])
    
    # Seletor de métrica temporal
    temporal_metric = st.selectbox(
        "Selecione a métrica para análise temporal",
        ["Gastos", "ROI"]
    )
    
    if temporal_metric == "Gastos":
        y_col = 'gasto'
        title = 'Evolução de Gastos por Departamento'
    else:
        y_col = 'roi'
        title = 'Evolução de ROI por Departamento'
    
    fig_temporal = px.line(
        temporal_data,
        x='data',
        y=y_col,
        color='departamento',
        title=title
    )
    
    # Adicionar média móvel
    for dept in temporal_data['departamento'].unique():
        dept_data = temporal_data[temporal_data['departamento'] == dept]
        fig_temporal.add_scatter(
            x=dept_data['data'],
            y=dept_data[y_col].rolling(window=3).mean(),
            name=f'{dept} (MM-3)',
            line=dict(dash='dash'),
            showlegend=True
        )
    
    st.plotly_chart(fig_temporal, use_container_width=True)
    
    # 5. Análise de Correlações Financeiras
    st.subheader("Correlações Financeiras")
    
    # Selecionar métricas financeiras relevantes
    financial_metrics = ['orcamento', 'gasto', 'roi_esperado', 'recursos_alocados', 
                        'velocidade_sprint', 'satisfacao_cliente']
    
    corr_matrix = df[financial_metrics].corr()
    
    fig_corr = go.Figure(data=go.Heatmap(
        z=corr_matrix,
        x=financial_metrics,
        y=financial_metrics,
        text=np.round(corr_matrix, 2),
        texttemplate='%{text}',
        textfont={"size": 10},
        hoverongaps=False,
        colorscale='RdBu',
        zmin=-1,
        zmax=1
    ))
    
    fig_corr.update_layout(
        title='Correlações entre Métricas Financeiras',
        height=500
    )
    
    st.plotly_chart(fig_corr, use_container_width=True)
    
    # 6. Insights e Recomendações
    st.subheader("Insights e Recomendações")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 💡 Principais Insights")
        
        # Calcular insights baseados nos dados
        top_roi_dept = df.groupby('departamento')['roi_esperado'].mean().nlargest(1)
        mais_eficiente = df_custos.nlargest(1, 'eficiencia')
        
        st.markdown(f"""
        - Melhor ROI: **{top_roi_dept.index[0]}** ({top_roi_dept.values[0]:.2f}x)
        - Maior eficiência: **{mais_eficiente['departamento'].iloc[0]}** ({mais_eficiente['eficiencia'].iloc[0]:.1f}%)
        - Eficiência média: **{df_custos['eficiencia'].mean():.1f}%**
        - Gasto total: **R$ {gasto_total:,.2f}**
        """)
        
    with col2:
        st.markdown("#### 🎯 Recomendações")
        
        # Identificar departamentos que precisam de atenço
        baixa_eficiencia = df_custos[df_custos['eficiencia'] < df_custos['eficiencia'].mean()]
        
        st.markdown("**Ações Prioritárias:**")
        for _, dept in baixa_eficiencia.iterrows():
            st.markdown(f"""
            - **{dept['departamento']}**:
              - Eficiência atual: {dept['eficiencia']:.1f}%
              - Meta: {df_custos['eficiencia'].mean():.1f}%
              - Redução necessária: R$ {(dept['gasto'] - dept['orcamento']) * (1 - df_custos['eficiencia'].mean()/100):,.2f})
            """)
        
        st.markdown("""
        **Estratégias Gerais:**
        - Otimizar alocação de recursos
        - Implementar controles de custos
        - Revisar processos ineficientes
        - Aumentar automação
        """)

def render_riscos_oportunidades(df):
    """Renderiza análise de riscos e oportunidades"""
    st.markdown('<div class="section-header">⚠️ Riscos e Oportunidades</div>', 
                unsafe_allow_html=True)
    
    # 1. Matriz de Riscos
    st.subheader("Matriz de Riscos")
    
    # Calcular score de risco
    df['risco_score'] = (
        df['bugs_criticos'] * 0.4 +
        (100 - df['progresso']) * 0.3 +
        (10 - df['satisfacao_cliente']) * 0.3
    )
    
    # Calcular score de impacto
    df['impacto_score'] = (
        df['roi_esperado'] * 0.4 +
        df['recursos_alocados'] * 0.3 +
        df['velocidade_sprint'] * 0.3
    )
    
    # Criar matriz de riscos interativa
    risk_view = st.selectbox(
        "Selecione a visualização da matriz",
        ["Riscos vs Impacto", "Riscos vs ROI", "Riscos vs Recursos"]
    )
    
    if risk_view == "Riscos vs Impacto":
        fig_risk = px.scatter(
            df,
            x='risco_score',
            y='impacto_score',
            color='departamento',
            size='orcamento',
            hover_data=['nome', 'progresso', 'bugs_criticos'],
            title='Matriz de Riscos vs Impacto'
        )
    elif risk_view == "Riscos vs ROI":
        fig_risk = px.scatter(
            df,
            x='risco_score',
            y='roi_esperado',
            color='departamento',
            size='orcamento',
            hover_data=['nome', 'progresso', 'bugs_criticos'],
            title='Matriz de Riscos vs ROI'
        )
    else:
        fig_risk = px.scatter(
            df,
            x='risco_score',
            y='recursos_alocados',
            color='departamento',
            size='orcamento',
            hover_data=['nome', 'progresso', 'bugs_criticos'],
            title='Matriz de Riscos vs Recursos'
        )
    
    # Adicionar linhas de referência
    fig_risk.add_hline(y=df['impacto_score'].median(), line_dash="dash", line_color="gray")
    fig_risk.add_vline(x=df['risco_score'].median(), line_dash="dash", line_color="gray")
    
    st.plotly_chart(fig_risk, use_container_width=True)
    
    # 2. KPIs de Risco
    st.subheader("KPIs de Risco")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        alto_risco = len(df[df['risco'] == 'Alto'])
        
        st.metric(
            "Projetos de Alto Risco",
            f"{alto_risco}",
            f"{(alto_risco/len(df)*100):.1f}% do total"  # Corrigido a formatação
        )
        
    with col2:
        risco_medio = df['risco_score'].mean()
        st.metric(
            "Risco Médio",
            f"{risco_medio:.1f}",
            "↓ 2.3 vs último mês"
        )
        
    with col3:
        bugs_total = df['bugs_criticos'].sum()
        st.metric(
            "Total Bugs Críticos",
            f"{bugs_total}",
            "↓ 15% vs média"
        )
        
    with col4:
        atraso_medio = 100 - df['progresso'].mean()
        st.metric(
            "Atraso Médio",
            f"{atraso_medio:.1f}%",
            "↓ 5% vs target"
        )
    
    # 3. Análise de Tendências
    st.subheader("Tendências de Risco")
    
    # Simular dados de tendência
    dates = pd.date_range(start='2024-01-01', periods=12, freq='M')
    trend_data = pd.DataFrame()
    
    for dept in df['departamento'].unique():
        base_risk = df[df['departamento'] == dept]['risco_score'].mean()
        risk_trend = base_risk + np.random.normal(0, base_risk * 0.1, len(dates)).cumsum()
        
        dept_trend = pd.DataFrame({
            'data': dates,
            'departamento': dept,
            'risco': risk_trend
        })
        trend_data = pd.concat([trend_data, dept_trend])
    
    fig_trend = px.line(
        trend_data,
        x='data',
        y='risco',
        color='departamento',
        title='Evolução do Risco por Departamento'
    )
    
    # Adicionar média móvel
    for dept in trend_data['departamento'].unique():
        dept_data = trend_data[trend_data['departamento'] == dept]
        fig_trend.add_scatter(
            x=dept_data['data'],
            y=dept_data['risco'].rolling(window=3).mean(),
            name=f'{dept} (MM-3)',
            line=dict(dash='dash'),
            showlegend=True
        )
    
    st.plotly_chart(fig_trend, use_container_width=True)
    
    # 4. Oportunidades
    st.subheader("Análise de Oportunidades")
    
    # Calcular score de oportunidade
    df['oportunidade_score'] = (
        df['roi_esperado'] * 0.4 +
        df['satisfacao_cliente'] * 0.3 +
        df['velocidade_sprint'] * 0.3)
    
    # Criar visualização de oportunidades
    opp_view = st.selectbox(
        "Selecione a visualizaç��o de oportunidades",
        ["Top Oportunidades", "Oportunidades por Departamento", "Matriz de Oportunidades"]
    )
    
    if opp_view == "Top Oportunidades":
        top_opp = df.nlargest(10, 'oportunidade_score')
        fig_opp = px.bar(
            top_opp,
            x='nome',
            y='oportunidade_score',
            color='departamento',
            title='Top 10 Oportunidades'
        )
        
    elif opp_view == "Oportunidades por Departamento":
        fig_opp = px.box(
            df,
            x='departamento',
            y='oportunidade_score',
            color='departamento',
            title='Distribuição de Oportunidades por Departamento'
        )
        
    else:
        fig_opp = px.scatter(
            df,
            x='risco_score',
            y='oportunidade_score',
            color='departamento',
            size='orcamento',
            hover_data=['nome', 'roi_esperado', 'satisfacao_cliente'],
            title='Matriz de Riscos vs Oportunidades'
        )
    
    st.plotly_chart(fig_opp, use_container_width=True)

def render_metricas_tecnicas(df):
    """Renderiza análise detalhada de métricas técnicas"""
    
    # Adicionar métricas técnicas simuladas ao DataFrame
    df = df.copy()  # Criar uma cópia para não modificar o original
    df['code_coverage'] = np.random.uniform(70, 95, size=len(df))
    df['tech_debt'] = np.random.uniform(5, 25, size=len(df))
    df['bugs_por_sprint'] = df['bugs_criticos'] / df['velocidade_sprint']
    df['eficiencia_tecnica'] = (df['velocidade_sprint'] * (100 - df['bugs_criticos'])) / 100
    
    st.markdown('<div class="section-header">🔧 Métricas Técnicas</div>', 
                unsafe_allow_html=True)
    
    # 1. KPIs Técnicos
    st.subheader("KPIs Técnicos")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        velocidade_media = df['velocidade_sprint'].mean()
        st.metric(
            "Velocidade Média",
            f"{velocidade_media:.1f}",
            f"{(velocidade_media/80 - 1)*100:.1f}% vs target"
        )
        
    with col2:
        coverage_medio = df['code_coverage'].mean()
        st.metric(
            "Code Coverage",
            f"{coverage_medio:.1f}%",
            f"{(coverage_medio - 80):.1f}% vs target"
        )
        
    with col3:
        tech_debt_medio = df['tech_debt'].mean()
        st.metric(
            "Tech Debt",
            f"{tech_debt_medio:.1f}d",
            f"{(15 - tech_debt_medio):.1f}d vs limite"
        )
        
    with col4:
        eficiencia_media = df['eficiencia_tecnica'].mean()
        st.metric(
            "Eficiência Técnica",
            f"{eficiencia_media:.1f}%",
            f"{(eficiencia_media - 85):.1f}% vs target"
        )
    
    # Calcular métricas por departamento
    dept_metrics = df.groupby('departamento').agg({
        'velocidade_sprint': 'mean',
        'bugs_criticos': 'sum',
        'code_coverage': 'mean',
        'tech_debt': 'mean',
        'eficiencia_tecnica': 'mean'
    }).round(2)

    # 2. Análise Técnica
    st.subheader("Análise Técnica")
    
    # Seletor de visualização
    tech_view = st.selectbox(
        "Selecione a visualização",
        ["Métricas por Departamento", "Correlações Técnicas", "Qualidade do Código"]
    )
    
    if tech_view == "Métricas por Departamento":
        fig = go.Figure()
        
        # Adicionar barras para cada métrica
        fig.add_trace(go.Bar(
            name='Velocidade',
            x=dept_metrics.index,
            y=dept_metrics['velocidade_sprint'],
            offsetgroup=0
        ))
        
        fig.add_trace(go.Bar(
            name='Code Coverage',
            x=dept_metrics.index,
            y=dept_metrics['code_coverage'],
            offsetgroup=1
        ))
        
        fig.add_trace(go.Bar(
            name='Tech Debt',
            x=dept_metrics.index,
            y=dept_metrics['tech_debt'],
            offsetgroup=2
        ))
        
        fig.update_layout(
            title='Métricas Técnicas por Departamento',
            barmode='group'
        )
        
    elif tech_view == "Correlações Técnicas":
        # Matriz de correlação
        corr_metrics = ['velocidade_sprint', 'bugs_criticos', 'code_coverage', 'tech_debt', 'eficiencia_tecnica']
        corr_matrix = df[corr_metrics].corr()
        
        fig = px.imshow(
            corr_matrix,
            text=np.round(corr_matrix, 2),
            aspect='auto',
            title='Correlações entre Métricas Técnicas'
        )
        
    else:  # Qualidade do Código
        fig = px.scatter(
            df,
            x='code_coverage',
            y='tech_debt',
            color='departamento',
            size='bugs_criticos',
            hover_data=['nome'],
            title='Qualidade do Código: Coverage vs Tech Debt'
        )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 3. Insights e Recomendações
    st.subheader("Insights e Recomendações")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 💡 Principais Insights")
        
        # Encontrar os melhores departamentos
        best_velocity = dept_metrics['velocidade_sprint'].idxmax()
        best_coverage = dept_metrics['code_coverage'].idxmax()
        lowest_debt = dept_metrics['tech_debt'].idxmin()
        
        st.markdown(f"""
        - Maior velocidade: **{best_velocity}** ({dept_metrics.loc[best_velocity, 'velocidade_sprint']:.1f})
        - Melhor coverage: **{best_coverage}** ({dept_metrics.loc[best_coverage, 'code_coverage']:.1f}%)
        - Menor tech debt: **{lowest_debt}** ({dept_metrics.loc[lowest_debt, 'tech_debt']:.1f}d)
        - Coverage médio: **{df['code_coverage'].mean():.1f}%**
        - Tech debt médio: **{df['tech_debt'].mean():.1f}d**
        """)
        
    with col2:
        st.markdown("#### 🎯 Recomendações")
        
        # Identificar áreas que precisam de atenção
        baixa_coverage = df[df['code_coverage'] < 80]['departamento'].unique()
        alto_debt = df[df['tech_debt'] > 15]['departamento'].unique()
        baixa_eficiencia = df[df['eficiencia_tecnica'] < df['eficiencia_tecnica'].mean()]['departamento'].unique()
        
        st.markdown("**Ações Prioritárias:**")
        
        if len(baixa_coverage) > 0:
            st.markdown(f"- Aumentar code coverage em: **{', '.join(baixa_coverage)}**")
            
        if len(alto_debt) > 0:
            st.markdown(f"- Reduzir tech debt em: **{', '.join(alto_debt)}**")
            
        if len(baixa_eficiencia) > 0:
            st.markdown(f"- Melhorar eficiência em: **{', '.join(baixa_eficiencia)}**")
        
        st.markdown("""
        **Melhorias Gerais:**
        - Implementar code reviews mais rigorosos
        - Aumentar cobertura de testes automatizados
        - Estabelecer metas de qualidade por sprint
        - Realizar refatoração contínua
        - Investir em automação e CI/CD
        """)

def render_bcg_matrix(df):
    """Renderiza uma matriz BCG interativa e profissional"""
    st.subheader("Matriz BCG - Análise de Portfolio")
    
    # Calcular métricas para a matriz BCG
    df_bcg = df.copy()
    df_bcg['crescimento_mercado'] = df_bcg['roi_esperado'] * 100  # Simulando crescimento
    df_bcg['participacao_relativa'] = df_bcg['recursos_alocados'] / df_bcg['recursos_alocados'].max()
    
    # Calcular tamanhos das bolhas baseado no orçamento
    df_bcg['tamanho_bolha'] = df_bcg['orcamento'] / df_bcg['orcamento'].max() * 100
    
    # Criar matriz BCG interativa
    fig = px.scatter(
        df_bcg,
        x='participacao_relativa',
        y='crescimento_mercado',
        size='tamanho_bolha',
        color='departamento',
        text='nome',
        hover_data={
            'participacao_relativa': ':.2f',
            'crescimento_mercado': ':.1f%',
            'roi_esperado': ':.2f',
            'orcamento': ':,.2f',
            'recursos_alocados': True,
            'nome': False
        },
        height=700  # Aumentar tamanho
    )
    
    # Personalizar layout
    fig.update_layout(
        title={
            'text': 'Matriz BCG - Análise Estratégica de Portfolio',
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'size': 24}
        },
        xaxis_title='Participação Relativa de Mercado',
        yaxis_title='Taxa de Crescimento do Mercado (%)',
        showlegend=True,
        legend_title_text='Departamentos',
        template='plotly_white'
    )
    
    # Adicionar linhas de referência
    fig.add_hline(
        y=df_bcg['crescimento_mercado'].median(),
        line_dash='dash',
        line_color='gray',
        annotation_text='Crescimento Médio'
    )
    
    fig.add_vline(
        x=df_bcg['participacao_relativa'].median(),
        line_dash='dash',
        line_color='gray',
        annotation_text='Participação Média'
    )
    
    # Adicionar anotações para quadrantes
    quadrant_annotations = [
        dict(x=0.25, y=75, text="ESTRELA", showarrow=False, font=dict(size=14, color='gold')),
        dict(x=0.25, y=25, text="VACA LEITEIRA", showarrow=False, font=dict(size=14, color='green')),
        dict(x=0.75, y=75, text="PONTO DE INTERROGAÇÃO", showarrow=False, font=dict(size=14, color='red')),
        dict(x=0.75, y=25, text="ABACAXI", showarrow=False, font=dict(size=14, color='gray'))
    ]
    
    fig.update_layout(annotations=quadrant_annotations)
    
    # Personalizar aparência
    fig.update_traces(
        marker=dict(
            line=dict(width=1, color='DarkSlateGrey'),
            opacity=0.7
        ),
        textposition='top center',
        textfont=dict(size=10)
    )
    
    # Adicionar interatividade
    fig.update_layout(
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Arial"
        )
    )
    
    # Exibir gráfico
    st.plotly_chart(fig, use_container_width=True)
    
    # Adicionar insights da matriz
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🎯 Insights Estratégicos")
        
        # Identificar projetos em cada quadrante
        estrelas = df_bcg[
            (df_bcg['crescimento_mercado'] > df_bcg['crescimento_mercado'].median()) &
            (df_bcg['participacao_relativa'] > df_bcg['participacao_relativa'].median())
        ]['nome'].tolist()
        
        vacas = df_bcg[
            (df_bcg['crescimento_mercado'] <= df_bcg['crescimento_mercado'].median()) &
            (df_bcg['participacao_relativa'] > df_bcg['participacao_relativa'].median())
        ]['nome'].tolist()
        
        st.markdown(f"""
        **Estrelas** ({len(estrelas)}):
        - {', '.join(estrelas[:3])}
        
        **Vacas Leiteiras** ({len(vacas)}):
        - {', '.join(vacas[:3])}
        """)
        
    with col2:
        st.markdown("#### 📈 Recomendações")
        st.markdown("""
        **Estratégias Sugeridas:**
        - 🌟 **Estrelas**: Investir para manter liderança
        - 🐮 **Vacas Leiteiras**: Maximizar geração de caixa
        - ❓ **Pontos de Interrogação**: Avaliar potencial de crescimento
        - 🍍 **Abacaxis**: Considerar desinvestimento
        """)

def aplicar_filtros(df):
    """Sidebar profissional com filtros e UI/UX aprimorada"""
    
    # Logo no topo do sidebar
    st.sidebar.markdown("""
        <div class="sidebar-logo">
            <img src="logoamigo.png" alt="Logo">
        </div>
    """, unsafe_allow_html=True)
    
    # Container principal dos filtros
    with st.sidebar:
        # Departamentos
        st.markdown("""
            <div class="filter-section">
                <div class="filter-header">
                    <i class="fas fa-building"></i> Departamentos
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        departamentos = st.multiselect(
            label="Selecione os departamentos",
            options=df['departamento'].unique(),
            default=df['departamento'].unique(),
            key="dept_select"
        )

        # Status
        st.markdown("""
            <div class="filter-section">
                <div class="filter-header">
                    <i class="fas fa-chart-pie"></i> Status
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        status = st.multiselect(
            label="Selecione os status",
            options=df['status'].unique(),
            default=df['status'].unique(),
            key="status_select"
        )

        # Prioridade
        st.markdown("""
            <div class="filter-section">
                <div class="filter-header">
                    <i class="fas fa-flag"></i> Prioridade
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        prioridades = st.multiselect(
            label="Selecione as prioridades",
            options=df['prioridade'].unique(),
            default=df['prioridade'].unique(),
            key="priority_select"
        )

        # Aplicar filtros
        df_filtered = df.copy()
        if departamentos:
            df_filtered = df_filtered[df_filtered['departamento'].isin(departamentos)]
        if status:
            df_filtered = df_filtered[df_filtered['status'].isin(status)]
        if prioridades:
            df_filtered = df_filtered[df_filtered['prioridade'].isin(prioridades)]

        # Informações dos filtros
        total_filtrado = len(df_filtered)
        total_original = len(df)
        
        if total_filtrado < total_original:
            st.markdown(f"""
                <div class="filter-info">
                    <div style="margin-bottom: 0.5rem;">
                        <strong>Resumo dos Filtros</strong>
                    </div>
                    <div>
                        Exibindo {total_filtrado} de {total_original} registros
                        <div class="status-badge">
                            {(total_filtrado/total_original)*100:.1f}% do total
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

        # Footer
        st.markdown(f"""
            <div class="sidebar-footer">
                <div>Última atualização: {datetime.now().strftime('%d/%m/%Y %H:%M')}</div>
                <div class="status-badge" style="margin-top: 0.5rem;">
                    Sistema Online
                </div>
            </div>
        """, unsafe_allow_html=True)

    return df_filtered

def processar_dados_produtos(df):
    """Processa e prepara os dados para a análise de produtos"""
    
    # Criar DataFrame de produtos com dados simulados
    df_produtos = pd.DataFrame({
        'produto': [
            'Produto A', 'Produto B', 'Produto C', 
            'Produto D', 'Produto E', 'Produto F'
        ],
        'receita': np.random.uniform(100000, 1000000, 6),
        'crescimento': np.random.uniform(5, 30, 6),
        'churn': np.random.uniform(1, 10, 6),
        'ltv': np.random.uniform(1000, 5000, 6),
        'cac': np.random.uniform(100, 1000, 6),
        'nps': np.random.uniform(30, 90, 6),
        'usuarios_ativos': np.random.randint(1000, 10000, 6),
        'conversao': np.random.uniform(1, 5, 6),
        'departamento': np.random.choice(df['departamento'].unique(), 6),
        'status': np.random.choice(['Ativo', 'Beta', 'Desenvolvimento'], 6),
        'data_lancamento': pd.date_range(start='2023-01-01', periods=6, freq='M')
    })
    
    # Calcular métricas derivadas
    df_produtos['ltv_cac_ratio'] = df_produtos['ltv'] / df_produtos['cac']
    df_produtos['receita_por_usuario'] = df_produtos['receita'] / df_produtos['usuarios_ativos']
    df_produtos['meses_desde_lancamento'] = (pd.Timestamp.now() - df_produtos['data_lancamento']).dt.days / 30
    df_produtos['receita_mensal'] = df_produtos['receita'] / df_produtos['meses_desde_lancamento']
    
    # Normalizar algumas métricas para visualização
    df_produtos['tamanho_mercado'] = np.random.uniform(0.5, 1.5, 6)  # Para matriz BCG
    df_produtos['participacao_mercado'] = np.random.uniform(0.1, 0.8, 6)  # Para matriz BCG
    
    # Adicionar metas e benchmarks
    df_produtos['meta_receita'] = df_produtos['receita'] * 1.2
    df_produtos['benchmark_nps'] = 70
    df_produtos['benchmark_churn'] = 5
    df_produtos['benchmark_ltv_cac'] = 3
    
    # Adicionar tendências (últimos 6 meses)
    for produto in df_produtos['produto']:
        # Simular tendência de receita
        base_receita = df_produtos.loc[df_produtos['produto'] == produto, 'receita'].values[0]
        tendencia = np.random.uniform(0.8, 1.2, 6) * base_receita
        df_produtos.loc[df_produtos['produto'] == produto, 'tendencia_receita'] = tendencia.mean()
        
        # Simular tendência de usuários
        base_usuarios = df_produtos.loc[df_produtos['produto'] == produto, 'usuarios_ativos'].values[0]
        tendencia = np.random.uniform(0.9, 1.1, 6) * base_usuarios
        df_produtos.loc[df_produtos['produto'] == produto, 'tendencia_usuarios'] = tendencia.mean()
    
    # Adicionar classificações
    df_produtos['classificacao_receita'] = pd.qcut(df_produtos['receita'], q=4, labels=['D', 'C', 'B', 'A'])
    df_produtos['classificacao_crescimento'] = pd.qcut(df_produtos['crescimento'], q=3, labels=['Baixo', 'Médio', 'Alto'])
    df_produtos['saude_produto'] = np.where(
        (df_produtos['ltv_cac_ratio'] >= 3) & (df_produtos['churn'] < 5),
        'Saudável',
        'Precisa Atenção'
    )
    
    return df_produtos

def render_visualizacoes_interativas(df):
    """Renderiza visualizações interativas dos dados"""
    st.markdown('<div class="section-header">📊 Visualizações Interativas</div>', 
                unsafe_allow_html=True)
    
    # Seletor de visualização
    viz_type = st.selectbox(
        "Selecione o tipo de visualização",
        ["Métricas por Departamento", "Análise Temporal", "Correlações", "Distribuições"]
    )
    
    if viz_type == "Métricas por Departamento":
        # Seletor de métrica
        metric = st.selectbox(
            "Selecione a métrica",
            ["progresso", "roi_esperado", "velocidade_sprint", "satisfacao_cliente"]
        )
        
        # Criar gráfico de barras
        fig = px.bar(
            df,
            x='departamento',
            y=metric,
            color='departamento',
            title=f'{metric.replace("_", " ").title()} por Departamento',
            color_discrete_sequence=['#2C3E50', '#34495E', '#7F8C8D', '#95A5A6', '#BDC3C7']
        )
        
        # Adicionar média como linha
        fig.add_hline(
            y=df[metric].mean(),
            line_dash="dash",
            annotation_text=f"Média: {df[metric].mean():.2f}"
        )
        
    elif viz_type == "Análise Temporal":
        # Simular dados temporais
        dates = pd.date_range(start='2024-01-01', periods=12, freq='M')
        temporal_data = pd.DataFrame()
        
        for dept in df['departamento'].unique():
            base_progress = df[df['departamento'] == dept]['progresso'].mean()
            progress_trend = base_progress + np.random.normal(0, 5, len(dates)).cumsum()
            
            dept_data = pd.DataFrame({
                'data': dates,
                'departamento': dept,
                'progresso': progress_trend
            })
            temporal_data = pd.concat([temporal_data, dept_data])
        
        fig = px.line(
            temporal_data,
            x='data',
            y='progresso',
            color='departamento',
            title='Evolução do Progresso por Departamento'
        )
        
    elif viz_type == "Correlações":
        # Selecionar métricas para correlação
        metrics = ['progresso', 'roi_esperado', 'velocidade_sprint', 
                  'satisfacao_cliente', 'bugs_criticos']
        
        corr_matrix = df[metrics].corr()
        
        fig = px.imshow(
            corr_matrix,
            text=np.round(corr_matrix, 2),
            aspect='auto',
            title='Matriz de Correlações',
            color_continuous_scale=[
                [0, '#2C3E50'],
                [0.5, '#7F8C8D'],
                [1, '#BDC3C7']
            ]
        )
        
    else:  # Distribuiões
        # Seletor de métrica
        metric = st.selectbox(
            "Selecione a métrica para análise de distribuição",
            ["progresso", "roi_esperado", "velocidade_sprint", "satisfacao_cliente"]
        )
        
        fig = px.histogram(
            df,
            x=metric,
            color='departamento',
            title=f'Distribuição de {metric.replace("_", " ").title()}',
            marginal="box"
        )
    
    # Exibir gráfico
    st.plotly_chart(fig, use_container_width=True)
    
    # Adicionar insights
    st.subheader("Insights")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📈 Destaques")
        # Calcular e exibir métricas relevantes
        top_dept = df.groupby('departamento')['progresso'].mean().nlargest(1)
        best_roi = df.groupby('departamento')['roi_esperado'].mean().nlargest(1)
        
        st.markdown(f"""
        - Melhor progresso: **{top_dept.index[0]}** ({top_dept.values[0]:.1f}%)
        - Melhor ROI: **{best_roi.index[0]}** ({best_roi.values[0]:.2f}x)
        - Satisfação média: **{df['satisfacao_cliente'].mean():.1f}/10**
        """)
        
    with col2:
        st.markdown("#### 🎯 Oportunidades")
        # Identificar áreas de melhoria
        low_progress = df.groupby('departamento')['progresso'].mean().nsmallest(1)
        high_bugs = df.groupby('departamento')['bugs_criticos'].sum().nlargest(1)
        
        st.markdown(f"""
        - Foco em progresso: **{low_progress.index[0]}** ({low_progress.values[0]:.1f})
        - Reduzir bugs em: **{high_bugs.index[0]}** ({high_bugs.values[0]:.0f} bugs)
        - Velocidade média: **{df['velocidade_sprint'].mean():.1f}**
        """)

def render_analise_operacional(df):
    """Renderiza análise operacional detalhada"""
    st.markdown('<div class="section-header">🔄 Análise Operacional</div>', 
                unsafe_allow_html=True)
    
    # Calcular métricas operacionais
    df['eficiencia_tecnica'] = df['progresso'] / df['recursos_alocados']
    df['taxa_bugs'] = df['bugs_criticos'] / df['recursos_alocados']
    
    # 1. KPIs Operacionais
    st.subheader("KPIs Operacionais")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        velocidade_media = df['velocidade_sprint'].mean()
        st.metric(
            "Velocidade Média",
            f"{velocidade_media:.1f}",
            f"{(velocidade_media - 80):.1f} vs target"
        )
        
    with col2:
        eficiencia_media = df['eficiencia_tecnica'].mean()
        st.metric(
            "Eficiência Técnica",
            f"{eficiencia_media:.2f}",
            f"{(eficiencia_media - 0.8):.2f} vs baseline"
        )
        
    with col3:
        bugs_total = df['bugs_criticos'].sum()
        st.metric(
            "Bugs Críticos",
            f"{bugs_total}",
            f"{df['bugs_criticos'].mean():.1f} média/projeto"
        )
        
    with col4:
        recursos_total = df['recursos_alocados'].sum()
        st.metric(
            "Total Recursos",
            f"{recursos_total}",
            f"{df['recursos_alocados'].mean():.1f} média/projeto"
        )
    
    # 2. Análise de Eficiência
    st.subheader("Análise de Eficiência")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Eficiência por departamento
        fig_eficiencia = px.bar(
            df.groupby('departamento')['eficiencia_tecnica'].mean().reset_index(),
            x='departamento',
            y='eficiencia_tecnica',
            title='Eficiência Técnica por Departamento',
            color='eficiencia_tecnica',
            color_continuous_scale='RdYlGn'
        )
        st.plotly_chart(fig_eficiencia)
        
    with col2:
        # Relação entre recursos e bugs
        fig_bugs = px.scatter(
            df,
            x='recursos_alocados',
            y='bugs_criticos',
            color='departamento',
            size='progresso',
            title='Recursos vs Bugs Críticos',
            hover_data=['nome']
        )
        st.plotly_chart(fig_bugs)
    
    # 3. Insights Operacionais
    st.subheader("Insights Operacionais")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 💡 Principais Insights")
        
        # Calcular insights
        baixa_eficiencia = df[df['eficiencia_tecnica'] < df['eficiencia_tecnica'].mean()]['departamento'].unique()
        alto_bugs = df[df['taxa_bugs'] > df['taxa_bugs'].mean()]['departamento'].unique()
        
        st.markdown(f"""
        - Eficiência média: **{eficiencia_media:.2f}**
        - Velocidade média: **{velocidade_media:.1f}**
        - Total de bugs críticos: **{bugs_total}**
        - Recursos alocados: **{recursos_total}**
        """)
        
    with col2:
        st.markdown("#### 🎯 Recomendações")
        
        st.markdown("**Ações Prioritárias:**")
        
        if len(baixa_eficiencia) > 0:
            st.markdown(f"- Melhorar eficiência em: **{', '.join(baixa_eficiencia)}**")
            
        if len(alto_bugs) > 0:
            st.markdown(f"- Reduzir bugs em: **{', '.join(alto_bugs)}**")

def render_analise_financeira(df):
    """Renderiza análise financeira detalhada"""
    st.markdown('<div class="section-header">💰 Análise Financeira</div>', 
                unsafe_allow_html=True)
    
    # Calcular métricas financeiras
    orcamento_total = df['orcamento'].sum()
    gasto_total = df['gasto'].sum()
    roi_medio = df['roi_esperado'].mean()
    
    # 1. KPIs Financeiros
    st.subheader("KPIs Financeiros")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Orçamento Total",
            f"R$ {orcamento_total:,.2f}",
            f"{((gasto_total/orcamento_total - 1) * 100):.1f}% vs planejado"
        )
        
    with col2:
        st.metric(
            "ROI Médio",
            f"{roi_medio:.2f}x",
            f"{(roi_medio - 2):.2f}x vs target"
        )
        
    with col3:
        eficiencia = (orcamento_total - gasto_total) / orcamento_total * 100
        st.metric(
            "Eficiência Orçamentária",
            f"{eficiencia:.1f}%",
            f"{(eficiencia - 10):.1f}% vs meta"
        )
        
    with col4:
        custo_medio = gasto_total / len(df)
        st.metric(
            "Custo Médio/Projeto",
            f"R$ {custo_medio:,.2f}",
            f"{((custo_medio/50000 - 1) * 100):.1f}% vs benchmark"
        )
    
    # 2. Análise de Custos
    st.subheader("Análise de Custos")
    
    # Preparar dados de custos
    df_custos = df.groupby('departamento').agg({
        'orcamento': 'sum',
        'gasto': 'sum'
    }).reset_index()
    
    df_custos['eficiencia'] = (df_custos['orcamento'] - df_custos['gasto']) / df_custos['orcamento'] * 100
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Gráfico de barras comparando orçamento vs gasto
        fig_custos = go.Figure(data=[
            go.Bar(name='Orçamento', x=df_custos['departamento'], y=df_custos['orcamento']),
            go.Bar(name='Gasto', x=df_custos['departamento'], y=df_custos['gasto'])
        ])
        
        fig_custos.update_layout(
            title='Orçamento vs Gasto por Departamento',
            barmode='group'
        )
        st.plotly_chart(fig_custos)
        
    with col2:
        # Gráfico de eficiência
        fig_eficiencia = px.bar(
            df_custos,
            x='departamento',
            y='eficiencia',
            title='Eficiência Orçamentária por Departamento',
            color='eficiencia',
            color_continuous_scale='RdYlGn'
        )
        st.plotly_chart(fig_eficiencia)
    
    # 3. Análise Temporal
    st.subheader("Análise Temporal")
    
    # Simular dados temporais
    dates = pd.date_range(start='2024-01-01', periods=12, freq='M')
    temporal_data = pd.DataFrame()
    
    for dept in df['departamento'].unique():
        base_cost = df[df['departamento'] == dept]['gasto'].mean()
        cost_trend = base_cost + np.random.normal(0, base_cost * 0.1, len(dates)).cumsum()
        roi_trend = np.random.normal(2.5, 0.2, len(dates)).cumsum()
        
        dept_data = pd.DataFrame({
            'data': dates,
            'departamento': dept,
            'gasto': cost_trend,
            'roi': roi_trend
        })
        temporal_data = pd.concat([temporal_data, dept_data])
    
    # Gráfico de linha temporal
    fig_temporal = px.line(
        temporal_data,
        x='data',
        y='gasto',
        color='departamento',
        title='Evolução de Gastos por Departamento'
    )
    
    st.plotly_chart(fig_temporal, use_container_width=True)
    
    # 4. Insights e Recomendações
    st.subheader("Insights e Recomendações")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 💡 Principais Insights")
        
        # Calcular insights
        mais_eficiente = df_custos.nlargest(1, 'eficiencia')
        menos_eficiente = df_custos.nsmallest(1, 'eficiencia')
        
        st.markdown(f"""
        - Departamento mais eficiente: **{mais_eficiente['departamento'].iloc[0]}** ({mais_eficiente['eficiencia'].iloc[0]:.1f}%)
        - Departamento menos eficiente: **{menos_eficiente['departamento'].iloc[0]}** ({menos_eficiente['eficiencia'].iloc[0]:.1f}%)
        - Eficiência média: **{df_custos['eficiencia'].mean():.1f}%**
        - Gasto total: **R$ {gasto_total:,.2f}**
        """)
        
    with col2:
        st.markdown("#### 🎯 Recomendações")
        
        # Identificar departamentos que precisam de atenção
        baixa_eficiencia = df_custos[df_custos['eficiencia'] < df_custos['eficiencia'].mean()]
        
        st.markdown("**Ações Prioritárias:**")
        for _, dept in baixa_eficiencia.iterrows():
            st.markdown(f"""
            - **{dept['departamento']}**:
              - Eficiência atual: {dept['eficiencia']:.1f}%
              - Meta: {df_custos['eficiencia'].mean():.1f}%
              - Redução necessária: R$ {(dept['gasto'] - dept['orcamento']) * (1 - df_custos['eficiencia'].mean()/100):,.2f})
            """)
        
        st.markdown("""
        **Estratégias Gerais:**
        - Otimizar alocação de recursos
        - Implementar controles de custos
        - Revisar processos ineficientes
        - Aumentar automação
        """)

def render_riscos_oportunidades(df):
    """Renderiza análise de riscos e oportunidades"""
    st.markdown('<div class="section-header">⚠️ Riscos e Oportunidades</div>', 
                unsafe_allow_html=True)
    
    # 1. Matriz de Riscos
    st.subheader("Matriz de Riscos")
    
    # Calcular score de risco
    df['risco_score'] = (
        df['bugs_criticos'] * 0.4 +
        (100 - df['progresso']) * 0.3 +
        (10 - df['satisfacao_cliente']) * 0.3
    )
    
    # Calcular score de impacto
    df['impacto_score'] = (
        df['roi_esperado'] * 0.4 +
        df['recursos_alocados'] * 0.3 +
        df['velocidade_sprint'] * 0.3
    )
    
    # Criar matriz de riscos interativa
    risk_view = st.selectbox(
        "Selecione a visualização da matriz",
        ["Riscos vs Impacto", "Riscos vs ROI", "Riscos vs Recursos"]
    )
    
    if risk_view == "Riscos vs Impacto":
        fig_risk = px.scatter(
            df,
            x='risco_score',
            y='impacto_score',
            color='departamento',
            size='orcamento',
            hover_data=['nome', 'progresso', 'bugs_criticos'],
            title='Matriz de Riscos vs Impacto'
        )
    elif risk_view == "Riscos vs ROI":
        fig_risk = px.scatter(
            df,
            x='risco_score',
            y='roi_esperado',
            color='departamento',
            size='orcamento',
            hover_data=['nome', 'progresso', 'bugs_criticos'],
            title='Matriz de Riscos vs ROI'
        )
    else:
        fig_risk = px.scatter(
            df,
            x='risco_score',
            y='recursos_alocados',
            color='departamento',
            size='orcamento',
            hover_data=['nome', 'progresso', 'bugs_criticos'],
            title='Matriz de Riscos vs Recursos'
        )
    
    # Adicionar linhas de referência
    fig_risk.add_hline(y=df['impacto_score'].median(), line_dash="dash", line_color="gray")
    fig_risk.add_vline(x=df['risco_score'].median(), line_dash="dash", line_color="gray")
    
    st.plotly_chart(fig_risk, use_container_width=True)
    
    # 2. KPIs de Risco
    st.subheader("KPIs de Risco")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        alto_risco = len(df[df['risco'] == 'Alto'])
        
        st.metric(
            "Projetos de Alto Risco",
            f"{alto_risco}",
            f"{(alto_risco/len(df)*100):.1f}% do total"  # Corrigido a formatação
        )
        
    with col2:
        risco_medio = df['risco_score'].mean()
        st.metric(
            "Risco Médio",
            f"{risco_medio:.1f}",
            "↓ 2.3 vs último mês"
        )
        
    with col3:
        bugs_total = df['bugs_criticos'].sum()
        st.metric(
            "Total Bugs Críticos",
            f"{bugs_total}",
            "↓ 15% vs média"
        )
        
    with col4:
        atraso_medio = 100 - df['progresso'].mean()
        st.metric(
            "Atraso Médio",
            f"{atraso_medio:.1f}%",
            "↓ 5% vs target"
        )
    
    # 3. Análise de Tendências
    st.subheader("Tendências de Risco")
    
    # Simular dados de tendência
    dates = pd.date_range(start='2024-01-01', periods=12, freq='M')
    trend_data = pd.DataFrame()
    
    for dept in df['departamento'].unique():
        base_risk = df[df['departamento'] == dept]['risco_score'].mean()
        risk_trend = base_risk + np.random.normal(0, base_risk * 0.1, len(dates)).cumsum()
        
        dept_trend = pd.DataFrame({
            'data': dates,
            'departamento': dept,
            'risco': risk_trend
        })
        trend_data = pd.concat([trend_data, dept_trend])
    
    fig_trend = px.line(
        trend_data,
        x='data',
        y='risco',
        color='departamento',
        title='Evolução do Risco por Departamento'
    )
    
    # Adicionar média móvel
    for dept in trend_data['departamento'].unique():
        dept_data = trend_data[trend_data['departamento'] == dept]
        fig_trend.add_scatter(
            x=dept_data['data'],
            y=dept_data['risco'].rolling(window=3).mean(),
            name=f'{dept} (MM-3)',
            line=dict(dash='dash'),
            showlegend=True
        )
    
    st.plotly_chart(fig_trend, use_container_width=True)
    
    # 4. Oportunidades
    st.subheader("Análise de Oportunidades")
    
    # Calcular score de oportunidade
    df['oportunidade_score'] = (
        df['roi_esperado'] * 0.4 +
        df['satisfacao_cliente'] * 0.3 +
        df['velocidade_sprint'] * 0.3)
    
    # Criar visualização de oportunidades
    opp_view = st.selectbox(
        "Selecione a visualização de oportunidades",
        ["Top Oportunidades", "Oportunidades por Departamento", "Matriz de Oportunidades"]
    )
    
    if opp_view == "Top Oportunidades":
        top_opp = df.nlargest(10, 'oportunidade_score')
        fig_opp = px.bar(
            top_opp,
            x='nome',
            y='oportunidade_score',
            color='departamento',
            title='Top 10 Oportunidades'
        )
        
    elif opp_view == "Oportunidades por Departamento":
        fig_opp = px.box(
            df,
            x='departamento',
            y='oportunidade_score',
            color='departamento',
            title='Distribuição de Oportunidades por Departamento'
        )
        
    else:
        fig_opp = px.scatter(
            df,
            x='risco_score',
            y='oportunidade_score',
            color='departamento',
            size='orcamento',
            hover_data=['nome', 'roi_esperado', 'satisfacao_cliente'],
            title='Matriz de Riscos vs Oportunidades'
        )
    
    st.plotly_chart(fig_opp, use_container_width=True)

def render_analise_estrategica(df):
    """Renderiza análise estratégica detalhada"""
    st.markdown('<div class="section-header">📈 Análise Estratégica</div>', 
                unsafe_allow_html=True)
    
    # 1. KPIs Estratégicos
    st.subheader("KPIs Estratégicos")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        roi_medio = df['roi_esperado'].mean()
        st.metric(
            "ROI Médio",
            f"{roi_medio:.2f}x",
            f"{(roi_medio - 2):.2f}x vs target"
        )
        
    with col2:
        satisfacao_media = df['satisfacao_cliente'].mean()
        st.metric(
            "Satisfação Cliente",
            f"{satisfacao_media:.1f}/10",
            f"{(satisfacao_media - 8):.1f} vs meta"
        )
        
    with col3:
        alto_risco = len(df[df['risco'] == 'Alto'])
        st.metric(
            "Projetos Alto Risco",
            f"{alto_risco}",
            f"{(alto_risco/len(df)*100):.1f}% do total"
        )
        
    with col4:
        velocidade_media = df['velocidade_sprint'].mean()
        st.metric(
            "Velocidade Média",
            f"{velocidade_media:.1f}",
            f"{(velocidade_media - 80):.1f} vs baseline"
        )
    
    # 2. Matriz de Prioridades
    st.subheader("Matriz de Prioridades")
    
    # Criar scatter plot de ROI vs Risco
    fig_matriz = px.scatter(
        df,
        x='roi_esperado',
        y='velocidade_sprint',
        color='departamento',
        size='orcamento',
        hover_data=['nome', 'progresso', 'status'],
        title='Matriz ROI vs Velocidade'
    )
    
    # Adicionar linhas de referência
    fig_matriz.add_hline(y=df['velocidade_sprint'].median(), line_dash="dash")
    fig_matriz.add_vline(x=df['roi_esperado'].median(), line_dash="dash")
    
    st.plotly_chart(fig_matriz, use_container_width=True)
    
    # 3. Análise de Tendências
    st.subheader("Análise de Tendências")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # ROI por departamento
        fig_roi = px.bar(
            df.groupby('departamento')['roi_esperado'].mean().reset_index(),
            x='departamento',
            y='roi_esperado',
            title='ROI Médio por Departamento',
            color='roi_esperado',
            color_continuous_scale='RdYlGn'
        )
        st.plotly_chart(fig_roi)
        
    with col2:
        # Satisfação por departamento
        fig_satisfacao = px.box(
            df,
            x='departamento',
            y='satisfacao_cliente',
            title='Distribuição de Satisfação por Departamento'
        )
        st.plotly_chart(fig_satisfacao)
    
    # 4. Análise de Riscos
    st.subheader("Análise de Riscos")
    
    # Distribuição de riscos por departamento
    fig_riscos = px.bar(
        df.groupby(['departamento', 'risco']).size().reset_index(name='count'),
        x='departamento',
        y='count',
        color='risco',
        title='Distribuição de Riscos por Departamento',
        barmode='group'
    )
    
    st.plotly_chart(fig_riscos, use_container_width=True)
    
    # 5. Insights e Recomendações
    st.subheader("Insights e Recomendações")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 💡 Principais Insights")
        
        # Calcular insights
        top_roi = df.groupby('departamento')['roi_esperado'].mean().nlargest(1)
        top_satisfacao = df.groupby('departamento')['satisfacao_cliente'].mean().nlargest(1)
        
        st.markdown(f"""
        - Melhor ROI: **{top_roi.index[0]}** ({top_roi.values[0]:.2f}x)
        - Maior satisfação: **{top_satisfacao.index[0]}** ({top_satisfacao.values[0]:.1f}/10)
        - Projetos alto risco: **{alto_risco}** ({(alto_risco/len(df)*100):.1f}%)
        - Velocidade média: **{velocidade_media:.1f}**
        """)
        
    with col2:
        st.markdown("#### 🎯 Recomendações")
        
        # Identificar áreas que precisam de atenção
        baixo_roi = df[df['roi_esperado'] < df['roi_esperado'].mean()]['departamento'].unique()
        baixa_satisfacao = df[df['satisfacao_cliente'] < 8]['departamento'].unique()
        
        st.markdown("**Ações Prioritárias:**")
        
        if len(baixo_roi) > 0:
            st.markdown(f"- Melhorar ROI em: **{', '.join(baixo_roi)}**")
            
        if len(baixa_satisfacao) > 0:
            st.markdown(f"- Aumentar satisfação em: **{', '.join(baixa_satisfacao)}**")
        
        st.markdown("""
        **Estratégias Gerais:**
        - Otimizar alocação de recursos
        - Implementar gestão de riscos
        - Melhorar processos de entrega
        - Focar em satisfação do cliente
        """)

def main():
    """Função principal do dashboard"""
    
    # Carregar e preparar dados
    df = gerar_projetos_simulados()
    
    # Aplicar filtros via sidebar
    df_filtered = aplicar_filtros(df)
    
    # Renderizar título principal
    st.title("Cockpit @ CEO" )
    
    # Tabs para diferentes visões
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Visão Geral",
        "💰 Financeiro",
        "🔄 Operacional",
        "📈 Estratégico"
    ])
    
    # Renderizar conteúdo das tabs
    with tab1:
        render_visao_geral(df_filtered)
        
    with tab2:
        render_analise_financeira(df_filtered)
        
    with tab3:
        render_analise_operacional(df_filtered)
        
    with tab4:
        render_analise_estrategica(df_filtered)
        render_bcg_matrix(df_filtered)
        render_riscos_oportunidades(df_filtered)
        render_metricas_tecnicas(df_filtered)
        render_analise_produtos(df_filtered)

# Executar a aplicação
if __name__ == "__main__":
    main()
