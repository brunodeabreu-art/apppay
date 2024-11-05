import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from datetime import datetime, timedelta

st.set_page_config(page_title='AmigoPay - Assistente de Negociação', layout='wide')

st.markdown(
    """
    <div style='
        background-color: #0066cc; 
        color: white; 
        padding: 5px; 
        border-radius: 5px; 
        text-align: center;
        margin-bottom: 20px;
        animation: pulse 2s infinite;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    '>
        <style>
            @keyframes pulse {
                0% { opacity: 1; }
                50% { opacity: 0.8; }
                100% { opacity: 1; }
            }
        </style>
        <strong></strong>  
    </div>
    """,
    unsafe_allow_html=True
)

# Custos por modalidade
CUSTOS = {
    'debito': 49,
    'credito_1x': 58,
    'credito_2x': 31,
    'credito_3x': 1,
    'credito_4x': 11,
    'credito_5x': 11,
    'credito_6x': 11,
    'credito_7x': 11,
    'credito_8x': 11,
    'credito_9x': 11,
    'credito_10x': 11,
    'credito_11x': 11,
    'credito_12x': 11,
    'outras' : 1831
    
}

@st.cache_data
def load_data(arquivo):
    df = pd.read_csv(arquivo)
    df['mdr'] = df.get('mdr', 2.0)
    df['modalidade'] = df['modalidade'].str.lower()
    df['data_transacao'] = pd.to_datetime(df['data_transacao'])
    return df

def process_data(df):
    df[['taxa_sugerida', 'spread']] = df.apply(
        lambda x: pd.Series([
            CUSTOS.get(x['modalidade'].lower(), CUSTOS['outras']),
            x.get('mdr', 0) - CUSTOS.get(x['modalidade'].lower(), CUSTOS['outras'])
        ]),
        axis=1
    )
    return df

def classificar_cliente_por_volume(volume_mensal):
    if volume_mensal <= 50000: return 'Pequeno Porte'
    elif volume_mensal <= 300000: return 'M\u00e9dio Porte'
    return 'Grande Porte'

# Tabela de taxas sugeridas
TAXAS_SUGERIDAS = pd.DataFrame({
    'Modalidade': ['Debito', 'Credito 1x', 'Credito 2x', 'Credito 3x', 'Credito 4x', 'Credito 5x',
                   'Credito 6x', 'Credito 7x', 'Credito 8x', 'Credito 9x', 'Credito 10x', 'Credito 11x', 'Credito 12x'],
    '0-50k': [1.462, 2.637, 2.923, 2.925, 2.922, 2.923,
              2.932, 3.178, 3.178, 3.178, 3.179, 3.179, 3.179],
    '50-300k': [1.434, 2.543, 2.900, 2.904, 2.901, 2.906,
                2.920, 3.192, 3.192, 3.193, 3.197, 3.199, 3.205],
    '300k+': [1.370, 2.371, 2.779, 2.804, 2.828, 2.852,
              2.886, 3.128, 3.146, 3.162, 3.177, 3.178, 3.184]
})

arquivo_carregado = st.sidebar.file_uploader("Carregue aqui o Extrato de Transações", type=["csv"])

if arquivo_carregado:
    df = process_data(load_data(arquivo_carregado))
    
    data_min, data_max = df['data_transacao'].min(), df['data_transacao'].max()
    intervalo_datas = st.sidebar.date_input("Per\u00edodo", value=(data_min, data_max),
                                          min_value=data_min, max_value=data_max)
    
    filtros = {
        'bandeira': st.sidebar.selectbox('Bandeira', ['Todas'] + sorted(df['bandeira'].unique())),
        'modalidade': st.sidebar.selectbox('Modalidade', ['Todas'] + sorted(df['modalidade'].unique()))
    }
    
    mask = (df['data_transacao'].dt.date >= intervalo_datas[0]) & \
           (df['data_transacao'].dt.date <= intervalo_datas[1])
    for campo, valor in filtros.items():
        if valor != 'Todas':
            mask &= (df[campo] == valor)
    
    df_filtrado = df[mask].copy()
    
    # Calculando volume mensal m\u00e9dio para classifica\u00e7\u00e3o
    volume_mensal = df_filtrado.groupby([df_filtrado['data_transacao'].dt.to_period('M')])['valor'].sum()
    volume_mensal_medio = volume_mensal.mean()
    classificacao_cliente = classificar_cliente_por_volume(volume_mensal_medio)
    
    # Selecionando a coluna correta de taxas sugeridas
    if classificacao_cliente == 'Pequeno Porte':
        coluna_taxas = '0-50k'
    elif classificacao_cliente == 'M\u00e9dio Porte':
        coluna_taxas = '50-300k'
    else:
        coluna_taxas = '300k+'
    
    # Criando DataFrame com taxas sugeridas e margem de negocia\u00e7\u00e3o
    taxas_filtradas = TAXAS_SUGERIDAS[['Modalidade', coluna_taxas]].copy()
    taxas_filtradas.columns = ['Modalidade', 'Taxa Sugerida (%)']
    
    # Adicionando custos e calculando margem
    taxas_filtradas['Custo (%)'] = taxas_filtradas['Modalidade'].apply(
        lambda x: CUSTOS.get(x.lower().replace(' ', '_'), CUSTOS['outras'])
    )
    taxas_filtradas['Margem de Negocia\u00e7\u00e3o (%)'] = (
        taxas_filtradas['Taxa Sugerida (%)'] - taxas_filtradas['Custo (%)']
    ).round(3)
    
    tabs = st.tabs(["Vis\u00e3o Geral", "An\u00e1lise Temporal", "M\u00e9tricas Avan\u00e7adas", 
                    "An\u00e1lise de Risco", "An\u00e1lise de Bandeiras"])
    
    with tabs[0]:
        st.header("Vis\u00e3o Geral")
        
        # Criando 6 colunas
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        
        volume_mensal = df_filtrado.groupby(df_filtrado['data_transacao'].dt.to_period('M'))['valor'].sum().mean()
        desvio_padrao_mensal = df_filtrado.groupby(df_filtrado['data_transacao'].dt.to_period('M'))['valor'].sum().std()
        
        # M\u00e9tricas
        col1.metric('Volume Total', f"R$ {df_filtrado['valor'].sum():,.2f}")
        col2.metric('Quantidade de Transa\u00e7\u00f5es', len(df_filtrado))
        col3.metric('Ticket M\u00e9dio', f"R$ {df_filtrado['valor'].mean():,.2f}")
        col4.metric('Volume Mensal', f"R$ {volume_mensal:,.2f}")
        col5.metric('Desvio Padr\u00e3o Mensal', f"R$ {desvio_padrao_mensal:,.2f}")
        col6.metric('Classifica\u00e7\u00e3o Cliente', classificacao_cliente)
        
        # Exibindo a tabela de taxas sugeridas filtradas com margem
        st.subheader(f"Taxas e Margens para {classificacao_cliente}")
        st.table(taxas_filtradas)
            
        fig = px.line(df_filtrado.groupby('data_transacao')['valor'].sum().reset_index(),
                     x='data_transacao', y='valor', title='Volume Di\u00e1rio')
        st.plotly_chart(fig)
    
    with tabs[1]:
        st.header("An\u00e1lise Temporal")
        temporal = df_filtrado.groupby('data_transacao').agg({
            'valor': ['sum', 'count'],
            'mdr': 'mean'
        }).round(2)
        temporal.columns = ['_'.join(col).strip() for col in temporal.columns.values]
        temporal = temporal.reset_index()
        
        fig1 = px.line(temporal, x='data_transacao', y='valor_sum', 
                      title='Volume Total por Data')
        st.plotly_chart(fig1)
        
        fig2 = px.line(temporal, x='data_transacao', y='valor_count', 
                      title='Quantidade de Transa\u00e7\u00f5es por Data')
        st.plotly_chart(fig2)
        
        fig3 = px.line(temporal, x='data_transacao', y='mdr_mean', 
                      title='MDR M\u00e9dio por Data')
        st.plotly_chart(fig3)
    
    with tabs[2]:
        
        st.header("M\u00e9tricas Avan\u00e7adas")
        metricas_avancadas = df_filtrado.groupby('modalidade').agg({
            'valor': ['count', 'sum', 'mean'],
            'mdr': 'mean',
            'spread': 'mean'
        }).round(2)
        metricas_avancadas.columns = ['_'.join(col).strip() for col in metricas_avancadas.columns.values]
        metricas_avancadas = metricas_avancadas.reset_index()
        st.write(metricas_avancadas)
        
        df_mensal_modalidade = df_filtrado.groupby([df_filtrado['data_transacao'].dt.to_period('M'), 'modalidade'])['valor'].sum().reset_index()
        df_mensal_modalidade['data_transacao'] = df_mensal_modalidade['data_transacao'].dt.to_timestamp()
        fig_modalidade = px.bar(df_mensal_modalidade, x='data_transacao', y='valor', color='modalidade',
                                title='Composi\u00e7\u00e3o do Volume por Modalidade por M\u00eas', barmode='stack')
        st.plotly_chart(fig_modalidade)
    
    with tabs[3]:
        st.header("An\u00e1lise de Risco")
        risco = px.scatter(df_filtrado, x='valor', y='mdr',
                         color='modalidade', title='Rela\u00e7\u00e3o Valor x MDR')
        st.plotly_chart(risco)
        
        volatilidade_diaria = df_filtrado.groupby('data_transacao')['valor'].std().reset_index()
        volatilidade_mensal = df_filtrado.groupby(df_filtrado['data_transacao'].dt.to_period('M'))['valor'].std().reset_index()
        volatilidade_mensal['data_transacao'] = volatilidade_mensal['data_transacao'].dt.to_timestamp()
        
        fig_vol_diaria = px.line(volatilidade_diaria, x='data_transacao', y='valor',
                                 title='Volatilidade Di\u00e1ria do Volume')
        st.plotly_chart(fig_vol_diaria)
        
        fig_vol_mensal = px.line(volatilidade_mensal, x='data_transacao', y='valor',
                                 title='Volatilidade Mensal do Volume')
        st.plotly_chart(fig_vol_mensal)
    
    with tabs[4]:
        st.header("Análise de Bandeiras")
    
    # Distribution of bandeiras by modalidade
        bandeiras_modalidade = df_filtrado.groupby(['modalidade', 'bandeira']).agg({
        'valor': ['sum', 'count'],
        'mdr': 'mean'
         }).round(2)
        bandeiras_modalidade.columns = ['_'.join(col).strip() for col in bandeiras_modalidade.columns.values]
        bandeiras_modalidade = bandeiras_modalidade.reset_index()
        st.write("Distribuição por Modalidade e Bandeira:", bandeiras_modalidade)
    
        col1, col2 = st.columns(2)
        with col1:
        # Treemap showing the hierarchy of modalidade and bandeira
            fig_tree = px.treemap(
            df_filtrado, 
            path=['modalidade', 'bandeira'], 
            values='valor',
            title='Distribuição Hierárquica por Modalidade e Bandeira'
        )
        st.plotly_chart(fig_tree)
    
        with col2:
        # Stacked bar chart showing the distribution
            fig_bar = px.bar(
            df_filtrado, 
            x='modalidade', 
            y='valor',
            color='bandeira',
            title='Distribuição de Valor por Modalidade e Bandeira',
            barmode='stack'
            )
        st.plotly_chart(fig_bar)
    
    # Additional visualization for MDR distribution
        fig_box = px.box(
            df_filtrado, 
            x='modalidade', 
            y='mdr',
            color='bandeira',
            title='Distribuição de MDR por Modalidade e Bandeira'
            )
        st.plotly_chart(fig_box)
else:
    st.markdown("""
    <div style="text-align: center; width: 100%; padding: 20px;">
        <h1 style="color: #FFFFF; margin-bottom: 20px;">Bem-vindo ao Assistente de Negociações - AmigoPay</h1>
        <p style="font-size: 16px; line-height: 1.6; max-width: 800px; margin: 0 auto;">
            Essa aplicação auxilia vendedor na tomada de decisão sobre quais taxas ofertar ao Cliente.
            Ela avalia extratos de maquinetas POS dos clientes e retorna as taxas ótimas sugeridas. 
            Além disso, ele traz um playground para simulações de cenários e faz análises sobre margens de negociação de acordo com o perfil do cliente.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="margin-top: 100px; text-align: center;">
        <h2 style="color: white; font-size: 24px;">
            Faça upload do extrato das transações no painel ao lado e comece agora mesmo.
        </h2>
    </div>
""", unsafe_allow_html=True)
    
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
   

   # Interactive demo with sample parameters
    st.subheader("Playground - Amigo Pay")
    
    # Sample parameter selection
    col1, col2 = st.columns(2)
    with col1:
        demo_modalidade = st.selectbox(
            'Selecione uma modalidade de exemplo:',
            ['débito', 'crédito_1x', 'crédito_2x', 'crédito_3x']
        )
        st.markdown("""
        <style>
        .stSlider-range {
            background-color: #0066cc !important;
        }
        </style>
        """, unsafe_allow_html=True)

        demo_valor = st.slider('Valor da transação:', 100, 10000, 1000)
    
    with col2:
        demo_bandeira = st.selectbox(
            'Selecione uma bandeira:',
            ['Visa', 'Mastercard', 'Elo', 'Amex']
        )
        demo_mdr = st.slider('MDR (%):', 0.0, 5.0, 2.0)
    
    # Create sample transaction
    if st.button('Simular Transação'):
        sample_data = pd.DataFrame({
            'modalidade': [demo_modalidade],
            'bandeira': [demo_bandeira],
            'valor': [demo_valor],
            'mdr': [demo_mdr],
            'data_transacao': [pd.Timestamp.now()]
        })
        
        st.write("### Resultado da Simulação:")
        st.dataframe(sample_data)
        
        # Sample metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Valor Bruto", f"R$ {demo_valor:,.2f}")
        with col2:
            st.metric("Taxa MDR", f"{demo_mdr}%")
        with col3:
            st.metric("Valor Líquido", f"R$ {demo_valor * (1 - demo_mdr/100):,.2f}")
    
    
    st.write("")
    st.write("")
    st.write("")
    st.write("")

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

# Function to remove outliers
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# # Create the simulator
# st.set_page_config(page_title="Simulador de Spread", layout="wide")

st.title("Simulador - Pricing")
st.markdown("---")

# Load data
arquivo = st.selectbox('Selecione o arquivo:', ['clientes.csv', 'clientes_raw.csv'])

try:
    # Carrega o arquivo selecionado
    df = pd.read_csv(arquivo)
    
    # Calcula o ticket médio
    df['TicketMedio'] = df['VolumeTotal'] / df['NumeroOperacoes']
    
    # Mostra o total de linhas e as primeiras 5 linhas
    st.write('Total de registros:', df.shape[0])
    st.dataframe(df.head())
    
except FileNotFoundError:
    st.error(f'Arquivo {arquivo} não encontrado!')
except Exception as e:
    st.error(f'Erro: {str(e)}')

# Sidebar
st.sidebar.header("Configurações")
target_rate = st.sidebar.slider(
    "Taxa Target (%)",
    min_value=0.0,
    max_value=10.0,
    value=3.0,
    step=0.1
) / 100

# Option to remove outliers
remove_outliers_option = st.sidebar.checkbox("Remover Outliers", value=False)

# Process data
if remove_outliers_option:
    df = remove_outliers(df, 'TaxaMediaPonderada')
    df = remove_outliers(df, 'VolumeMediaMensal')

# Main metrics
col1, col2, col3, col4, col5 = st.columns(5)

clientes_acima = df[df['TaxaMediaPonderada'] > target_rate]
clientes_abaixo = df[df['TaxaMediaPonderada'] <= target_rate]

above_rate = len(clientes_acima)
below_rate = len(clientes_abaixo)
volume_mensal_acumulado = clientes_acima['VolumeMediaMensal'].sum()
projecao_13_meses = volume_mensal_acumulado * 13
ticket_medio_global = df['TicketMedio'].mean()

col1.metric("Clientes Acima da Taxa", f"{above_rate} ({(above_rate/len(df)*100):.2f}%)")
col2.metric("Clientes Abaixo da Taxa", f"{below_rate} ({(below_rate/len(df)*100):.2f}%)")
col3.metric("Volume Mensal Potencial", f"R$ {volume_mensal_acumulado:.2f}")
col4.metric("Proje\u00e7\u00e3o 13 Meses", f"R$ {projecao_13_meses:.2f}")
            
# Detailed analysis by band
st.markdown("### Análise Detalhada por Revenue Band")

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
    height=500
)

st.plotly_chart(fig_dist, use_container_width=True)

# Box Plot of rates by band
fig_box = px.box(
    df,
    x='BandaCliente',
    y='TaxaMediaPonderada',
    title='Distribuição das Taxas por Banda - Box Plot',
    labels={'TaxaMediaPonderada': 'Taxa Média Ponderada', 'BandaCliente': 'Banda do Cliente'}
)
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
    size='VolumeMediaMensal',
    title='Relação entre Volume, Taxa e Ticket Médio',
    labels={
        'VolumeMediaMensal': 'Volume Médio Mensal (R$)',
        'TaxaMediaPonderada': 'Taxa Média Ponderada',
        'BandaCliente': 'Banda do Cliente',
        'VolumeMediaMensal': 'VolumeMediaMensal'
    }
)
fig_scatter.add_hline(
    y=target_rate,
    line_dash="dash",
    line_color="red",
    annotation_text=f"Taxa Alvo: {target_rate:.1%}"
)
st.plotly_chart(fig_scatter, use_container_width=True)

# Correlation Heatma

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
    title='Distribuição Win/Lose de Clientes por Volume e Taxa',
    barmode='stack',
    height=500
)

st.plotly_chart(fig_vol, use_container_width=True)

# Potential Volume by Category  
volume_potential = df[df['TaxaMediaPonderada'] > target_rate].groupby('BandaCliente')['VolumeMediaMensal'].sum().reset_index()

fig_pot = px.bar(
    volume_potential,
    x='BandaCliente',
    y='VolumeMediaMensal',
    title=f'Volume Mensal Potencial por Categoria (Taxa > {target_rate:.1%})',
    labels={'BandaCliente': 'Banda do Cliente', 'VolumeMediaMensal': 'Volume Potencial (R$)'}
)

st.plotly_chart(fig_pot, use_container_width=True)

# Detailed Statistics
st.markdown("### Estatísticas Detalhadas")
analysis_df = df.groupby('BandaCliente').agg({
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

# New Machine Learning Section
st.markdown("## Machine Learning")

# Data preparation for ML
features_for_clustering = ['TaxaMediaPonderada', 'VolumeMediaMensal', 'TicketMedio']
X_cluster = df[features_for_clustering].copy()

# Normalization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

# K-means Clustering
st.markdown("### Essa seção utiliza o algoritmo de segmentação (K-means) para identificar segmentações")
n_clusters = st.slider("Selecione o Número de Segmentos Desejado", min_value=2, max_value=8, value=4)

kmeans = KMeans(n_clusters=n_clusters, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# 3D Cluster Visualization
fig_3d = px.scatter_3d(
    df,
    x='VolumeMediaMensal',
    y='TaxaMediaPonderada',
    z='TicketMedio',
    color='Cluster',
    title='Segmentação 3D de Clientes',
    labels={
        'VolumeMediaMensal': 'Volume Médio Mensal',
        'TaxaMediaPonderada': 'Taxa Média Ponderada',
        'TicketMedio': 'Ticket Médio'
    }
)
st.plotly_chart(fig_3d, use_container_width=True)

# Cluster Analysis
cluster_analysis = df.groupby('Cluster').agg({
    'TaxaMediaPonderada': ['mean', 'count'],
    'VolumeMediaMensal': 'mean',
    'TicketMedio': 'mean'
}).round(4)

cluster_analysis.columns = ['Taxa Média', 'Quantidade Clientes', 'Volume Médio', 'Ticket Médio']
st.markdown("#### Características dos Segmentos")
st.dataframe(
    cluster_analysis.style.format({
        'Taxa Média': '{:.2%}',
        'Volume Médio': 'R$ {:,.2f}',
        'Ticket Médio': 'R$ {:,.2f}'
    })
)

st.write()
st.write()



# Linear Regression
st.markdown("### Regressão Linear - % target")

# Prepare data for regression
X_reg = df[['VolumeMediaMensal', 'TicketMedio', 'NumeroOperacoes']]
y_reg = df['TaxaMediaPonderada']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate metrics
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# Display metrics
col1, col2 = st.columns(2)
col1.metric("R² Score", f"{r2:.3f}")
col2.metric("RMSE", f"{rmse:.4f}")


# Rate Simulator
st.markdown("#### 🎯 Simulador de Taxa")
col1, col2, col3 = st.columns(3)

volume_sim = col1.number_input("Volume Mensal", min_value=0.0, value=100000.0, step=10000.0)

if st.button("Calcular Taxa Prevista"):
    # Make prediction
    taxa_prevista = model.predict([[volume_sim, ticket_sim, num_op_sim]])[0]
    st.success(f"Taxa Prevista: {taxa_prevista:.2%}")
    
    # Compare with market average
    taxa_media_mercado = df['TaxaMediaPonderada'].mean()
    diferenca = taxa_prevista - taxa_media_mercado
    
    st.info(f"""
    - Taxa Média do Mercado: {taxa_media_mercado:.2%}
    - Diferença: {diferenca:.2%} {'acima' if diferenca > 0 else 'abaixo'} da média
    """)

# Scatter plot of actual vs predicted rates
fig_pred = px.scatter(
    x=y_test,
    y=y_pred,
    labels={'x': 'Taxa Real', 'y': 'Taxa Prevista'},
    title='Comparação entre Taxas Reais e Previstas'
)
fig_pred.add_trace(
    go.Scatter(
        x=[y_test.min(), y_test.max()],
        y=[y_test.min(), y_test.max()],
        mode='lines',
        name='Linha de Referência',
        line=dict(dash='dash')
    )
)
st.plotly_chart(fig_pred, use_container_width=True)

# Download predictive analysis results
analysis_results = pd.DataFrame({
    'Métrica': ['R² Score', 'RMSE', 'Número de Clusters', 'Taxa Média de Mercado'],
    'Valor': [r2, rmse, n_clusters, taxa_media_mercado]
})

st.download_button(
    label="📥 Download da Análise Preditiva",
    data=analysis_results.to_csv().encode('utf-8'),
    file_name='analise_preditiva.csv',
    mime='text/csv'
)

# Additional insights
st.markdown("### 📊 Insights Adicionais")

# Distribution of predictions vs actual values
fig_dist = go.Figure()
fig_dist.add_trace(go.Histogram(x=y_test, name='Valores Reais', opacity=0.7))
fig_dist.add_trace(go.Histogram(x=y_pred, name='Valores Previstos', opacity=0.7))
fig_dist.update_layout(
    title='Distribuição das Taxas: Reais vs Previstas',
    barmode='overlay',
    xaxis_title='Taxa',
    yaxis_title='Frequência'
)
st.plotly_chart(fig_dist, use_container_width=True)

# Residual analysis
residuals = y_test - y_pred
fig_residuals = px.scatter(
    x=y_pred,
    y=residuals,
    labels={'x': 'Valores Previstos', 'y': 'Resíduos'},
    title='Análise de Resíduos'
)
fig_residuals.add_hline(y=0, line_dash="dash", line_color="red")
st.plotly_chart(fig_residuals, use_container_width=True)

# Final considerations
st.markdown("""
### 📝 Considerações Finais
- O modelo de clustering identificou padrões naturais de segmentação dos clientes
- A regressão linear permite prever taxas com base nas características do cliente
- Os resultados podem ser utilizados para otimização de pricing e segmentação
""")

# Cache the results
if 'model' not in st.session_state:
    st.session_state['model'] = model
if 'scaler' not in st.session_state:
    st.session_state['scaler'] = scaler
if 'kmeans' not in st.session_state:
    st.session_state['kmeans'] = kmeans





    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    st.write("")
    
    # Interactive demo section
    with st.expander("Precisa de ajuda?"):
        st.markdown("""
    ### Como usar esta ferramenta:
    1. Use o botão no sidebar para carregar seu arquivo CSV
    2. O arquivo deve conter as seguintes colunas:
        - data_transacao
        - modalidade
        - bandeira
        - valor
        - mdr (opcional)
        
    """)


  
    # Help section
    with st.expander("Dúvidas Frequentes"):
        st.markdown("""
        ### Dúvidas Frequentes:
        1. **Qual formato de arquivo é aceito?**
           - Apenas arquivos CSV com as colunas necessárias
        
        2. **Como preparar meus dados?**
           - Certifique-se que as datas estejam em formato YYYY-MM-DD
           - Valores devem usar ponto como separador decimal
           - Modalidades devem estar em minúsculas
        
        3. **Preciso de ajuda adicional?**
           - Entre em contato: bruno.abreu@amigotech.com.br ou via Slack.
        """)
    # Adding an advanced footer to the Streamlit application
# Create a styled footer with Julius AI branding
    st.markdown("---")  # Separator line

# Using columns to create a centered layout
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown(
        """
        <div style='text-align: center; color: #666666; padding: 20px;'>
            <p>Desenvolvido por</p>
               Time de Dados @ Amigo Tech
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
