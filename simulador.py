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

# Create the simulator
st.set_page_config(page_title="Simulador de Spread", layout="wide")

st.title("📊 Simulador de Spread Avançado")
st.markdown("---")

# Load data
try:
    df = pd.read_csv('clientes_agregados_com_bandas.csv')
    df['TicketMedio'] = df['VolumeTotal'] / df['NumeroOperacoes']
except FileNotFoundError:
    st.error("Arquivo 'clientes_agregados_com_bandas.csv' não encontrado!")
    st.stop()

# Sidebar
st.sidebar.header("Configurações")
target_rate = st.sidebar.slider(
    "Taxa Desejada (%)",
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

col1.metric("Clientes Acima da Taxa", f"{above_rate} ({(above_rate/len(df)*100):.1f}%)")
col2.metric("Clientes Abaixo da Taxa", f"{below_rate} ({(below_rate/len(df)*100):.1f}%)")
col3.metric("Volume Mensal Potencial", f"R$ {volume_mensal_acumulado:,.2f}")
col4.metric("Projeção 13 Meses", f"R$ {projecao_13_meses:,.2f}")
col5.metric("Ticket Médio Global", f"R$ {ticket_medio_global:,.2f}")

# Detailed analysis by band
st.markdown("### Análise Detalhada por Banda")

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
    title='Distribuição das Taxas por Banda (Box Plot)',
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
    size='TicketMedio',
    title='Relação entre Volume, Taxa e Ticket Médio',
    labels={
        'VolumeMediaMensal': 'Volume Médio Mensal (R$)',
        'TaxaMediaPonderada': 'Taxa Média Ponderada',
        'BandaCliente': 'Banda do Cliente',
        'TicketMedio': 'Ticket Médio'
    }
)
fig_scatter.add_hline(
    y=target_rate,
    line_dash="dash",
    line_color="red",
    annotation_text=f"Taxa Alvo: {target_rate:.1%}"
)
st.plotly_chart(fig_scatter, use_container_width=True)

# Correlation Heatmap
st.markdown("### Matriz de Correlação")
correlation_data = df[['TaxaMediaPonderada', 'VolumeMediaMensal', 'TicketMedio', 'NumeroOperacoes']].corr()
fig_corr = px.imshow(
    correlation_data,
    title='Matriz de Correlação',
    labels=dict(color="Correlação"),
    color_continuous_scale='RdBu'
)
st.plotly_chart(fig_corr, use_container_width=True)

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
    label="📥 Download da Análise Completa",
    data=analysis_df.to_csv().encode('utf-8'),
    file_name='analise_completa.csv',
    mime='text/csv'
)

# New Machine Learning Section
st.markdown("## 🤖 Análise Preditiva")

# Data preparation for ML
features_for_clustering = ['TaxaMediaPonderada', 'VolumeMediaMensal', 'TicketMedio']
X_cluster = df[features_for_clustering].copy()

# Normalization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

# K-means Clustering
st.markdown("### 📊 Segmentação de Clientes (K-means)")
n_clusters = st.slider("Número de Segmentos", min_value=2, max_value=8, value=4)

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

# Continuação do código anterior...

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

# Linear Regression
st.markdown("### 📈 Previsão de Taxas (Regressão Linear)")

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

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': X_reg.columns,
    'Importância': model.coef_
})

fig_importance = px.bar(
    feature_importance,
    x='Feature',
    y='Importância',
    title='Importância das Variáveis na Previsão da Taxa'
)
st.plotly_chart(fig_importance, use_container_width=True)

# Rate Simulator
st.markdown("#### 🎯 Simulador de Taxa")
col1, col2, col3 = st.columns(3)

volume_sim = col1.number_input("Volume Mensal", min_value=0.0, value=100000.0, step=10000.0)
ticket_sim = col2.number_input("Ticket Médio", min_value=0.0, value=5000.0, step=1000.0)
num_op_sim = col3.number_input("Número de Operações", min_value=1, value=20, step=1)

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

