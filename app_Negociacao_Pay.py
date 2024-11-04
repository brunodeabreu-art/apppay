import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from datetime import datetime, timedelta

st.set_page_config(
    page_title="AmigoPay @ Assistente de Negociações",
    layout="wide",
    initial_sidebar_state="collapsed"  # Mudado de "expanded" para "collapsed"
)

# Força o tema branco para todos
st.markdown("""
    <style>
        /* Main container */
        .stApp {
            background-color: #FFFFFF;
        }
        
        /* Sidebar */
        .css-1d391kg {
            background-color: #F0F0F0;
        }
        
        /* Cards/Boxes */
        .stMarkdown {
            background-color: #FFFFFF;
        }
        
        /* Text color */
        .stMarkdown, .stText {
            color: #262730;
        }
    </style>
""", unsafe_allow_html=True)

# Primeiro o header centralizado


# Custos por modalidade
CUSTOS = {
    'debito': 0.99,
    'credito_1x': 1.58,
    'credito_2x': 1.21,
    'credito_3x': 1.21,
    'credito_4x': 1.21,
    'credito_5x': 1.21,
    'credito_6x': 1.21,
    'credito_7x': 1.21,
    'credito_8x': 1.21,
    'credito_9x': 1.21,
    'credito_10x': 1.21,
    'credito_11x': 1.21,
    'credito_12x': 1.21,
    'outras' : 1.21
    
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

arquivo_carregado = st.sidebar.file_uploader("Carregar CSV", type=["csv"])

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
        <h1 style="color: #333; margin-bottom: 20px;">Bem-vindo ao Assistente de Negociações - AmigoPay</h1>
        <p style="font-size: 16px; line-height: 1.6; max-width: 800px; margin: 0 auto;">
            Essa aplicação auxilia o Vendedor na tomada de decisão sobre quais taxas ofertar ao Cliente.
            Ela avalia extratos de maquinetas POS dos clientes e retorna as taxas sugeridas. 
            Além disso ele traz um simulador de transações e faz algumas análises para facilitar a decisão
        </p>
    </div>
    """, unsafe_allow_html=True)

# Espaçamento
    st.markdown("<br>", unsafe_allow_html=True)

# Estilo para o uploader
    st.markdown("""
    <style>
        .upload-container {
            display: flex;
            justify-content: center;
            margin-top: 20px;
        }
        .stFileUploader {
            width: 400px;
        }
        .stFileUploader > div {
            background-color: white;
            border: 1px dashed #cccccc;
            padding: 20px;
            border-radius: 8px;
        }
        .stFileUploader > div:hover {
            border-color: #808080;
            background-color: #fafafa;
        }
        /* Centraliza o conteúdo do uploader */
        .stFileUploader > div > div {
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

# Uploader centralizado
col1, col2, col3 = st.columns([1,2,1])
with col2:
    uploaded_file = st.file_uploader("📄 Arraste ou selecione o arquivo", 
                                    type=['xlsx', 'csv'],
                                    label_visibility="collapsed")

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
   

    # # Interactive demo with sample parameters
    # st.subheader("Simulador de Transações")
    
    # # Sample parameter selection
    # col1, col2 = st.columns(2)
    # with col1:
    #     demo_modalidade = st.selectbox(
    #         'Selecione uma modalidade de exemplo:',
    #         ['débito', 'crédito_1x', 'crédito_2x', 'crédito_3x']
    #     )
    #     st.markdown("""
    #     <style>
    #     .stSlider-range {
    #         background-color: #0066cc !important;
    #     }
    #     </style>
    #     """, unsafe_allow_html=True)

    #     demo_valor = st.slider('Valor da transação:', 100, 10000, 1000)
    
    # with col2:
    #     demo_bandeira = st.selectbox(
    #         'Selecione uma bandeira:',
    #         ['Visa', 'Mastercard', 'Elo', 'Amex']
    #     )
    #     demo_mdr = st.slider('MDR (%):', 0.0, 5.0, 2.0)
    
    # # Create sample transaction
    # if st.button('Simular Transação'):
    #     sample_data = pd.DataFrame({
    #         'modalidade': [demo_modalidade],
    #         'bandeira': [demo_bandeira],
    #         'valor': [demo_valor],
    #         'mdr': [demo_mdr],
    #         'data_transacao': [pd.Timestamp.now()]
    #     })
        
    #     st.write("### Resultado da Simulação:")
    #     st.dataframe(sample_data)
        
    #     # Sample metrics
    #     col1, col2, col3 = st.columns(3)
    #     with col1:
    #         st.metric("Valor Bruto", f"R$ {demo_valor:,.2f}")
    #     with col2:
    #         st.metric("Taxa MDR", f"{demo_mdr}%")
    #     with col3:
    #         st.metric("Valor Líquido", f"R$ {demo_valor * (1 - demo_mdr/100):,.2f}")
    
    
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

