# app.py

import streamlit as st
import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from io import StringIO, BytesIO
from collections import Counter
import datetime

# Configuração da página
st.set_page_config(
    page_title="Análise Avançada de Carrinhos",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilo personalizado
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stButton>button {
        width: 100%;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

# Funções de Utilidade
@st.cache_data
def load_data(uploaded_file):
    """Carrega e processa o arquivo CSV"""
    try:
        try:
            df = pd.read_csv(uploaded_file, header=None, encoding='utf-8', on_bad_lines='skip')
        except:
            string_data = StringIO(uploaded_file.getvalue().decode('utf-8'))
            df = pd.read_csv(string_data, header=None, encoding='utf-8', on_bad_lines='skip')
        
        # Garantir que estamos trabalhando com uma única coluna
        if df.shape[1] > 1:
            df[0] = df.apply(lambda x: ','.join(x.dropna().astype(str)), axis=1)
            df = df[[0]]
        return df
    except Exception as e:
        st.error(f"Erro ao carregar o arquivo: {str(e)}")
        return None

@st.cache_data
def process_transactions(df):
    """Processa as transações do DataFrame"""
    try:
        # Converter string de produtos em lista
        transactions = df[0].str.split(',')
        transactions = transactions.apply(lambda x: [item.strip() for item in x if item.strip()])
        
        # Criar lista de produtos únicos e matriz de transações
        unique_items = sorted(list(set([item for sublist in transactions for item in sublist])))
        basket_sets = pd.DataFrame(
            [[1 if item in transaction else 0 for item in unique_items]
             for transaction in transactions],
            columns=unique_items
        )
        return basket_sets, transactions, unique_items
    except Exception as e:
        st.error(f"Erro ao processar transações: {str(e)}")
        return None, None, None

@st.cache_data
def calculate_product_metrics(basket_sets, transactions):
    """Calcula métricas por produto"""
    product_metrics = {}
    
    # Frequência absoluta
    product_freq = basket_sets.sum()
    
    # Percentual de presença em carrinhos
    product_presence = (basket_sets.sum() / len(basket_sets)) * 100
    
    # Média de outros produtos quando este está presente
    avg_other_products = {}
    for col in basket_sets.columns:
        baskets_with_product = basket_sets[basket_sets[col] == 1]
        if len(baskets_with_product) > 0:
            avg_other_products[col] = (baskets_with_product.sum().sum() - baskets_with_product[col].sum()) / len(baskets_with_product)
        else:
            avg_other_products[col] = 0
    
    # Diversidade de combinações
    combination_diversity = {}
    for col in basket_sets.columns:
        baskets_with_product = basket_sets[basket_sets[col] == 1]
        unique_combinations = len(set([tuple(sorted(t)) for t in transactions[basket_sets[col] == 1]]))
        combination_diversity[col] = unique_combinations
    
    return {
        'frequency': product_freq,
        'presence_percent': product_presence,
        'avg_other_products': pd.Series(avg_other_products),
        'combination_diversity': pd.Series(combination_diversity)
    }

@st.cache_data
def analyze_product_segments(basket_sets, product_metrics):
    """Análise avançada de segmentação de produtos"""
    segments = pd.DataFrame({
        'frequencia': product_metrics['frequency'],
        'presenca': product_metrics['presence_percent'],
        'diversidade': product_metrics['combination_diversity'],
        'media_outros': product_metrics['avg_other_products']
    })
    
    # Normalização dos valores
    for col in segments.columns:
        segments[f'{col}_norm'] = (segments[col] - segments[col].min()) / (segments[col].max() - segments[col].min())
    
    # Score composto
    segments['score_composto'] = (
        segments['frequencia_norm'] * 0.3 +
        segments['presenca_norm'] * 0.3 +
        segments['diversidade_norm'] * 0.2 +
        segments['media_outros_norm'] * 0.2
    )
    
    # Categorização
    segments['categoria'] = pd.qcut(
        segments['score_composto'],
        q=4,
        labels=['Baixo Desempenho', 'Desempenho Regular', 'Alto Desempenho', 'Produto Estrela']
    )
    
    return segments

@st.cache_data
def generate_advanced_recommendations(basket_sets, transactions, unique_items, min_support=0.01, min_confidence=0.5, min_lift=1):
    """Gera recomendações avançadas usando regras de associação"""
    try:
        # Gerar regras de associação
        frequent_itemsets = apriori(basket_sets, min_support=min_support, use_colnames=True)
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence, num_itemsets=len(frequent_itemsets))
        rules = rules[rules['lift'] >= min_lift]
        
        # Adicionar métricas adicionais
        rules['strength'] = rules['confidence'] * rules['lift']
        rules['support_ratio'] = rules['support'] / rules['antecedent support']
        
        # Converter frozensets para listas
        rules['antecedents'] = rules['antecedents'].apply(list)
        rules['consequents'] = rules['consequents'].apply(list)
        
        return rules.sort_values('strength', ascending=False)
    except Exception as e:
        st.error(f"Erro ao gerar recomendações: {str(e)}")
        return pd.DataFrame()

@st.cache_data
def create_network_visualization(rules, max_rules=50):
    """Cria visualização em rede das regras de associação"""
    G = nx.DiGraph()
    
    # Usar apenas as top regras para melhor visualização
    top_rules = rules.head(max_rules)
    
    # Adicionar nós e arestas
    for _, rule in top_rules.iterrows():
        for ant in rule['antecedents']:
            for cons in rule['consequents']:
                G.add_edge(
                    ant, cons,
                    weight=rule['lift'],
                    confidence=rule['confidence'],
                    support=rule['support']
                )
    
    # Calcular layout
    pos = nx.spring_layout(G, k=1, iterations=50)
    
    # Criar traces para plotly
    edge_trace = []
    node_trace = []
    
    # Adicionar arestas
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        
        edge_trace.append(
            go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                line=dict(width=edge[2]['weight'], color='#888'),
                hoverinfo='text',
                text=f"Lift: {edge[2]['weight']:.2f}<br>Confidence: {edge[2]['confidence']:.2f}",
                mode='lines'
            )
        )
    
    # Adicionar nós
    node_x = []
    node_y = []
    node_text = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)
    
    node_trace.append(
        go.Scatter(
            x=node_x,
            y=node_y,
            text=node_text,
            mode='markers+text',
            hoverinfo='text',
            marker=dict(
                size=20,
                line_width=2,
                color='lightblue'
            ),
            textposition="top center"
        )
    )
    
    # Criar figura
    fig = go.Figure(data=edge_trace + node_trace,
                   layout=go.Layout(
                       showlegend=False,
                       hovermode='closest',
                       margin=dict(b=20,l=5,r=5,t=40),
                       xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                       yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                   ))
    
    return fig

def generate_recommendation_card(rule):
    """Gera um card visual aprimorado para uma regra de associação"""
    confidence_color = 'green' if rule['confidence'] > 0.7 else 'orange' if rule['confidence'] > 0.5 else 'red'
    lift_color = 'green' if rule['lift'] > 2 else 'orange' if rule['lift'] > 1.5 else 'blue'
    
    card_html = f"""
    <div style="padding: 15px; border-radius: 10px; background-color: #f0f2f6; margin: 10px 0; border-left: 5px solid #1f77b4;">
        <div style="margin: 10px 0;">
            <strong style="color: #1f77b4;">Se o cliente comprar:</strong><br>
            <div style="background-color: white; padding: 8px; border-radius: 5px; margin: 5px 0;">
                {', '.join(rule['antecedents'])}
            </div>
            <strong style="color: #1f77b4;">Provavelmente comprará:</strong><br>
            <div style="background-color: white; padding: 8px; border-radius: 5px; margin: 5px 0;">
                {', '.join(rule['consequents'])}
            </div>
        </div>
        <div style="display: flex; justify-content: space-between; margin-top: 10px;">
            <span style="color: {confidence_color};">
                <strong>Confiança:</strong> {rule['confidence']:.2%}
            </span>
            <span style="color: {lift_color};">
                <strong>Lift:</strong> {rule['lift']:.2f}x
            </span>
            <span>
                <strong>Suporte:</strong> {rule['support']:.2%}
            </span>
        </div>
        <div style="margin-top: 5px; font-size: 0.9em; color: #666;">
            <strong>Força da Recomendação:</strong> {rule['strength']:.2f}
        </div>
    </div>
    """
    return card_html

def main():
    st.title('🛒 Super Análise de Carrinhos e Recomendações')
    
    # Sidebar com configurações avançadas
    with st.sidebar:
        st.header('⚙️ Configurações Avançadas')
        
        # Configurações de regras
        with st.expander("📊 Parâmetros de Regras", expanded=True):
            min_support = st.slider(
                'Suporte Mínimo',
                min_value=0.01,
                max_value=1.0,
                value=0.01,
                step=0.01,
                help='Frequência mínima de ocorrência dos itens'
            )
            
            min_confidence = st.slider(
                'Confiança Mínima',
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.05,
                help='Probabilidade mínima da regra ser verdadeira'
            )
            
            min_lift = st.slider(
                'Lift Mínimo',
                min_value=1.0,
                max_value=10.0,
                value=1.0,
                step=0.5,
                help='Força mínima da relação entre os itens'
            )
        
        # Filtros de visualização
        with st.expander("🎯 Filtros de Visualização", expanded=False):
            max_produtos_viz = st.number_input(
                "Máximo de Produtos na Visualização",
                min_value=5,
                max_value=50,
                value=15
            )
            
            min_freq_viz = st.number_input(
                "Frequência Mínima para Visualização",
                min_value=1,
                value=2
            )

    # Upload de arquivo
    st.markdown("""
    ### 📁 Upload de Dados
    Faça upload do seu arquivo CSV contendo os carrinhos de compras.
    Cada linha deve conter os produtos separados por vírgula.
    """)
    
    uploaded_file = st.file_uploader("Upload CSV", type="csv")
    
    if uploaded_file is not None:
        # Carregar e processar dados
        with st.spinner('Processando dados...'):
            df = load_data(uploaded_file)
            
            if df is not None:
                basket_sets, transactions, unique_items = process_transactions(df)
                
                if basket_sets is not None:
                    # Calcular métricas e gerar recomendações
                    product_metrics = calculate_product_metrics(basket_sets, transactions)
                    rules = generate_advanced_recommendations(
                        basket_sets, 
                        transactions, 
                        unique_items,
                        min_support,
                        min_confidence,
                        min_lift
                    )
                    segments = analyze_product_segments(basket_sets, product_metrics)
                    
                    # Interface principal com tabs
                    tab1, tab2, tab3, tab4, tab5 = st.tabs([
                        "📊 Dashboard",
                        "🎯 Sistema de Recomendação",
                        "📈 Análise Avançada",
                        "🔍 Explorador de Produtos",
                        "💾 Exportar Dados"
                    ])
                    
                    with tab1:
                        st.header("Dashboard Principal")
                        
                        # Métricas principais
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total de Transações", len(transactions))
                        with col2:
                            st.metric("Produtos Únicos", len(unique_items))
                        with col3:
                            st.metric("Regras Geradas", len(rules))
                        with col4:
                            st.metric("Média de Itens/Carrinho", 
                                    f"{transactions.apply(len).mean():.1f}")
                        
                        # Gráficos principais
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.subheader("Top Produtos")
                            top_products = product_metrics['frequency'].sort_values(
                                ascending=False).head(max_produtos_viz)
                            fig = px.bar(
                                x=top_products.index,
                                y=top_products.values,
                                title="Produtos Mais Frequentes",
                                labels={"x": "Produto", "y": "Frequência"}
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            st.subheader("Distribuição de Tamanhos")
                            sizes = transactions.apply(len)
                            fig = px.histogram(
                                sizes,
                                title="Distribuição do Tamanho dos Carrinhos",
                                labels={"value": "Número de Itens", "count": "Frequência"}
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    
                    with tab2:
                        st.header("Sistema Inteligente de Recomendação")
                        
                        # Sistema de busca por produto
                        selected_product = st.selectbox(
                            "🔍 Selecione um produto para ver recomendações específicas:",
                            options=unique_items
                        )
                        
                        if selected_product:
                            specific_rules = rules[rules['antecedents'].apply(
                                lambda x: selected_product in x)].sort_values('strength', ascending=False)
                            
                            if not specific_rules.empty:
                                st.subheader(f"Recomendações para: {selected_product}")
                                
                                # Mostrar top 5 recomendações em cards
                                for _, rule in specific_rules.head(5).iterrows():
                                    st.markdown(
                                        generate_recommendation_card(rule),
                                        unsafe_allow_html=True
                                    )
                                
                                # Visualização em rede
                                st.subheader("Rede de Associações")
                                network_fig = create_network_visualization(
                                    specific_rules,
                                    max_rules=10
                                )
                                st.plotly_chart(network_fig, use_container_width=True)
                            else:
                                st.info("Não foram encontradas recomendações específicas para este produto.")
                    
                    with tab3:
                        st.header("Análise Avançada")
                        
                        # Segmentação de produtos
                        st.subheader("Segmentação de Produtos")
                        fig = px.scatter(
                            segments,
                            x='frequencia',
                            y='presenca',
                            color='categoria',
                            size='score_composto',
                            hover_data=['diversidade', 'media_outros'],
                            title="Mapa de Segmentação de Produtos"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Matriz de correlação
                        st.subheader("Matriz de Correlação")
                        corr_matrix = basket_sets.corr()
                        fig = px.imshow(
                            corr_matrix,
                            title="Correlação entre Produtos",
                            aspect="auto"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with tab4:
                        st.header("Explorador de Produtos")
                        
                        # Seleção de produto para análise detalhada
                        product_to_analyze = st.selectbox(
                            "Selecione um produto para análise detalhada:",
                            options=unique_items,
                            key="product_explorer"
                        )
                        
                        if product_to_analyze:
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.metric(
                                    "Frequência Total",
                                    int(product_metrics['frequency'][product_to_analyze])
                                )
                                st.metric(
                                    "Presença em Carrinhos",
                                    f"{product_metrics['presence_percent'][product_to_analyze]:.1f}%"
                                )
                            
                            with col2:
                                st.metric(
                                    "Média de Outros Produtos",
                                    f"{product_metrics['avg_other_products'][product_to_analyze]:.1f}"
                                )
                                st.metric(
                                    "Diversidade de Combinações",
                                    int(product_metrics['combination_diversity'][product_to_analyze])
                                )
                            
                            # Produtos mais associados
                            st.subheader("Produtos Mais Associados")
                            related_products = basket_sets[basket_sets[product_to_analyze] == 1].sum()
                            related_products = related_products.sort_values(ascending=False)[1:11]
                            
                            fig = px.bar(
                                x=related_products.index,
                                y=related_products.values,
                                title=f"Top 10 Produtos Associados com {product_to_analyze}",
                                labels={"x": "Produto", "y": "Frequência de Associação"}
                            )
                            st.plotly_chart(fig, use_container_width=True)

                            # Produtos que mais aparecem juntos
                            st.subheader("Combinações Mais Frequentes")
                            
                            # Filtrar transações que contêm o produto selecionado
                            transactions_with_product = transactions[basket_sets[product_to_analyze] == 1]
                            
                            # Encontrar todas as combinações de produtos
                            combinations = []
                            for transaction in transactions_with_product:
                                if product_to_analyze in transaction:
                                    other_products = [p for p in transaction if p != product_to_analyze]
                                    for p in other_products:
                                        combinations.append((product_to_analyze, p))
                            
                            # Contar frequência das combinações
                            combination_counts = Counter(combinations)
                            
                            # Criar DataFrame com as combinações mais frequentes
                            top_combinations = pd.DataFrame(
                                [(p2, count) for (p1, p2), count in combination_counts.most_common(10)],
                                columns=['Produto', 'Frequência']
                            )
                            
                            fig = px.bar(
                                top_combinations,
                                x='Produto',
                                y='Frequência',
                                title=f"Top 10 Combinações com {product_to_analyze}",
                                labels={"Produto": "Produto", "Frequência": "Frequência de Combinação"}
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    
                    with tab5:
                        st.header("Exportar Dados")
                        
                        export_format = st.radio(
                            "Formato de Exportação",
                            ["Excel", "CSV", "JSON"]
                        )
                        
                        export_content = st.multiselect(
                            "Selecione os dados para exportar",
                            ["Métricas Básicas", "Produtos", "Regras", "Segmentação"]
                        )
                        
                        if st.button("Preparar Exportação"):
                            report_data = {
                                "Métricas Básicas": pd.DataFrame({
                                    "Métrica": ["Total de Transações", "Produtos Únicos", 
                                              "Regras Geradas", "Média de Itens/Carrinho"],
                                    "Valor": [len(transactions), len(unique_items), 
                                            len(rules), transactions.apply(len).mean()]
                                }),
                                "Produtos": pd.DataFrame(product_metrics['frequency']).reset_index(),
                                "Regras": rules,
                                "Segmentação": segments
                            }
                            
                            if export_format == "Excel":
                                output = BytesIO()
                                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                                    for content in export_content:
                                        if content in report_data:
                                            report_data[content].to_excel(
                                                writer,
                                                sheet_name=content,
                                                index=False
                                            )
                                
                                st.download_button(
                                    "📥 Download Relatório (Excel)",
                                    output.getvalue(),
                                    "analise_carrinhos.xlsx",
                                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                )
                            
                            elif export_format == "CSV":
                                for content in export_content:
                                    if content in report_data:
                                        csv = report_data[content].to_csv(index=False)
                                        st.download_button(
                                            f"📥 Download {content} (CSV)",
                                            csv,
                                            f"{content.lower()}.csv",
                                            "text/csv"
                                        )
                            
                            else:  # JSON
                                for content in export_content:
                                    if content in report_data:
                                        json_str = report_data[content].to_json(orient="records")
                                        st.download_button(
                                            f"📥 Download {content} (JSON)",
                                            json_str,
                                            f"{content.lower()}.json",
                                            "application/json"
                                        )

    # Footer
    st.markdown("---")
    st.markdown("""
        💡 **Dicas de Uso:**
        - Ajuste os parâmetros no sidebar para refinar a análise
        - Use o sistema de recomendação para descobrir padrões interessantes
        - Explore diferentes visualizações nas abas
        - Exporte os dados para análise offline
        
        📊 **Sobre a Análise:**
        - As regras de associação são baseadas no algoritmo Apriori
        - O lift indica a força da associação entre produtos
        - A confiança indica a probabilidade condicional da regra
    """)

if __name__ == "__main__":
    main()
