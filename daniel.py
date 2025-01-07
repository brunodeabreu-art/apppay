import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Estilo CSS aprimorado
ESTILO_CSS = """
/* Estilos para o Dashboard Executivo */
body {
    background-color: #f0f2f6;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

h1, h2, h3 {
    color: #2c3e50;
}

.metric {
    font-size: 2.5em;
    color: #27ae60;
    text-align: center;
}

table {
    width: 100%;
    border-collapse: collapse;
    margin-bottom: 20px;
}

th, td {
    padding: 12px;
    border: 1px solid #bdc3c7;
    text-align: left;
}

th {
    background-color: #34495e;
    color: white;
}

tr:nth-child(even) {
    background-color: #ecf0f1;
}

.expander-header {
    font-weight: bold;
    font-size: 1.1em;
    color: #34495e;
}

.axis-title {
    font-size: 14px;
    color: #2c3e50;
}

.legend-title {
    font-size: 12px;
    color: #2c3e50;
}
"""

# Função para carregar o estilo CSS
def carregar_estilo():
    st.markdown(f"<style>{ESTILO_CSS}</style>", unsafe_allow_html=True)

# Função para obter os dados de comparação
def obter_dados_comparacao():
    dados = {
        "Categoria": [
            "1. AUTOMAÇÃO BÁSICA"] * 10 + 
            ["2. AUTOMAÇÃO AVANÇADA"] * 10 +
            ["3. INTELIGÊNCIA ARTIFICIAL"] * 10 +
            ["4. GESTÃO DE DADOS"] * 10 +
            ["5. EXPERIÊNCIA DO USUÁRIO"] * 10 +
            ["6. INTEGRAÇÃO E APIs"] * 10 +
            ["6. SEGURANÇA E COMPLIANCE"] * 10 +
            ["7. ANALYTICS E RELATÓRIOS"] * 10 +
            ["8. MOBILE E DISPOSITIVOS"] * 10 +
            ["9. COLABORAÇÃO E SOCIAL"] * 10 +
            ["10. GESTÃO DE DOCUMENTOS"] * 10 +
            ["11. PERFORMANCE E ESCALABILIDADE"] * 10 +
            ["12. SUPORTE E MANUTENÇÃO"] * 10 +
            ["13. CUSTOMIZAÇÃO E DESENVOLVIMENTO"] * 10 +
            ["14. GOVERNANÇA E ADMINISTRAÇÃO"] * 10,
        "Critério": [
            # 1. AUTOMAÇÃO BÁSICA
            "Automação de tarefas simples",
            "Automação de tarefas complexas",
            "Automação de marketing",
            "Automação de vendas",
            "Automação baseada em IA",
            "Fluxos de trabalho personalizados",
            "Processos de aprovação",
            "Automação multi-departamental",
            "Automação de documentos",
            "Automação de contratos",
            # 2. AUTOMAÇÃO AVANÇADA
            "Automação de processos cross-funcionais",
            "Automação de ciclo de vida do cliente",
            "Automação de previsões",
            "Automação de relatórios",
            "Automação de atendimento",
            "Automação de campanhas",
            "Automação de social media",
            "Automação de email marketing",
            "Automação de lead scoring",
            "Automação de territory management",
            # 3. INTELIGÊNCIA ARTIFICIAL
            "Previsão de vendas",
            "Score de leads automático",
            "Recomendações de ações",
            "Análise de sentimento",
            "Insights preditivos",
            "Reconhecimento de padrões",
            "IA para atendimento",
            "IA para marketing",
            "IA para vendas",
            "IA para análise de dados",
            # 4. GESTÃO DE DADOS
            "Qualidade de dados",
            "Deduplicação",
            "Enriquecimento de dados",
            "Normalização",
            "Validação",
            "Segmentação",
            "Histórico de mudanças",
            "Backup automático",
            "Recuperação de dados",
            "Arquivamento",
            # 5. EXPERIÊNCIA DO USUÁRIO
            "Interface intuitiva",
            "Personalização de interface",
            "Acessibilidade",
            "Responsividade",
            "Performance",
            "Facilidade de uso",
            "Curva de aprendizado",
            "Customização de views",
            "Temas e layouts",
            "Atalhos e produtividade",
            # 6. INTEGRAÇÃO E APIs
            "API REST",
            "API SOAP",
            "Webhooks",
            "Integrações nativas",
            "Conectores de terceiros",
            "ETL capabilities",
            "Sincronização em tempo real",
            "Middleware support",
            "API rate limits",
            "API documentation",
            # 6. SEGURANÇA E COMPLIANCE
            "Criptografia de dados",
            "Criptografia em nível de campo",
            "Certificações (ISO, SOC)",
            "Conformidade com GDPR",
            "Conformidade com HIPAA",
            "Auditoria de acessos",
            "Logs detalhados",
            "Controle de permissões",
            "Backup multi-região",
            "Recuperação de desastres",
            # 7. ANALYTICS E RELATÓRIOS
            "Relatórios básicos",
            "Relatórios avançados",
            "Dashboards dinâmicos",
            "Análise preditiva",
            "Análise de tendências",
            "Personalização de relatórios",
            "Exportação de dados",
            "Visualização de dados",
            "Big Data Analytics",
            "Integração com Tableau",
            # 8. MOBILE E DISPOSITIVOS
            "App móvel",
            "Funcionalidades offline",
            "Sincronização automática",
            "Personalização no app",
            "Geolocalização",
            "Notificações push",
            "Mobile analytics",
            "Suporte a múltiplos dispositivos",
            "Performance no app",
            "Atualizações automáticas",
            # 9. COLABORAÇÃO E SOCIAL
            "Chat interno",
            "Feed social",
            "Integração com Slack",
            "Videoconferência",
            "Compartilhamento de arquivos",
            "Comentários em tarefas",
            "Gestão de equipes",
            "Colaboração em tempo real",
            "Integração com redes sociais",
            "Gamificação",
            # 10. GESTÃO DE DOCUMENTOS
            "Armazenamento de documentos",
            "Controle de versão",
            "Assinatura eletrônica",
            "Pesquisa de documentos",
            "Compartilhamento seguro",
            "Templates de documentos",
            "Automação de contratos",
            "Integração com Google Drive",
            "Integração com OneDrive",
            "Integração com SharePoint",
            # 11. PERFORMANCE E ESCALABILIDADE
            "Suporte a grandes volumes de dados",
            "Performance em alta carga",
            "Escalabilidade horizontal",
            "Escalabilidade vertical",
            "Tempo de resposta do sistema",
            "Suporte a múltiplos idiomas",
            "Suporte a múltiplas moedas",
            "Performance em dispositivos móveis",
            "Redundância de dados",
            "Tolerância a falhas",
            # 12. SUPORTE E MANUTENÇÃO
            "Suporte técnico 24/7",
            "Base de conhecimento",
            "Comunidade de usuários",
            "Consultoria especializada",
            "Atualizações automáticas",
            "Frequência de atualizações",
            "Documentação técnica",
            "Treinamento online",
            "Certificações para usuários",
            "Suporte em múltiplos idiomas",
            # 13. CUSTOMIZAÇÃO E DESENVOLVIMENTO
            "Campos customizados",
            "Fluxos customizados",
            "Objetos personalizados",
            "Templates",
            "Desenvolvimento de apps",
            "Marketplace",
            "Ferramentas de desenvolvimento",
            "Integração com IDEs",
            "Suporte a linguagens de programação",
            "Testes automatizados",
            # 14. GOVERNANÇA E ADMINISTRAÇÃO
            "Controle de permissões",
            "Logs de auditoria",
            "Monitoramento de atividades",
            "Gestão de usuários",
            "Gestão de perfis",
            "Políticas de segurança",
            "Configuração de regras",
            "Relatórios de conformidade",
            "Gestão de licenças",
            "Administração centralizada"
        ],
        "Bitrix24": [
            # 1. AUTOMAÇÃO BÁSICA
            "8",
            "5",
            "6",
            "7",
            "3",
            "6",
            "7",
            "4",
            "6",
            "5",
            # 2. AUTOMAÇÃO AVANÇADA
            "4",
            "5",
            "3",
            "6",
            "5",
            "6",
            "7",
            "7",
            "4",
            "3",
            # 3. INTELIGÊNCIA ARTIFICIAL
            "2",
            "3",
            "2",
            "1",
            "2",
            "1",
            "2",
            "2",
            "2",
            "1",
            # 4. GESTÃO DE DADOS
            "6",
            "5",
            "4",
            "5",
            "6",
            "5",
            "3",
            "7",
            "6",
            "5",
            # 5. EXPERIÊNCIA DO USUÁRIO
            "8",
            "6",
            "7",
            "7",
            "7",
            "8",
            "8",
            "6",
            "7",
            "6",
            # 6. INTEGRAÇÃO E APIs
            "4",
            "4",
            "5",
            "5",
            "6",
            "3",
            "5",
            "3",
            "5",
            "6",
            # 6. SEGURANÇA E COMPLIANCE
            "7",
            "2",
            "6",
            "7",
            "4",
            "3",
            "3",
            "4",
            "5",
            "4",
            # 7. ANALYTICS E RELATÓRIOS
            "8",
            "5",
            "6",
            "2",
            "4",
            "6",
            "3",
            "6",
            "3",
            "2",
            # 8. MOBILE E DISPOSITIVOS
            "8",
            "5",
            "6",
            "4",
            "5",
            "7",
            "3",
            "7",
            "6",
            "6",
            # 9. COLABORAÇÃO E SOCIAL
            "9",
            "8",
            "4",
            "5",
            "6",
            "8",
            "4",
            "6",
            "3",
            "5",
            # 10. GESTÃO DE DOCUMENTOS
            "7",
            "5",
            "5",
            "7",
            "5",
            "7",
            "5",
            "8",
            "8",
            "4",
            # 11. PERFORMANCE E ESCALABILIDADE
            "5",
            "6",
            "4",
            "5",
            "7",
            "8",
            "6",
            "6",
            "5",
            "4",
            # 12. SUPORTE E MANUTENÇÃO
            "4",
            "6",
            "5",
            "3",
            "6",
            "5",
            "6",
            "5",
            "3",
            "6",
            # 13. CUSTOMIZAÇÃO E DESENVOLVIMENTO
            "4",
            "5",
            "3",
            "8",
            "3",
            "4",
            "4",
            "2",
            "2",
            "2",
            # 14. GOVERNANÇA E ADMINISTRAÇÃO
            "4",
            "4",
            "2",
            "3",
            "4",
            "3",
            "5",
            "4",
            "6",
            "5"
        ],
        "Salesforce": [
            # 1. AUTOMAÇÃO BÁSICA
            "10",
            "10",
            "9",
            "10",
            "10",
            "10",
            "10",
            "10",
            "9",
            "9",
            # 2. AUTOMAÇÃO AVANÇADA
            "10",
            "10",
            "10",
            "10",
            "9",
            "10",
            "9",
            "10",
            "10",
            "9",
            # 3. INTELIGÊNCIA ARTIFICIAL
            "10",
            "10",
            "10",
            "9",
            "10",
            "9",
            "10",
            "10",
            "10",
            "10",
            # 4. GESTÃO DE DADOS
            "10",
            "9",
            "10",
            "9",
            "10",
            "10",
            "10",
            "10",
            "10",
            "9",
            # 5. EXPERIÊNCIA DO USUÁRIO
            "9",
            "10",
            "10",
            "10",
            "9",
            "8",
            "6",
            "10",
            "9",
            "9",
            # 6. INTEGRAÇÃO E APIs
            "10",
            "10",
            "10",
            "10",
            "10",
            "9",
            "10",
            "10",
            "9",
            "10",
            # 6. SEGURANÇA E COMPLIANCE
            "10",
            "10",
            "10",
            "10",
            "10",
            "10",
            "10",
            "10",
            "10",
            "10",
            # 7. ANALYTICS E RELATÓRIOS
            "10",
            "10",
            "10",
            "10",
            "10",
            "10",
            "10",
            "10",
            "10",
            "10",
            # 8. MOBILE E DISPOSITIVOS
            "10",
            "10",
            "10",
            "10",
            "9",
            "10",
            "10",
            "10",
            "10",
            "10",
            # 9. COLABORAÇÃO E SOCIAL
            "8",
            "9",
            "10",
            "9",
            "10",
            "10",
            "10",
            "10",
            "10",
            "9",
            # 10. GESTÃO DE DOCUMENTOS
            "10",
            "10",
            "9",
            "10",
            "10",
            "10",
            "10",
            "10",
            "10",
            "10",
            # 11. PERFORMANCE E ESCALABILIDADE
            "10",
            "10",
            "10",
            "10",
            "10",
            "10",
            "10",
            "10",
            "10",
            "10",
            # 12. SUPORTE E MANUTENÇÃO
            "10",
            "10",
            "10",
            "10",
            "10",
            "10",
            "10",
            "10",
            "10",
            "10",
            # 13. CUSTOMIZAÇÃO E DESENVOLVIMENTO
            "10",
            "10",
            "10",
            "10",
            "10",
            "10",
            "10",
            "10",
            "10",
            "10",
            # 14. GOVERNANÇA E ADMINISTRAÇÃO
            "10",
            "10",
            "10",
            "10",
            "10",
            "10",
            "10",
            "10",
            "10",
            "10"
        ]
    }

    df = pd.DataFrame(dados)
    return df

# Função para aplicar formatação condicional nas pontuações
def aplicar_formatacao(row):
    try:
        bitrix_score = int(row['Bitrix24'])
        salesforce_score = int(row['Salesforce'])
    except:
        return [''] * 2

    # Definição das cores com base na pontuação
    bitrix_color = 'green' if bitrix_score >= 8 else 'orange' if bitrix_score >=5 else 'red'
    salesforce_color = 'green' if salesforce_score >= 8 else 'orange' if salesforce_score >=5 else 'red'

    return [
        f'<span style="color:{bitrix_color}; font-weight:bold">{bitrix_score}/10</span>',
        f'<span style="color:{salesforce_color}; font-weight:bold">{salesforce_score}/10</span>'
    ]

def main():
    st.set_page_config(page_title="Dashboard Executivo", layout="wide")
    carregar_estilo()

    st.title("Comparativo de Plataformas")

    # Seção 1: Visão Geral
    st.header("Visão Geral")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Salesforce")
        st.metric(label="Média Global", value="9.9")

    with col2:
        st.subheader("Bitrix24")
        st.metric(label="Média Global", value="5.6")

    # Seção 2: Metodologia de Avaliação
    st.header("Metodologia de Avaliação")
    st.markdown("""
    **Critérios de Pontuação**
    - **10-9:** Excelente - Funcionalidade completa e avançada
    - **8-7:** Muito Bom - Atende a maioria das necessidades empresariais
    - **6-5:** Bom - Funcionalidade básica presente
    - **4-3:** Regular - Funcionalidade limitada
    - **2-1:** Fraco - Funcionalidade muito básica ou ausente
    """)

    # Seção 3: Detalhamento das Categorias e Critérios
    st.header("Detalhamento das Categorias e Critérios")

    dados_comparacao = obter_dados_comparacao()

    # Aplicar formatação condicional nas pontuações
    dados_comparacao[['Bitrix24', 'Salesforce']] = dados_comparacao.apply(aplicar_formatacao, axis=1, result_type='expand')

    categorias = dados_comparacao['Categoria'].unique()

    for categoria in categorias:
        # Utiliza Expander para cada categoria para melhorar a navegação
        with st.expander(categoria, expanded=False):
            df_categoria = dados_comparacao[dados_comparacao['Categoria'] == categoria][['Critério', 'Bitrix24', 'Salesforce']]
            st.markdown(
                df_categoria.to_html(escape=False, index=False),
                unsafe_allow_html=True
            )

    # Seção 4: Gráficos de Análise
    # st.header("Análise Visual")

    # Preparar dados para os gráficos
    # Remover categorias duplicadas
    # dados_para_graficos = dados_comparacao.copy()
    # dados_para_graficos['Bitrix24'] = dados_para_graficos['Bitrix24'].str.replace('/10', '').astype(int)
    # dados_para_graficos['Salesforce'] = dados_para_graficos['Salesforce'].str.replace('/10', '').astype(int)

    # Gráfico de Barras Comparativo das Médias por Categoria
    # st.subheader("Média de Pontuação por Categoria")
    # media_por_categoria = dados_para_graficos.groupby('Categoria')[['Bitrix24', 'Salesforce']].mean().reset_index()
    # media_por_categoria_melted = media_por_categoria.melt(id_vars='Categoria', var_name='Plataforma', value_name='Média')

    # plt.figure(figsize=(14, 7))
    # sns.barplot(data=media_por_categoria_melted, x='Categoria', y='Média', hue='Plataforma')
    # plt.xticks(rotation=45, ha='right')
    # plt.xlabel("Categoria")
    # plt.ylabel("Média de Pontuação")
    # plt.title("Comparação das Médias por Categoria")
    # plt.legend(title='Plataforma')
    # st.pyplot(plt)

    # Gráfico de Radar para Comparação Geral
    # st.subheader("Comparação Radar das Pontuações")
    # categorias_unicas = media_por_categoria['Categoria'].tolist()
    # categorias_unicas += categorias_unicas[:1]  # Fechar o radar

    # valores_bitrix = media_por_categoria['Bitrix24'].tolist()
    # valores_bitrix += valores_bitrix[:1]

    # valores_salesforce = media_por_categoria['Salesforce'].tolist()
    # valores_salesforce += valores_salesforce[:1]

    # angles = [n / float(len(categorias_unicas)) * 2 * 3.141592653589793 for n in range(len(categorias_unicas))]

    # plt.figure(figsize=(8, 8))
    # ax = plt.subplot(111, polar=True)
    # plt.xticks(angles[:-1], categorias_unicas[:-1], color='grey', size=8)

    # ax.plot(angles, valores_bitrix, linewidth=1, linestyle='solid', label="Bitrix24")
    # ax.fill(angles, valores_bitrix, alpha=0.1)

    # ax.plot(angles, valores_salesforce, linewidth=1, linestyle='solid', label="Salesforce")
    # ax.fill(angles, valores_salesforce, alpha=0.1)

    # plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    # st.pyplot(plt)

    # Gráfico de Distribuição de Pontuações
    # st.subheader("Distribuição das Pontuações")
    # plt.figure(figsize=(14, 7))
    # sns.histplot(dados_para_graficos['Bitrix24'], color='green', label='Bitrix24', kde=True, alpha=0.6)
    # sns.histplot(dados_para_graficos['Salesforce'], color='blue', label='Salesforce', kde=True, alpha=0.6)
    # plt.xlabel("Pontuação")
    # plt.ylabel("Frequência")
    # plt.title("Distribuição das Pontuações - Bitrix24 vs Salesforce")
    # plt.legend()
    # st.pyplot(plt)

if __name__ == "__main__":
    main()
