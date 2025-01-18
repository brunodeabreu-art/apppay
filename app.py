import streamlit as st
from openai import OpenAI
from datetime import datetime, timedelta
import calendar
from dotenv import load_dotenv
import os
import requests
import random
import time

# Carrega as variáveis de ambiente
load_dotenv()

# Inicializa cliente OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def buscar_imagem(tema):
    """
    Busca uma imagem relacionada ao tema usando múltiplas fontes
    """
    try:
        # Lista de imagens de alta qualidade para fallback por categoria
        imagens_fallback = {
            "marketing": [
                "https://images.unsplash.com/photo-1432888622747-4eb9a8f1fafd",
                "https://images.unsplash.com/photo-1460925895917-afdab827c52f",
                "https://images.unsplash.com/photo-1611926653458-09294b3142bf"
            ],
            "business": [
                "https://images.unsplash.com/photo-1486406146926-c627a92ad1ab",
                "https://images.unsplash.com/photo-1507679799987-c73779587ccf",
                "https://images.unsplash.com/photo-1573164713988-8665fc963095"
            ],
            "social": [
                "https://images.unsplash.com/photo-1611926653458-09294b3142bf",
                "https://images.unsplash.com/photo-1562577309-4932fdd64cd1",
                "https://images.unsplash.com/photo-1527689368864-3a821dbccc34"
            ],
            "technology": [
                "https://images.unsplash.com/photo-1518770660439-4636190af475",
                "https://images.unsplash.com/photo-1488590528505-98d2b5aba04b",
                "https://images.unsplash.com/photo-1451187580459-43490279c0fa"
            ],
            "default": [
                "https://images.unsplash.com/photo-1557426272-fc759fdf7a8d",
                "https://images.unsplash.com/photo-1512314889357-e157c22f938d",
                "https://images.unsplash.com/photo-1504868584819-f8e8b4b6d7e3"
            ]
        }

        # Determina a categoria baseada no tema
        tema_lower = tema.lower()
        categoria = "default"
        for key in imagens_fallback.keys():
            if key in tema_lower:
                categoria = key
                break

        # Seleciona uma imagem aleatória da categoria
        imagem_url = random.choice(imagens_fallback[categoria])

        # Adiciona parâmetros para evitar cache
        timestamp = int(time.time())
        imagem_url = f"{imagem_url}?v={timestamp}"

        # Verifica se a URL está acessível
        response = requests.head(imagem_url, timeout=5)
        if response.status_code == 200:
            return imagem_url

        # Se falhar, tenta uma imagem da categoria default
        return random.choice(imagens_fallback["default"])

    except Exception as e:
        st.error(f"Erro ao buscar imagem: {str(e)}")
        # Retorna uma imagem default garantida
        return "https://images.unsplash.com/photo-1557426272-fc759fdf7a8d"

def gerar_conteudo(tema, rede_social, dia_semana):
    """
    Função para gerar conteúdo específico para cada rede social com validação
    """
    try:
        prompts = {
            "Instagram": f"""Crie uma publicação envolvente para Instagram sobre {tema}, considerando que é {dia_semana}.
            A resposta DEVE seguir EXATAMENTE esta estrutura:
            [LEGENDA]
            (texto envolvente e humanizado, máximo 2200 caracteres)
            
            [HASHTAGS]
            (10-15 hashtags relevantes)""",
            
            "LinkedIn": f"""Crie uma publicação profissional para LinkedIn sobre {tema}, considerando que é {dia_semana}.
            A resposta DEVE seguir EXATAMENTE esta estrutura:
            [TÍTULO]
            (título chamativo)
            
            [CONTEÚDO]
            (texto profissional e informativo)
            
            [HASHTAGS]
            (5-7 hashtags profissionais)""",
            
            "Facebook": f"""Crie uma publicação informal para Facebook sobre {tema}, considerando que é {dia_semana}.
            A resposta DEVE seguir EXATAMENTE esta estrutura:
            [CONTEÚDO]
            (texto conversacional com emojis)
            
            [CALL-TO-ACTION]
            (chamada clara para ação)
            
            [HASHTAGS]
            (3-5 hashtags relevantes)"""
        }
        
        resposta = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Você é um especialista em marketing digital com foco em conteúdo humanizado e engajador. Sempre formate o texto adequadamente e use emojis apropriados."},
                {"role": "user", "content": prompts[rede_social]}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        
        conteudo = resposta.choices[0].message.content.strip()
        
        # Validação do conteúdo
        if len(conteudo) < 10:
            raise Exception("Conteúdo gerado muito curto")
            
        return conteudo
    except Exception as e:
        return f"Erro ao gerar conteúdo: {str(e)}"

def criar_calendario_posts(inicio_data, tema):
    """
    Cria um calendário de posts para 15 dias
    """
    calendario = []
    redes_sociais = ["Instagram", "LinkedIn", "Facebook"]
    
    for i in range(15):
        data = inicio_data + timedelta(days=i)
        dia_semana = calendar.day_name[data.weekday()]
        
        # Alterna entre redes sociais
        rede_social = redes_sociais[i % len(redes_sociais)]
        
        calendario.append({
            "data": data.strftime("%d/%m/%Y"),
            "dia_semana": dia_semana,
            "rede_social": rede_social,
            "tema": tema
        })
    
    return calendario

def main():
    st.set_page_config(page_title="Planejador de Conteúdo Social", 
                      page_icon="📊",
                      layout="wide",
                      initial_sidebar_state="collapsed")  # Sidebar recolhida por padrão
    
    # Estilo personalizado atualizado
    st.markdown("""
        <style>
        .calendar-card {
            background-color: #ffffff;
            border-radius: 8px;
            padding: 12px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            border: 1px solid #eee;
            max-width: 400px;
            margin: 8px auto;
        }
        .post-header {
            display: flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 12px;
        }
        .date-info {
            font-size: 0.8em;
            color: #666;
            flex: 1;
        }
        .network-badge {
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.7em;
            font-weight: 500;
            color: #fff;
        }
        .instagram-badge {
            background-color: #E1306C;
        }
        .linkedin-badge {
            background-color: #0077B5;
        }
        .facebook-badge {
            background-color: #4267B2;
        }
        .post-image {
            width: 100%;
            height: 160px;
            object-fit: cover;
            border-radius: 4px;
            margin-bottom: 12px;
        }
        .content-preview {
            font-size: 0.85em;
            line-height: 1.5;
            color: #333;
            max-height: 120px;
            overflow-y: auto;
            margin-bottom: 12px;
            padding-right: 8px;
        }
        .content-preview::-webkit-scrollbar {
            width: 4px;
        }
        .content-preview::-webkit-scrollbar-thumb {
            background-color: #ddd;
            border-radius: 4px;
        }
        .metrics-container {
            display: flex;
            justify-content: space-between;
            font-size: 0.75em;
            color: #666;
            padding-top: 8px;
            border-top: 1px solid #eee;
        }
        .stMarkdown {
            max-width: 100% !important;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.title("📊 Planejador Estratégico de Conteúdo Digital")
    
    with st.sidebar:
        st.header("⚙️ Configurações do Planejamento")
        tema = st.text_input("Tema Principal:", placeholder="Ex: Marketing Digital")
        data_inicio = st.date_input("Data de Início da Campanha:", datetime.now())
        
        metricas = st.expander("📈 Métricas Sugeridas")
        with metricas:
            st.write("""
            - Engajamento médio por post
            - Alcance orgânico
            - Taxa de conversão
            - Crescimento de seguidores
            """)
        
        if st.button("🎯 Gerar Plano Estratégico", use_container_width=True):
            if tema:
                with st.spinner("Gerando plano estratégico de conteúdo..."):
                    calendario = criar_calendario_posts(data_inicio, tema)
                    st.session_state['calendario'] = calendario
                    st.session_state['posts_gerados'] = {}
                    
                    # Gerar conteúdo para todos os posts automaticamente
                    with st.spinner("Gerando conteúdo para todas as publicações..."):
                        for idx, post in enumerate(calendario):
                            conteudo = gerar_conteudo(post['tema'], 
                                                    post['rede_social'],
                                                    post['dia_semana'])
                            imagem_url = buscar_imagem(post['tema'])
                            st.session_state['posts_gerados'][idx] = {
                                'conteudo': conteudo,
                                'imagem': imagem_url
                            }
                        st.success("✅ Plano de conteúdo gerado com sucesso!")
            else:
                st.warning("⚠️ Por favor, defina o tema principal da campanha.")
    
    if 'calendario' in st.session_state:
        # Visualização em formato de calendário mensal
        mes_atual = data_inicio.strftime("%B %Y")
        st.subheader(f"📅 Calendário Editorial - {mes_atual}")
        
        # Organizar posts por semana
        semanas = {}
        for idx, post in enumerate(st.session_state['calendario']):
            # Convertendo data_inicio para datetime
            data_inicio_dt = datetime.combine(data_inicio, datetime.min.time())
            post_date = datetime.strptime(post['data'], "%d/%m/%Y")
            semana = (post_date - data_inicio_dt).days // 7
            if semana not in semanas:
                semanas[semana] = []
            semanas[semana].append((idx, post))
        
        # Exibir calendário por semanas
        for semana, posts in semanas.items():
            st.subheader(f"Semana {semana + 1}")
            
            # Dividir posts em grupos de 3
            for i in range(0, len(posts), 3):
                grupo_posts = posts[i:i+3]
                cols = st.columns(3)
                
                for col, (idx, post) in zip(cols, grupo_posts):
                    with col:
                        with st.container():
                            if idx in st.session_state['posts_gerados']:
                                post_data = st.session_state['posts_gerados'][idx]
                                st.markdown(f"""
                                <div class="calendar-card">
                                    <div class="post-header">
                                        <div class="date-info">
                                            {post['data']} • {post['dia_semana']}
                                        </div>
                                        <span class="network-badge {post['rede_social'].lower()}-badge">
                                            {post['rede_social']}
                                        </span>
                                    </div>
                                """, unsafe_allow_html=True)
                                
                                if post_data['imagem']:
                                    st.image(post_data['imagem'], 
                                           use_column_width=True,
                                           output_format="PNG")
                                
                                conteudo_formatado = post_data['conteudo'].replace('\n', '<br/>')
                                st.markdown(f"""
                                    <div class="content-preview">
                                        {conteudo_formatado}
                                    </div>
                                    <div class="metrics-container">
                                        <span>📈 2.5k visualizações</span>
                                        <span>💬 4.2% engajamento</span>
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
    
    # Rodapé com informações técnicas
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center'>
            <p>Desenvolvido com tecnologias avançadas de IA</p>
            <small>Utilizando OpenAI GPT-3.5 para geração de conteúdo e APIs de imagem para recursos visuais</small>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 
