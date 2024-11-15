import streamlit as st
import folium
from folium import IFrame
from streamlit_folium import st_folium
import pandas as pd
from openai import OpenAI
import os
from dotenv import load_dotenv

# Carregar variáveis de ambiente
load_dotenv()

# Inicializar cliente OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Inicialização do estado da sessão
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'output' not in st.session_state:
    st.session_state.output = ''

if 'error' not in st.session_state:
    st.session_state.error = None

# Configurar a página
st.set_page_config(
    page_title="São Francisco Explorer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dados dos pontos turísticos
pontos_turisticos = {
    "Golden Gate Bridge": {
        "lat": 37.8199,
        "lon": -122.4783,
        "desc": "Icônica ponte vermelha, símbolo da cidade",
        "img": "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0c/GoldenGateBridge-001.jpg/320px-GoldenGateBridge-001.jpg",
        "categoria": ["dia", "pôr do sol"],
        "horario": "24 horas",
        "melhor_hora": "Nascer ou pôr do sol"
    },
    "Alcatraz": {
        "lat": 37.8270,
        "lon": -122.4230,
        "desc": "Antiga prisão federal em uma ilha",
        "img": "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0a/Alcatraz_Island_photo_D_Ramey_Logan.jpg/320px-Alcatraz_Island_photo_D_Ramey_Logan.jpg",
        "categoria": ["dia"],
        "horario": "09:00 - 16:30",
        "melhor_hora": "Manhã"
    },
    "Fisherman's Wharf": {
        "lat": 37.8080,
        "lon": -122.4177,
        "desc": "Área histórica à beira-mar com restaurantes e lojas",
        "img": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/38/Fishermans_Wharf_Sign.jpg/320px-Fishermans_Wharf_Sign.jpg",
        "categoria": ["dia", "noite"],
        "horario": "10:00 - 22:00",
        "melhor_hora": "Tarde"
    },
    "Lombard Street": {
        "lat": 37.8021,
        "lon": -122.4187,
        "desc": "Famosa rua sinuosa com oito curvas",
        "img": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4a/Lombard_Street_SF.jpg/320px-Lombard_Street_SF.jpg",
        "categoria": ["dia"],
        "horario": "24 horas",
        "melhor_hora": "Manhã"
    },
    "Chinatown": {
        "lat": 37.7941,
        "lon": -122.4078,
        "desc": "Maior Chinatown fora da Ásia",
        "img": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4b/SF_Chinatown_Dragon_Gate.jpg/320px-SF_Chinatown_Dragon_Gate.jpg",
        "categoria": ["dia", "noite"],
        "horario": "10:00 - 21:00",
        "melhor_hora": "Tarde"
    }
}

# Remover o mapeamento antigo e usar diretamente as chaves
roteiros = {
    "2 dias": {
        "Dia 1": {
            "Manhã": ["Golden Gate Bridge", "Presidio"],
            "Tarde": ["Fisherman's Wharf", "Pier 39"],
            "Noite": ["Ghirardelli Square", "North Beach"]
        },
        "Dia 2": {
            "Manhã": ["Alcatraz"],
            "Tarde": ["Chinatown", "Union Square"],
            "Noite": ["Nob Hill", "Top of the Mark"]
        }
    },
    "7 dias": {
        "Dia 1": {
            "Manhã": ["Golden Gate Bridge"],
            "Tarde": ["Presidio", "Palace of Fine Arts"],
            "Noite": ["Marina District"]
        },
        "Dia 2": {
            "Manhã": ["Alcatraz"],
            "Tarde": ["Fisherman's Wharf", "Pier 39"],
            "Noite": ["Ghirardelli Square"]
        },
        "Dia 3": {
            "Manhã": ["Chinatown"],
            "Tarde": ["Union Square", "SoMa"],
            "Noite": ["North Beach"]
        },
        "Dia 4": {
            "Manhã": ["Golden Gate Park"],
            "Tarde": ["California Academy of Sciences"],
            "Noite": ["Haight-Ashbury"]
        },
        "Dia 5": {
            "Manhã": ["Twin Peaks"],
            "Tarde": ["Mission District"],
            "Noite": ["Castro District"]
        },
        "Dia 6": {
            "Manhã": ["Muir Woods"],
            "Tarde": ["Sausalito"],
            "Noite": ["Tiburon"]
        },
        "Dia 7": {
            "Manhã": ["Lombard Street"],
            "Tarde": ["Coit Tower"],
            "Noite": ["Top of the Mark"]
        }
    },
    "14 dias": {
        "Dia 1": {
            "Manhã": ["Golden Gate Bridge"],
            "Tarde": ["Presidio"],
            "Noite": ["Marina District"]
        }
    },
    "20 dias": {
        "Dia 1": {
            "Manhã": ["Golden Gate Bridge"],
            "Tarde": ["Presidio"],
            "Noite": ["Marina District"]
        }
    }
}

# Função para obter o roteiro baseado na duração
def get_roteiro(duracao):
    return roteiros[duracao]

# Função para criar o mapa interativo
def criar_mapa():
    m = folium.Map(
        location=[37.7749, -122.4194],
        zoom_start=13,
        tiles="CartoDB positron"
    )
    
    for nome, info in pontos_turisticos.items():
        html = f"""
            <div style="width:300px">
                <h4>{nome}</h4>
                <img src="{info['img']}" width="100%">
                <p>{info['desc']}</p>
            </div>
        """
        iframe = IFrame(html=html, width=320, height=280)
        popup = folium.Popup(iframe)
        
        folium.Marker(
            [info['lat'], info['lon']],
            popup=popup,
            tooltip=nome,
            icon=folium.Icon(color='red', icon='info-sign')
        ).add_to(m)
    
    return m

# Função para o assistente virtual
def get_assistant_response(prompt):
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Você é um guia turístico especialista em São Francisco, conhecendo profundamente todos os pontos turísticos, história e cultura da cidade."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Erro ao processar sua pergunta: {str(e)}"

# Estilo personalizado
st.markdown("""
<style>
    .main {
        padding: 0rem 1rem;
    }
    .block-container {
        padding-top: 2rem;
    }
    .css-1d391kg {
        padding: 1rem 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: white;
        border-radius: 5px;
        color: #0f52ba;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background-color: #0f52ba !important;
        color: white !important;
    }
    .chat-container {
        border: 1px solid #e6e6e6;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Interface principal
st.title("🌉 São Francisco Explorer")

# Layout principal com três colunas
col_mapa, col_chat, col_info = st.columns([2, 1, 1])

with col_mapa:
    st.subheader("Mapa Interativo")
    mapa = criar_mapa()
    st_folium(mapa, width=800, height=500)

with col_chat:
    st.subheader("💬 Assistente Virtual")
    with st.container():
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Chat container com scroll
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.messages[-5:]:  # Mostrar últimas 5 mensagens
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

        # Input do chat
        if prompt := st.chat_input("Pergunte sobre São Francisco..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                response = get_assistant_response(prompt)
                st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

with col_info:
    st.subheader("ℹ️ Informações Rápidas")
    with st.expander("🌤️ Clima Atual", expanded=True):
        st.write("22°C - Ensolarado")
        st.progress(0.7, "Condições para turismo: Ótimas")
    
    with st.expander("🎫 Ingressos Populares"):
        st.write("- Alcatraz: $41.00")
        st.write("- CityPASS: $76.00")
        st.write("- Cable Car: $8.00")

# Tabs para conteúdo adicional
tab1, tab2 = st.tabs(["📍 Roteiros Detalhados", "🌟 Atrações"])

with tab1:
    col_filtros, col_roteiro = st.columns([1, 3])
    
    with col_filtros:
        duracao = st.selectbox(
            "Duração da Viagem",
            list(roteiros.keys())  # Usar diretamente as chaves do dicionário
        )
        
        tipo_roteiro = st.multiselect(
            "Tipo de Roteiro",
            ["Turístico", "Cultural", "Gastronômico", "Noturno"],
            default=["Turístico"]
        )
        
        orcamento = st.slider(
            "Orçamento Diário ($)",
            50, 500, 200
        )

    with col_roteiro:
        try:
            roteiro_selecionado = roteiros[duracao]  # Acessar diretamente o roteiro
            
            for dia, atividades in roteiro_selecionado.items():
                with st.expander(f"📅 {dia}", expanded=True):
                    cols = st.columns(3)
                    for periodo, col in zip(["Manhã", "Tarde", "Noite"], cols):
                        with col:
                            st.markdown(f"**{periodo}**")
                            for local in atividades[periodo]:
                                info = pontos_turisticos.get(local, {})
                                st.write(f"- {local}")
                                if info:
                                    st.image(info['img'], width=150)
                                    st.caption(info['desc'])
        except Exception as e:
            st.error(f"Erro ao carregar roteiro: {str(e)}")
            st.write("Por favor, tente novamente ou selecione outro roteiro.")

with tab2:
    # Filtros para atrações
    col_filtros_atracoes, col_lista_atracoes = st.columns([1, 3])
    
    with col_filtros_atracoes:
        categoria = st.multiselect(
            "Categorias",
            ["Todos", "Dia", "Noite", "Família", "Cultura", "Natureza"],
            default=["Todos"]
        )
        
        preco = st.select_slider(
            "Faixa de Preço",
            options=["$", "$$", "$$$", "$$$$"],
            value="$$"
        )

    with col_lista_atracoes:
        for nome, info in pontos_turisticos.items():
            if "Todos" in categoria or any(cat.lower() in [c.lower() for c in categoria] for cat in info.get('categoria', [])):
                with st.container():
                    col_img, col_desc = st.columns([1, 2])
                    with col_img:
                        st.image(info.get('img', ''), width=200)
                    with col_desc:
                        st.subheader(nome)
                        st.write(info.get('desc', ''))
                        st.write(f"🕒 Horário: {info.get('horario', 'Não informado')}")
                        st.write(f"✨ Melhor momento: {info.get('melhor_hora', 'Não informado')}")
                st.divider()

# Sidebar atualizado
with st.sidebar:
    st.header("📱 Menu Rápido")
    
    # Clima
    st.subheader("🌤️ Previsão 5 dias")
    for i in range(5):
        st.write(f"Dia {i+1}: 18°C - 22°C")
    
    # Transportes
    st.subheader("🚌 Transporte")
    opcoes_transporte = {
        "BART": "Sistema de metrô rápido",
        "Muni": "Ônibus e bondes locais",
        "Cable Cars": "Bondes históricos"
    }
    for tipo, desc in opcoes_transporte.items():
        with st.expander(tipo):
            st.write(desc)
    
    # Dicas
    st.subheader("💡 Dicas do Dia")
    st.info("Hoje é ótimo para visitar Alcatraz! Reserve com antecedência.")
