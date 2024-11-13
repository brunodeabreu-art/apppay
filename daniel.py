import streamlit as st
from PIL import Image
import requests
from io import BytesIO

# Configuração inicial com tema escuro e minimalista
st.set_page_config(
    page_title="Capivara Assessoria | Cultura, Tecnologia & Urbanismo",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS atualizado para landing page profissional
st.markdown("""
    <style>
        /* Reset e variáveis */
        :root {
            --primary: #0A0A0A;
            --secondary: #1E3329;
            --accent: #FF4B36;
            --text: #FFFFFF;
            --spacing: 120px;
        }

        /* Estilos gerais */
        .stApp {
            background: var(--primary);
            color: var(--text);
        }

        /* Hero Section */
        .hero-section {
            height: 90vh;
            position: relative;
            display: flex;
            align-items: center;
            margin: -80px -80px 0 -80px;
            padding: 0 80px;
        }

        .hero-background {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            object-fit: cover;
            opacity: 0.6;
        }

        .hero-content {
            position: relative;
            z-index: 2;
            max-width: 800px;
        }

        /* Grid de Notícias */
        .news-grid {
            display: grid;
            grid-template-columns: repeat(12, 1fr);
            gap: 2rem;
            margin: var(--spacing) 0;
        }

        .news-main {
            grid-column: span 8;
            height: 600px;
            position: relative;
        }

        .news-secondary {
            grid-column: span 4;
            height: 290px;
            position: relative;
        }

        .news-card {
            position: relative;
            overflow: hidden;
            border-radius: 8px;
        }

        .news-image {
            width: 100%;
            height: 100%;
            object-fit: cover;
            transition: transform 0.6s ease;
        }

        .news-overlay {
            position: absolute;
            bottom: 0;
            left: 0;
            right: 0;
            padding: 2rem;
            background: linear-gradient(transparent, rgba(0,0,0,0.9));
            transform: translateY(20px);
            opacity: 0;
            transition: all 0.4s ease;
        }

        .news-card:hover .news-image {
            transform: scale(1.05);
        }

        .news-card:hover .news-overlay {
            transform: translateY(0);
            opacity: 1;
        }

        /* Seções */
        .section-title {
            font-size: 2.5rem;
            font-weight: 300;
            margin: var(--spacing) 0 3rem 0;
            text-align: center;
        }

        /* Tags */
        .tag {
            background: var(--accent);
            color: var(--text);
            padding: 0.5rem 1rem;
            border-radius: 4px;
            font-size: 0.9rem;
            display: inline-block;
            margin-bottom: 1rem;
        }
    </style>
""", unsafe_allow_html=True)

# Hero Section
st.markdown("""
    <div class="hero-section">
        <img src="https://images.unsplash.com/photo-1573164713988-8665fc963095" 
             class="hero-background" 
             alt="Hero Background">
        <div class="hero-content">
            <h1 style="font-size: 4.5rem; font-weight: 300; margin-bottom: 2rem;">
                Narrativas que Transformam
            </h1>
            <p style="font-size: 1.5rem; font-weight: 300; opacity: 0.8;">
                Conectando histórias de inovação, cultura e urbanismo
            </p>
        </div>
    </div>
""", unsafe_allow_html=True)

# Destaques da Semana
st.markdown('<h2 class="section-title">Destaques da Semana</h2>', unsafe_allow_html=True)

st.markdown("""
    <div class="news-grid">
        <div class="news-main news-card">
            <img src="https://images.unsplash.com/photo-1522202176988-66273c2fd55f" 
                 class="news-image" 
                 alt="Tech Hub">
            <div class="news-overlay">
                <span class="tag">Tecnologia</span>
                <h3 style="font-size: 2rem; margin-bottom: 1rem;">
                    Hub de Inovação na Periferia Forma 100 Desenvolvedores
                </h3>
                <p style="font-size: 1.1rem; opacity: 0.8;">
                    Projeto revoluciona formação tech em comunidades
                </p>
            </div>
        </div>
        <div class="news-secondary news-card">
            <img src="https://images.unsplash.com/photo-1531384441138-2736e62e0919" 
                 class="news-image" 
                 alt="Arte Urbana">
            <div class="news-overlay">
                <span class="tag">Cultura</span>
                <h3>Festival Afro-Tech</h3>
            </div>
        </div>
        <div class="news-secondary news-card">
            <img src="https://images.unsplash.com/photo-1523450001312-faa4e2e37f0f" 
                 class="news-image" 
                 alt="Urbanismo">
            <div class="news-overlay">
                <span class="tag">Urbanismo</span>
                <h3>Revitalização Comunitária</h3>
            </div>
        </div>
    </div>
""", unsafe_allow_html=True)

# Histórias em Destaque
st.markdown('<h2 class="section-title">Histórias em Destaque</h2>', unsafe_allow_html=True)

historias_destaque = [
    {
        "imagem": "https://images.unsplash.com/photo-1573497019940-1c28c88b4f3e",
        "tag": "Empreendedorismo",
        "titulo": "Fintech Revoluciona Crédito na Periferia",
        "texto": "Startup liderada por jovem empreendedora já impactou mais de 10 mil famílias"
    },
    {
        "imagem": "https://images.unsplash.com/photo-1539701938214-0d9736e1c16b",
        "tag": "Tecnologia",
        "titulo": "Do Código à Comunidade",
        "texto": "Escola de programação gratuita forma nova geração de desenvolvedores"
    },
    {
        "imagem": "https://images.unsplash.com/photo-1524601500432-1e1a4c71d692",
        "tag": "Cultura",
        "titulo": "Arte que Transforma",
        "texto": "Coletivo de artistas revitaliza espaços abandonados com tecnologia"
    }
]

# Grid de histórias
cols = st.columns(3)
for idx, historia in enumerate(historias_destaque):
    with cols[idx]:
        st.markdown(f"""
            <div style="margin-bottom: 2rem;">
                <img src="{historia['imagem']}" 
                     style="width: 100%; height: 300px; object-fit: cover; border-radius: 8px; margin-bottom: 1rem;">
                <span class="tag">{historia['tag']}</span>
                <h3 style="font-size: 1.5rem; margin: 1rem 0;">{historia['titulo']}</h3>
                <p style="opacity: 0.8;">{historia['texto']}</p>
            </div>
        """, unsafe_allow_html=True)

# Nova seção: Cases de Sucesso com Slider Interativo
st.markdown("""
    <div class="cases-section">
        <h2 class="section-title">Cases de Sucesso</h2>
        <div class="cases-slider">
            <div class="case-card active">
                <img src="https://images.unsplash.com/photo-1573497019940-1c28c88b4f3e" alt="Case 1">
                <div class="case-content">
                    <span class="tag">Tech</span>
                    <h3>AfroHub</h3>
                    <p>+300% de visibilidade na mídia</p>
                    <div class="metrics">
                        <div class="metric">
                            <span class="number">150</span>
                            <span class="label">Matérias</span>
                        </div>
                        <div class="metric">
                            <span class="number">2M</span>
                            <span class="label">Alcance</span>
                        </div>
                    </div>
                </div>
            </div>
            <!-- Adicione mais cases aqui -->
        </div>
    </div>
""", unsafe_allow_html=True)

# Seção: Áreas de Atuação com Hover Effects
areas_atuacao = [
    {
        "icone": "🎨",
        "titulo": "Cultura",
        "descricao": "Festivais, exposições e eventos culturais"
    },
    {
        "icone": "💻",
        "titulo": "Tecnologia",
        "descricao": "Startups, inovação e transformação digital"
    },
    {
        "icone": "🏙️",
        "titulo": "Urbanismo",
        "descricao": "Projetos urbanos e desenvolvimento social"
    },
    {
        "icone": "🌱",
        "titulo": "Sustentabilidade",
        "descricao": "Iniciativas verdes e impacto social"
    }
]

st.markdown("""
    <div class="areas-section">
        <h2 class="section-title">Áreas de Atuação</h2>
        <div class="areas-grid">
""", unsafe_allow_html=True)

for area in areas_atuacao:
    st.markdown(f"""
        <div class="area-card">
            <div class="area-icon">{area['icone']}</div>
            <h3>{area['titulo']}</h3>
            <p>{area['descricao']}</p>
            <div class="area-hover">
                <ul>
                    <li>Assessoria de Imprensa</li>
                    <li>Gestão de Redes Sociais</li>
                    <li>Produção de Conteúdo</li>
                    <li>Relações Públicas</li>
                </ul>
            </div>
        </div>
    """, unsafe_allow_html=True)

# Seção: Timeline de Projetos
st.markdown("""
    <div class="timeline-section">
        <h2 class="section-title">Nossa Trajetória</h2>
        <div class="timeline">
            <div class="timeline-item">
                <div class="timeline-dot"></div>
                <div class="timeline-content">
                    <h3>2024</h3>
                    <p>Lançamento do maior hub de inovação da América Latina</p>
                </div>
            </div>
            <!-- Adicione mais itens na timeline -->
        </div>
    </div>
""", unsafe_allow_html=True)

# Seção: Números e Impacto com Contador Animado
st.markdown("""
    <div class="impact-section">
        <h2 class="section-title">Nosso Impacto</h2>
        <div class="impact-grid">
            <div class="impact-card">
                <span class="counter">500+</span>
                <p>Matérias publicadas</p>
            </div>
            <div class="impact-card">
                <span class="counter">50M+</span>
                <p>Alcance total</p>
            </div>
            <div class="impact-card">
                <span class="counter">100+</span>
                <p>Clientes atendidos</p>
            </div>
        </div>
    </div>
""", unsafe_allow_html=True)

# CSS adicional para as novas seções
st.markdown("""
    <style>
        /* Cases Slider */
        .cases-slider {
            overflow-x: auto;
            scroll-snap-type: x mandatory;
            display: flex;
            gap: 2rem;
            padding: 2rem 0;
        }

        .case-card {
            flex: 0 0 400px;
            scroll-snap-align: start;
            background: var(--secondary);
            border-radius: 12px;
            overflow: hidden;
            transition: transform 0.3s ease;
        }

        .case-card:hover {
            transform: translateY(-10px);
        }

        /* Áreas de Atuação */
        .areas-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 2rem;
            padding: 2rem 0;
        }

        .area-card {
            position: relative;
            background: var(--secondary);
            padding: 2rem;
            border-radius: 12px;
            text-align: center;
            overflow: hidden;
        }

        .area-hover {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: var(--accent);
            display: flex;
            align-items: center;
            justify-content: center;
            opacity: 0;
            transition: opacity 0.3s ease;
        }

        .area-card:hover .area-hover {
            opacity: 1;
        }

        /* Timeline */
        .timeline {
            position: relative;
            padding: 2rem 0;
        }

        .timeline::before {
            content: '';
            position: absolute;
            left: 50%;
            width: 2px;
            height: 100%;
            background: var(--accent);
        }

        .timeline-item {
            display: flex;
            justify-content: space-between;
            padding: 2rem 0;
        }

        /* Impacto */
        .impact-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 2rem;
            padding: 2rem 0;
        }

        .impact-card {
            text-align: center;
            padding: 2rem;
            background: var(--secondary);
            border-radius: 12px;
        }

        .counter {
            font-size: 3rem;
            font-weight: 700;
            color: var(--accent);
        }

        /* Animações */
        @keyframes countUp {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .counter {
            animation: countUp 2s ease-out forwards;
        }

        /* Responsividade */
        @media (max-width: 768px) {
            .impact-grid {
                grid-template-columns: 1fr;
            }

            .timeline::before {
                left: 0;
            }

            .timeline-item {
                flex-direction: column;
            }
        }
    </style>

    <script>
        // Animação dos contadores
        const counters = document.querySelectorAll('.counter');
        const options = {
            threshold: 1,
            rootMargin: "0px"
        };

        const observer = new IntersectionObserver((entries, observer) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('animate');
                    observer.unobserve(entry.target);
                }
            });
        }, options);

        counters.forEach(counter => observer.observe(counter));
    </script>
""", unsafe_allow_html=True) 
