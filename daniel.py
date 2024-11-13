import streamlit as st
from PIL import Image
import requests
from io import BytesIO

# Configuração inicial
st.set_page_config(
    page_title="Capivara Assessoria",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS com nova paleta de cores
st.markdown("""
    <style>
        /* Cores e variáveis */
        :root {
            --primary: #0A0A0A;
            --secondary: #0B3C2D;    /* Verde escuro como cor secundária */
            --accent: #15573F;       /* Verde mais claro para hover */
            --accent-light: #1B694C; /* Verde claro para elementos menores */
            --text: #FFFFFF;
            --text-secondary: #E0E0E0;
            --overlay: rgba(11, 60, 45, 0.9);
            --spacing: 100px;
        }

        /* Atualizações nos elementos existentes */
        .case-card:hover {
            border-color: var(--secondary);
            box-shadow: 0 10px 30px rgba(11, 60, 45, 0.2);
        }

        .area-card:hover .area-hover {
            background: var(--overlay);
        }

        .timeline::before {
            background: var(--secondary);
        }

        .impact-card .counter {
            color: var(--accent);
        }

        .tag {
            background: var(--secondary);
        }

        .news-tag {
            background-color: var(--secondary);
        }

        .service-icon {
            color: var(--accent);
        }

        .cta-button {
            background-color: var(--secondary);
        }

        .cta-button:hover {
            background-color: var(--accent);
        }

        /* Mantendo os estilos de cards e interatividade */
        .interactive-card {
            background: rgba(11, 60, 45, 0.1);
            border: 1px solid var(--secondary);
        }

        .interactive-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 30px rgba(11, 60, 45, 0.2);
        }

        /* Mantendo os gradientes atualizados */
        .hero-section::before {
            background: linear-gradient(
                45deg, 
                rgba(11, 60, 45, 0.9),
                rgba(27, 105, 76, 0.7)
            );
        }

        .news-overlay {
            background: linear-gradient(
                transparent,
                var(--overlay)
            );
        }

        /* Atualizando elementos de destaque */
        .highlight {
            color: var(--accent);
        }

        .metric-value {
            color: var(--accent);
        }

        /* Mantendo as animações e outros estilos inalterados */
        /* ... resto do CSS anterior ... */
    </style>
""", unsafe_allow_html=True)

# Hero Section Minimalista
st.markdown("""
    <div style="height: 90vh; display: flex; align-items: center; padding: 0 5%;">
        <div class="fade-in">
            <h1 style="font-size: 4rem; font-weight: 300; margin-bottom: 2rem;">
                Narrativas que Impactam
            </h1>
            <p style="font-size: 1.5rem; color: var(--text-secondary); max-width: 600px;">
                Conectando histórias de inovação social e transformação urbana
            </p>
        </div>
    </div>
""", unsafe_allow_html=True)

# Dados simulados para Cases de Sucesso
cases_sucesso = [
    {
        "cliente": "AfroTech Hub",
        "categoria": "Tecnologia",
        "resultados": {
            "mídia": "150+ matérias",
            "alcance": "2M+ pessoas",
            "ROI": "300% em visibilidade"
        },
        "imagem": "https://images.unsplash.com/photo-1573164713988-8665fc963095"
    },
    {
        "cliente": "Periferia Verde",
        "categoria": "Sustentabilidade",
        "resultados": {
            "mídia": "80+ matérias",
            "alcance": "1.5M+ pessoas",
            "ROI": "250% em engajamento"
        },
        "imagem": "https://images.unsplash.com/photo-1523450001312-faa4e2e37f0f"
    },
    {
        "cliente": "Arte Urbana Coletiva",
        "categoria": "Cultura",
        "resultados": {
            "mídia": "200+ matérias",
            "alcance": "3M+ pessoas",
            "ROI": "400% em visibilidade"
        },
        "imagem": "https://images.unsplash.com/photo-1531384441138-2736e62e0919"
    }
]

# Renderização dos Cases
st.markdown("<h2 style='text-align: center; margin: var(--spacing) 0;'>Cases de Sucesso</h2>", unsafe_allow_html=True)

for case in cases_sucesso:
    st.markdown(f"""
        <div class="interactive-card fade-in">
            <div style="display: flex; gap: 2rem; align-items: center;">
                <img src="{case['imagem']}" 
                     style="width: 200px; height: 200px; object-fit: cover; border-radius: 8px;">
                <div>
                    <span style="color: var(--secondary); background: rgba(11, 60, 45, 0.1); 
                                padding: 0.5rem 1rem; border-radius: 20px; font-size: 0.9rem;">
                        {case['categoria']}
                    </span>
                    <h3 style="font-size: 2rem; margin: 1rem 0;">{case['cliente']}</h3>
                    <div style="display: flex; gap: 2rem;">
                        <div>
                            <h4 style="color: var(--text-secondary);">Mídia</h4>
                            <p style="font-size: 1.2rem;">{case['resultados']['mídia']}</p>
                        </div>
                        <div>
                            <h4 style="color: var(--text-secondary);">Alcance</h4>
                            <p style="font-size: 1.2rem;">{case['resultados']['alcance']}</p>
                        </div>
                        <div>
                            <h4 style="color: var(--text-secondary);">ROI</h4>
                            <p style="font-size: 1.2rem;">{case['resultados']['ROI']}</p>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        <div style="height: 2rem;"></div>
    """, unsafe_allow_html=True)

# Projetos em Destaque
projetos_destaque = [
    {
        "titulo": "Programa de Formação Tech",
        "cliente": "AfroTech Hub",
        "impacto": "1000+ jovens formados",
        "midia": "150 matérias",
        "imagem": "https://images.unsplash.com/photo-1522202176988-66273c2fd55f"
    },
    {
        "titulo": "Festival de Arte Urbana",
        "cliente": "Coletivo Cultural",
        "impacto": "50k+ participantes",
        "midia": "80 matérias",
        "imagem": "https://images.unsplash.com/photo-1524601500432-1e1a4c71d692"
    },
    {
        "titulo": "Hortas Comunitárias",
        "cliente": "Periferia Verde",
        "impacto": "30 comunidades",
        "midia": "100 matérias",
        "imagem": "https://images.unsplash.com/photo-1523450001312-faa4e2e37f0f"
    }
]

# Grid de Projetos
st.markdown("<h2 style='text-align: center; margin: var(--spacing) 0;'>Projetos em Destaque</h2>", unsafe_allow_html=True)

cols = st.columns(3)
for idx, projeto in enumerate(projetos_destaque):
    with cols[idx]:
        st.markdown(f"""
            <div class="interactive-card">
                <img src="{projeto['imagem']}" 
                     style="width: 100%; height: 200px; object-fit: cover; border-radius: 8px; margin-bottom: 1rem;">
                <h3 style="font-size: 1.5rem; margin: 1rem 0;">{projeto['titulo']}</h3>
                <p style="color: var(--text-secondary);">{projeto['cliente']}</p>
                <div style="margin-top: 1rem;">
                    <span style="display: block; color: var(--accent);">Impacto: {projeto['impacto']}</span>
                    <span style="display: block; color: var(--accent);">Mídia: {projeto['midia']}</span>
                </div>
            </div>
        """, unsafe_allow_html=True)

# Métricas de Impacto com animação
st.markdown("""
    <div style="margin: var(--spacing) 0; text-align: center;">
        <h2 style="margin-bottom: var(--spacing);">Nosso Impacto</h2>
        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 2rem;">
            <div class="interactive-card">
                <h3 style="font-size: 3rem; color: var(--accent);">500+</h3>
                <p>Matérias publicadas</p>
            </div>
            <div class="interactive-card">
                <h3 style="font-size: 3rem; color: var(--accent);">50M+</h3>
                <p>Alcance total</p>
            </div>
            <div class="interactive-card">
                <h3 style="font-size: 3rem; color: var(--accent);">100+</h3>
                <p>Clientes atendidos</p>
            </div>
            <div class="interactive-card">
                <h3 style="font-size: 3rem; color: var(--accent);">30+</h3>
                <p>Prêmios conquistados</p>
            </div>
        </div>
    </div>
""", unsafe_allow_html=True) 
