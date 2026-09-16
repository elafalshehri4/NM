import base64
from html import escape
from pathlib import Path

import streamlit as st

from components.navigation import navbar
from utilities.ui import load_css


# =========================
# Page configuration
# =========================
st.set_page_config(
    page_title="Numerical Methods",
    page_icon="📘",
    layout="wide",
)


# =========================
# Theme and navigation
# =========================
load_css()
navbar(active_page="home")


# =========================
# Image helpers
# =========================
def image_to_base64(image_path: str) -> str:
    """Convert a local image into Base64 for use in HTML."""
    path = Path(image_path)
    if not path.exists() or not path.is_file():
        return ""
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def image_tag(image_data: str, css_class: str, alt_text: str) -> str:
    """Return an image tag only when the asset exists."""
    if not image_data:
        return ""

    return (
        f'<img src="data:image/png;base64,{image_data}" '
        f'class="{escape(css_class)}" alt="{escape(alt_text)}">'
    )


# =========================
# Home page assets
# =========================
hero_image = image_to_base64("assets/images/hero_science_visual.png")
structured_icon = image_to_base64("assets/icons/structured_lessons_icon.png")
practice_icon = image_to_base64("assets/icons/practice_test_icon.png")
solver_icon = image_to_base64("assets/icons/smart_solver_icon.png")


# =========================
# Learn, Practice, Solve cards
# =========================
journey_items = [
    {
        "title": "Learn",
        "description": (
            "Explore each numerical method through concise explanations, mathematical "
            "models, algorithm flowcharts, MATLAB/Python implementations, and solved "
            "applications."
        ),
        "link": "/Learn",
        "link_text": "Explore lessons",
        "icon": structured_icon,
        "class_name": "journey-learn",
    },
    {
        "title": "Assess",
        "description": (
            "Assess your understanding of each numerical method with interactive "
            "quizzes and instant feedback."
        ),
        "link": "/Quizzes",
        "link_text": "Take Quiz",
        "icon": practice_icon,
        "class_name": "journey-practice",
    },
    {
        "title": "Solve",
        "description": (
            "Solve your own problems, inspect each computational step, analyse errors, "
            "and export results."
        ),
        "link": "/Numerical_Solver",
        "link_text": "Launch Solver",
        "icon": solver_icon,
        "class_name": "journey-solve",
    },
]


def render_journey_cards(items: list[dict[str, str]]) -> str:
    """Build the Learn, Practice, and Solve card HTML."""
    cards = []

    for item in items:
        cards.append(
            f"""
            <a class="journey-card {escape(item['class_name'])}"
               href="{escape(item['link'])}" target="_self">
                {image_tag(item['icon'], 'journey-card-icon', item['title'])}
                <div class="journey-card-body">
                    <h3>{escape(item['title'])}</h3>
                    <p>{escape(item['description'])}</p>
                    <span>{escape(item['link_text'])} →</span>
                </div>
            </a>
            """
        )

    return "".join(cards)


# =========================
# Home-page styling
# =========================
st.html(
    """
    <style>
        /* Only the top hero is wider. The sections below remain unchanged. */
        .home-hero {
            width: min(1450px, calc(100% - 40px));
            margin: 14px auto 74px;
            display: grid;
            grid-template-columns: minmax(0, 1.05fr) minmax(360px, 0.95fr);
            align-items: center;
            gap: clamp(36px, 5vw, 76px);
            min-height: 560px;
            padding: 46px clamp(20px, 3vw, 44px);
        }

        .hero-left {
            position: relative;
            z-index: 2;
        }

        .hero-eyebrow {
            display: inline-flex;
            align-items: center;
            padding: 8px 14px;
            margin-bottom: 20px;
            border: 1px solid rgba(13, 148, 136, 0.24);
            border-radius: 999px;
            background: rgba(240, 253, 250, 0.9);
            color: #0f766e;
            font-size: 0.82rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }

        .hero-title-main {
            margin: 0;
            max-width: 690px;
            color: #0f2144;
            font-size: clamp(3.25rem, 6vw, 5.7rem);
            line-height: 0.98;
            letter-spacing: -0.055em;
        }

        .teal-word {
            color: #0f9f91;
        }

        .hero-description {
            max-width: 620px;
            margin: 26px 0 30px;
            color: #52627a;
            font-size: clamp(1.05rem, 1.6vw, 1.24rem);
            line-height: 1.72;
        }

        .hero-buttons {
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
        }

        .hero-buttons a {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            min-height: 50px;
            padding: 0 22px;
            border-radius: 14px;
            font-weight: 800;
            text-decoration: none !important;
            transition: transform 160ms ease, box-shadow 160ms ease,
                        border-color 160ms ease;
        }

        .hero-buttons a:hover {
            transform: translateY(-2px);
        }

        .btn-primary-ui {
            color: white !important;
            background: linear-gradient(135deg, #0f9f91, #21b7a9);
            box-shadow: 0 12px 28px rgba(15, 159, 145, 0.22);
        }

        .btn-soft-ui {
            color: #6446a8 !important;
            border: 1px solid #d8ccef;
            background: #f5f0ff;
        }

        .btn-solver-orange {
            color: white !important;
            border: 1px solid #f28a4b;
            background: linear-gradient(135deg, #ff9f5a, #f47b3f);
            box-shadow: 0 12px 28px rgba(244, 123, 63, 0.24);
        }

        .btn-solver-orange:hover {
            color: white !important;
            border-color: #e86d32;
            background: linear-gradient(135deg, #f89048, #e96b32);
        }

        .science-visual {
            position: relative;
            display: flex;
            align-items: center;
            justify-content: center;
            min-height: 420px;
        }

        .hero-science-img {
            position: relative;
            z-index: 1;
            width: min(500px, 100%);
            max-height: 470px;
            object-fit: contain;
            filter: drop-shadow(0 24px 30px rgba(15, 33, 68, 0.12));
        }

        .home-section {
            width: min(1180px, calc(100% - 32px));
            margin: 0 auto 78px;
        }

        .section-header-row {
            display: flex;
            align-items: end;
            justify-content: space-between;
            gap: 20px;
            margin-bottom: 26px;
        }

        .section-kicker {
            margin: 0 0 8px;
            color: #0f9f91;
            font-size: 0.78rem;
            font-weight: 850;
            letter-spacing: 0.1em;
            text-transform: uppercase;
        }

        .section-header-row h2 {
            margin: 0;
            color: #0f2144;
            font-size: clamp(2rem, 3vw, 3rem);
            letter-spacing: -0.035em;
        }

        .section-header-row p {
            max-width: 720px;
            margin: 10px 0 0;
            color: #5b6b82;
            font-size: 1.02rem;
            line-height: 1.7;
        }

        .journey-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 20px;
        }

        .journey-card {
            position: relative;
            display: flex;
            min-height: 255px;
            padding: 26px;
            border: 1px solid rgba(148, 163, 184, 0.24);
            border-radius: 24px;
            text-decoration: none !important;
            overflow: hidden;
            transition: transform 180ms ease, box-shadow 180ms ease;
        }

        .journey-card::after {
            content: "";
            position: absolute;
            right: -42px;
            bottom: -52px;
            width: 150px;
            height: 150px;
            border-radius: 50%;
            background: rgba(255, 255, 255, 0.48);
        }

        .journey-card:hover {
            transform: translateY(-6px);
            box-shadow: 0 22px 42px rgba(15, 23, 42, 0.11);
        }

        .journey-learn {
            background: linear-gradient(145deg, #eafaf7, #dff5f2);
        }

        .journey-practice {
            background: linear-gradient(145deg, #f4efff, #ece3ff);
        }

        .journey-solve {
            background: linear-gradient(145deg, #fff2ed, #ffe5dc);
        }

        .journey-card-icon {
            width: 72px;
            height: 72px;
            object-fit: cover;
            margin-right: 18px;
            flex: 0 0 auto;
            border-radius: 18px;
            clip-path: inset(0 round 18px);
        }

        .journey-card-body {
            position: relative;
            z-index: 1;
            display: flex;
            flex-direction: column;
        }

        .journey-card h3 {
            margin: 4px 0 10px;
            color: #10264b;
            font-size: 1.55rem;
        }

        .journey-card p {
            margin: 0 0 20px;
            color: #53637a;
            line-height: 1.65;
        }

        .journey-card span {
            margin-top: auto;
            color: #0f766e;
            font-weight: 850;
        }

        .about-section {
            width: min(920px, calc(100% - 32px));
            margin: 0 auto 72px;
            padding: 38px 34px;
            text-align: center;
            border-top: 1px solid rgba(148, 163, 184, 0.24);
        }

        .about-section .section-kicker {
            text-align: center;
        }

        .about-section h2 {
            margin: 0;
            color: #0f2144;
            font-size: clamp(2rem, 3vw, 2.7rem);
            letter-spacing: -0.035em;
        }

        .about-section p:last-child {
            max-width: 760px;
            margin: 14px auto 0;
            color: #5b6b82;
            font-size: 1.02rem;
            line-height: 1.75;
        }

        .footer-ui {
            width: min(1180px, calc(100% - 32px));
            margin: 0 auto 26px;
            padding: 20px 4px 8px;
            display: flex;
            justify-content: space-between;
            gap: 20px;
            border-top: 1px solid rgba(148, 163, 184, 0.25);
            color: #6b7a90;
            font-size: 0.94rem;
        }

        @media (max-width: 980px) {
            .home-hero {
                width: min(100% - 28px, 900px);
                grid-template-columns: 1fr;
                min-height: auto;
                text-align: center;
            }

            .hero-description {
                margin-left: auto;
                margin-right: auto;
            }

            .hero-buttons {
                justify-content: center;
            }

            .science-visual {
                min-height: 340px;
            }

            .journey-grid {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
        }

        @media (max-width: 650px) {
            .home-hero,
            .home-section,
            .about-section,
            .footer-ui {
                width: min(100% - 20px, 1180px);
            }

            .home-hero {
                margin-bottom: 54px;
                padding: 34px 18px;
            }

            .hero-title-main {
                font-size: clamp(2.7rem, 13vw, 4rem);
            }

            .hero-buttons a {
                width: 100%;
            }

            .science-visual {
                min-height: 280px;
            }

            .section-header-row {
                align-items: flex-start;
                flex-direction: column;
            }

            .journey-grid {
                grid-template-columns: 1fr;
            }

            .journey-card {
                min-height: 225px;
            }

            .about-section {
                padding: 30px 20px;
            }

            .footer-ui {
                flex-direction: column;
                text-align: center;
            }
        }
    </style>
    """
)


# =========================
# Hero
# =========================
st.html(
    f"""
    <section class="home-hero">
        <div class="hero-left">
            <div class="hero-eyebrow">
                Interactive Numerical Methods Learning Platform
            </div>

            <h1 class="hero-title-main">
                <span class="teal-word">Master</span> Numerical<br>
                Methods
            </h1>

            <p class="hero-description">
                Explore numerical methods through mathematical models, flowcharts,
                implementation hints, and MATLAB/Python code. Evaluate your understanding
                with quizzes and practise using interactive solvers.
            </p>

            <div class="hero-buttons">
                <a href="/Learn" target="_self" class="btn-primary-ui">
                    Start Learning →
                </a>

                <a href="/Quizzes" target="_self" class="btn-soft-ui">
                    Take Quiz →
                </a>

                <a href="/Numerical_Solver" target="_self"
                   class="btn-solver-orange">
                    Launch Solver →
                </a>
            </div>
        </div>

        <div class="science-visual">
            {image_tag(
                hero_image,
                "hero-science-img",
                "Numerical methods scientific visual",
            )}
        </div>
    </section>
    """
)


# =========================
# Learn → Practice → Solve
# =========================
st.html(
    f"""
    <section class="home-section">
        <div class="section-header-row">
            <div>
                <p class="section-kicker">Choose your path</p>
                <h2>Learn, Assess then Solve</h2>
                <p>
                    Follow the recommended learning path, or jump directly to any section.
                </p>
            </div>
        </div>

        <div class="journey-grid">
            {render_journey_cards(journey_items)}
        </div>
    </section>
    """
)


# =========================
# About
# =========================
st.html(
    """
    <section class="about-section">
        <p class="section-kicker">About the platform</p>
        <h2>Learn, Assess, and Solve with Confidence</h2>
        <p>
            A complete platform for numerical methods featuring mathematical models,
            algorithm flowcharts, MATLAB/Python implementations, interactive quizzes,
            and step-by-step solvers with exportable results-all in one place.
        </p>
    </section>
    """
)


# =========================
# Footer
# =========================
st.html(
    """
    <footer class="footer-ui">
        <div>NM • © 2026 Numerical Methods</div>
        <div>Keep learning. Keep solving. Keep growing.</div>
    </footer>
    """
)