import base64
from pathlib import Path

import streamlit as st

from components.navigation import navbar
from utilities.ui import load_css


st.set_page_config(
    page_title="Quizzes | Numerical Methods",
    page_icon="📝",
    layout="wide",
)

load_css()
navbar(active_page="quizzes")


def image_to_base64(image_path):
    path = Path(image_path)

    if not path.exists():
        return ""

    return base64.b64encode(path.read_bytes()).decode("utf-8")


root_icon = image_to_base64("assets/icons/root_finding_icon.png")
linear_icon = image_to_base64("assets/icons/linear_systems_icon.png")
interpolation_icon = image_to_base64("assets/icons/interpolation_icon.png")
integration_icon = image_to_base64("assets/icons/numerical_integration_icon.png")
odes_icon = image_to_base64("assets/icons/odes_icon.png")


st.html(
    """
    <section class="learn-hero">
        <div>
            <div class="page-label">QUIZ CENTER</div>
            <h1>Numerical Methods Quizzes</h1>
            <p>
                Select a method and assess your understanding with focused questions and
                instant feedback.
            </p>
        </div>
    </section>
    """
)

st.html(
    f"""
    <main class="learn-wrapper">
        <div class="section-header learn-header-row">
            <div>
                <h2>Choose a Quiz Topic</h2>
                <p>Select a method to open its quiz page.</p>
            </div>
        </div>

        <div class="learn-category-grid">

            <div class="learn-category-card card-mint">
                <img src="data:image/png;base64,{root_icon}" class="learn-category-icon" alt="Algebraic Equations">
                <h3>Algebraic Equations</h3>
                <p>Test nonlinear root-finding concepts.</p>

                <div class="method-list">
                    <a href="/Bisection_Quiz" target="_self">Bisection Method Quiz →</a>
                    <a href="/Secant_Quiz" target="_self">Secant Method Quiz →</a>
                    <a href="/Newton_Raphson_Quiz" target="_self">Newton-Raphson Quiz →</a>
                    <a href="/Multiple_Roots_Quiz" target="_self">Multiple Roots Quiz →</a>
                </div>
            </div>

            <div class="learn-category-card card-blue">
                <img src="data:image/png;base64,{linear_icon}" class="learn-category-icon" alt="Linear Systems">
                <h3>Linear Systems</h3>
                <p>Practice linear-system methods.</p>

                <div class="method-list">
                    <a href="/Naive_Gaussian_Elimination_Quiz" target="_self">Naive Gaussian Elimination Quiz →</a>
                    <a href="/Gaussian_Elimination_Scaled_Partial_Pivoting_Quiz" target="_self">Scaled Partial Pivoting Quiz →</a>
                    <a href="/Gauss_Jordan_Quiz" target="_self">Gauss-Jordan Quiz →</a>
                    <a href="/Tridiagonal_Systems_Quiz" target="_self">Tridiagonal Systems Quiz →</a>
                </div>
            </div>

            <div class="learn-category-card card-lavender">
                <img src="data:image/png;base64,{interpolation_icon}" class="learn-category-icon" alt="Differentiation">
                <h3>Differentiation</h3>
                <p>Test finite-difference and extrapolation techniques.</p>

                <div class="method-list">
                    <a href="/Forward_Difference_Quiz" target="_self">Forward Difference Quiz →</a>
                    <a href="/Backward_Difference_Quiz" target="_self">Backward Difference Quiz →</a>
                    <a href="/Central_Difference_Quiz" target="_self">Central Difference Quiz →</a>
                    <a href="/Richardson_Extrapolation_Quiz" target="_self">Richardson Extrapolation Quiz →</a>
                </div>
            </div>

            <div class="learn-category-card card-yellow">
                <img src="data:image/png;base64,{integration_icon}" class="learn-category-icon" alt="Integration">
                <h3>Integration</h3>
                <p>Test numerical integration concepts and formulas.</p>

                <div class="method-list">
                    <a href="/Trapezoidal_Quiz" target="_self">Trapezoidal Rule Quiz →</a>
                    <a href="/Simpsons_Quiz" target="_self">Simpson's Rules Quiz →</a>
                    <a href="/Gaussian_Quadrature_Quiz" target="_self">Gaussian Quadrature Quiz →</a>
                </div>
            </div>

            <div class="learn-category-card card-coral">
                <img src="data:image/png;base64,{odes_icon}" class="learn-category-icon" alt="ODEs">
                <h3>Ordinary Differential Equations</h3>
                <p>Test numerical methods for initial-value problems.</p>

                <div class="method-list">
                    <a href="/Euler_Quiz" target="_self">Euler Method Quiz →</a>
                    <a href="/Midpoint_Quiz" target="_self">Midpoint Method Quiz →</a>
                    <a href="/Heun_Quiz" target="_self">Heun Method Quiz →</a>
                    <a href="/Runge_Kutta_4_Quiz" target="_self">Fourth-Order Runge-Kutta Method Quiz →</a>
                </div>
            </div>

        </div>
    </main>
    """
)

st.html(
    """
    <footer class="footer-ui">
        <div>NM • © 2026 Numerical Methods</div>
        <div>Assess your knowledge. Build your confidence. ♥</div>
    </footer>
    """
)