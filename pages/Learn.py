import base64
from pathlib import Path

import streamlit as st

from components.navigation import navbar
from utilities.ui import load_css


st.set_page_config(
    page_title="Learn | Numerical Methods",
    page_icon="📘",
    layout="wide",
)

load_css()
navbar(active_page="learn")


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
            <div class="page-label">LEARN CENTER</div>
            <h1>Learn Numerical Methods</h1>
            <p>
                Choose a topic and explore each method through mathematical explanations,
                algorithm flowcharts, MATLAB/Python implementations, and solved applications.
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
                <h2>Choose a topic</h2>
                <p>Select a topic to browse its numerical methods and learn each method in detail.</p>
            </div>
        </div>

        <div class="learn-category-grid">

            <div class="learn-category-card card-mint">
                <img src="data:image/png;base64,{root_icon}" class="learn-category-icon" alt="Algebraic Equations">
                <h3>Algebraic Equations</h3>
                <p>Root-finding methods for nonlinear equations.</p>

                <div class="method-list">
                    <a href="/Bisection_Method" target="_self">Bisection Method →</a>
                    <a href="/Secant_Method" target="_self">Secant Method →</a>
                    <a href="/Newton_Raphson_Method" target="_self">Newton-Raphson Method →</a>
                    <a href="/Multiple_Roots_Method" target="_self">Multiple Roots Method →</a>
                </div>
            </div>

            <div class="learn-category-card card-blue">
                <img src="data:image/png;base64,{linear_icon}" class="learn-category-icon" alt="Linear Systems">
                <h3>Linear Systems</h3>
                <p>Methods for solving linear systems of equations.</p>

                <div class="method-list">
                    <a href="/Naive_Gaussian_Elimination" target="_self">Naive Gaussian Elimination →</a>
                    <a href="/Gaussian_Elimination_Scaled_Partial_Pivoting" target="_self">Scaled Partial Pivoting →</a>
                    <a href="/Gauss_Jordan_Method" target="_self">Gauss-Jordan Method →</a>
                    <a href="/Tridiagonal_Systems" target="_self">Tridiagonal Systems →</a>
                </div>
            </div>

            <div class="learn-category-card card-lavender">
                <img src="data:image/png;base64,{interpolation_icon}" class="learn-category-icon" alt="Differentiation">
                <h3>Differentiation</h3>
                <p>Finite-difference methods for estimating derivatives.</p>

                <div class="method-list">
                    <a href="/Forward_Difference" target="_self">Forward Difference →</a>
                    <a href="/Backward_Difference" target="_self">Backward Difference →</a>
                    <a href="/Central_Difference" target="_self">Central Difference →</a>
                    <a href="/Richardson_Extrapolation" target="_self">Richardson Extrapolation →</a>
                </div>
            </div>

            <div class="learn-category-card card-yellow">
                <img src="data:image/png;base64,{integration_icon}" class="learn-category-icon" alt="Integration">
                <h3>Integration</h3>
                <p>Numerical methods for approximating definite integrals.</p>

                <div class="method-list">
                    <a href="/Trapezoidal_Rule" target="_self">Trapezoidal Rule →</a>
                    <a href="/Simpsons_Rule" target="_self">Simpson's Rules →</a>
                    <a href="/Gaussian_Quadrature" target="_self">Gaussian Quadrature →</a>
                </div>
            </div>

            <div class="learn-category-card card-coral">
                <img src="data:image/png;base64,{odes_icon}" class="learn-category-icon" alt="ODEs">
                <h3>Ordinary Differential Equations</h3>
                <p>Methods for approximating initial-value problems.</p>

                <div class="method-list">
                    <a href="/Euler_Method" target="_self">Euler Method →</a>
                    <a href="/Midpoint_Method" target="_self">Midpoint Method →</a>
                    <a href="/Heun_Method" target="_self">Heun Method →</a>
                    <a href="/Runge_Kutta_4" target="_self">Fourth-Order Runge-Kutta Method →</a>
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
        <div>Learn step by step. Build your knowledge. ♥</div>
    </footer>
    """
)