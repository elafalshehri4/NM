import base64
from pathlib import Path

import streamlit as st

from components.navigation import navbar
from utilities.ui import load_css


st.set_page_config(
    page_title="Numerical Solver | Numerical Methods",
    page_icon="🧮",
    layout="wide",
)

load_css()
navbar(active_page="solver")


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
            <div class="page-label">SOLVER CENTER</div>
            <h1>Numerical Methods Solvers</h1>
            <p>
                Select a method to open its interactive solver, enter your data, inspect each
                computational step, analyse errors, and export the results.
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
                <h2>Choose a Solver Category</h2>
                <p>Select a method to open its solver page.</p>
            </div>
        </div>

        <div class="learn-category-grid">

            <div class="learn-category-card card-mint">
                <img src="data:image/png;base64,{root_icon}" class="learn-category-icon" alt="Algebraic Equations">
                <h3>Algebraic Equations</h3>
                <p>Solve nonlinear equations using iterative methods.</p>

                <div class="method-list">
                    <a href="/Bisection_Solver" target="_self">Bisection Solver →</a>
                    <a href="/Secant_Solver" target="_self">Secant Solver →</a>
                    <a href="/Newton_Raphson_Solver" target="_self">Newton-Raphson Solver →</a>
                    <a href="/Multiple_Roots_Solver" target="_self">Multiple Roots Solver →</a>
                </div>
            </div>

            <div class="learn-category-card card-blue">
                <img src="data:image/png;base64,{linear_icon}" class="learn-category-icon" alt="Linear Systems">
                <h3>Linear Systems</h3>
                <p>Solve linear systems using numerical elimination methods.</p>

                <div class="method-list">
                    <a href="/Naive_Gaussian_Elimination_Solver" target="_self">Naive Gaussian Elimination Solver →</a>
                    <a href="/Gaussian_Elimination_Scaled_Partial_Pivoting_Solver" target="_self">Scaled Partial Pivoting Solver →</a>
                    <a href="/Gauss_Jordan_Solver" target="_self">Gauss-Jordan Solver →</a>
                    <a href="/Tridiagonal_Systems_Solver" target="_self">Tridiagonal Systems Solver →</a>
                </div>
            </div>

            <div class="learn-category-card card-lavender">
                <img src="data:image/png;base64,{interpolation_icon}" class="learn-category-icon" alt="Differentiation">
                <h3>Differentiation</h3>
                <p>Estimate derivatives with numerical formulas.</p>

                <div class="method-list">
                    <a href="/Forward_Difference_Solver" target="_self">Forward Difference Solver →</a>
                    <a href="/Backward_Difference_Solver" target="_self">Backward Difference Solver →</a>
                    <a href="/Central_Difference_Solver" target="_self">Central Difference Solver →</a>
                    <a href="/Richardson_Extrapolation_Solver" target="_self">Richardson Extrapolation Solver →</a>
                </div>
            </div>

            <div class="learn-category-card card-yellow">
                <img src="data:image/png;base64,{integration_icon}" class="learn-category-icon" alt="Integration">
                <h3>Integration</h3>
                <p>Approximate definite integrals numerically.</p>

                <div class="method-list">
                    <a href="/Trapezoidal_Solver" target="_self">Trapezoidal Rule Solver →</a>
                    <a href="/Simpsons_Solver" target="_self">Simpson's Rules Solver →</a>
                    <a href="/Gaussian_Quadrature_Solver" target="_self">Gaussian Quadrature Solver →</a>
                </div>
            </div>

            <div class="learn-category-card card-coral">
                <img src="data:image/png;base64,{odes_icon}" class="learn-category-icon" alt="ODEs">
                <h3>Ordinary Differential Equations</h3>
                <p>Approximate solutions of initial-value problems.</p>

                <div class="method-list">
                    <a href="/Euler_Solver" target="_self">Euler Method Solver →</a>
                    <a href="/Midpoint_Solver" target="_self">Midpoint Method Solver →</a>
                    <a href="/Heun_Solver" target="_self">Heun Method Solver →</a>
                    <a href="/Runge_Kutta_4_Solver" target="_self">Fourth-Order Runge-Kutta Method Solver →</a>
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
        <div>Apply your knowledge. Build your skills. ♥</div>
    </footer>
    """
)