import base64
from pathlib import Path

import streamlit as st


def image_to_base64(image_path):
    """Convert a local image into Base64 for use in HTML."""
    path = Path(image_path)

    if not path.exists():
        return ""

    return base64.b64encode(
        path.read_bytes()
    ).decode("utf-8")


def navbar(active_page=""):
    """Display the shared navigation bar."""

    logo_image = image_to_base64(
        "assets/logo/nm_logo.png"
    )

    def active_class(page_name):
        return "active" if active_page == page_name else ""

    st.html(
        f"""
        <nav class="custom-navbar">

            <div class="logo-wrap">
                <a href="/" target="_self">
                    <img
                        src="data:image/png;base64,{logo_image}"
                        class="site-logo"
                        alt="Numerical Methods Logo"
                    >
                </a>
            </div>

            <div class="nav-links">

                <a
                    href="/"
                    target="_self"
                    class="{active_class('home')}"
                >
                    Home
                </a>

                <a
                    href="/Learn"
                    target="_self"
                    class="{active_class('learn')}"
                >
                    Learn
                </a>

                <a
                    href="/Quizzes"
                    target="_self"
                    class="{active_class('quizzes')}"
                >
                    Quizzes
                </a>

                <a
                    href="/Numerical_Solver"
                    target="_self"
                    class="{active_class('solver')}"
                >
                    Solver
                </a>

            </div>

        </nav>
        """
    )