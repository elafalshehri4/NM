import streamlit as st

from components.navigation import navbar
from utilities.ui import load_css


# =========================================================
# Page configuration
# =========================================================
st.set_page_config(
    page_title="Bisection Method | Numerical Methods",
    page_icon="📘",
    layout="wide",
)


# =========================================================
# Shared CSS and navbar
# =========================================================
load_css()
navbar(active_page="learn")


# =========================================================
# Page-specific visual styling
# =========================================================
st.markdown(
    """
    <style>
    .bisection-page {
        padding-top: 26px;
        padding-bottom: 12px;
    }

    .section-kicker {
        color: #0f766e;
        font-size: 12px;
        font-weight: 900;
        letter-spacing: 1.2px;
        text-transform: uppercase;
        margin-bottom: 5px;
    }

    .section-title {
        color: #0b1b3a;
        font-size: 25px;
        font-weight: 900;
        margin: 0 0 7px;
    }

    .section-intro {
        color: #475569;
        font-size: 15px;
        line-height: 1.65;
        margin: 0 0 18px;
    }

    .summary-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 14px;
        margin: 4px 0 22px;
    }

    .summary-card {
        min-height: 112px;
        padding: 18px;
        border-radius: 16px;
        border: 1px solid rgba(15, 61, 62, 0.10);
        box-shadow: 0 8px 20px rgba(15, 61, 62, 0.06);
    }

    .summary-card:nth-child(1) {
        background: linear-gradient(135deg, #f0fdfa, #ecfeff);
    }

    .summary-card:nth-child(2) {
        background: linear-gradient(135deg, #eff6ff, #e0f2fe);
    }

    .summary-card:nth-child(3) {
        background: linear-gradient(135deg, #f5f3ff, #faf5ff);
    }

    .summary-card:nth-child(4) {
        background: linear-gradient(135deg, #fff7ed, #fffbeb);
    }

    .summary-card span {
        display: block;
        color: #64748b;
        font-size: 12px;
        font-weight: 800;
        margin-bottom: 8px;
    }

    .summary-card strong {
        display: block;
        color: #0b1b3a;
        font-size: 18px;
        font-weight: 900;
        line-height: 1.25;
    }

    .concept-card {
        background: #ffffff;
        border: 1px solid rgba(15, 61, 62, 0.10);
        border-radius: 18px;
        padding: 22px 24px;
        height: 100%;
        box-shadow: 0 10px 24px rgba(15, 61, 62, 0.06);
    }

    .concept-card h3 {
        color: #0b1b3a;
        font-size: 19px;
        font-weight: 900;
        margin: 0 0 10px;
    }

    .concept-card p {
        color: #475569;
        font-size: 14px;
        line-height: 1.7;
        margin: 0;
    }

    .condition-list {
        display: grid;
        gap: 12px;
        margin-top: 10px;
    }

    .condition-item {
        display: flex;
        gap: 12px;
        align-items: flex-start;
        padding: 14px 16px;
        border-radius: 14px;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
    }

    .condition-number {
        width: 28px;
        height: 28px;
        min-width: 28px;
        border-radius: 50%;
        background: linear-gradient(135deg, #14b8a6, #0f766e);
        color: white;
        display: grid;
        place-items: center;
        font-size: 13px;
        font-weight: 900;
    }

    .condition-item div:last-child {
        color: #334155;
        font-size: 14px;
        line-height: 1.6;
    }

    .algorithm-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 12px;
        margin-top: 8px;
    }

    .algorithm-step {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 15px 16px;
    }

    .algorithm-step span {
        display: inline-block;
        color: #0f766e;
        font-size: 12px;
        font-weight: 900;
        margin-bottom: 6px;
    }

    .algorithm-step p {
        color: #334155;
        font-size: 14px;
        line-height: 1.55;
        margin: 0;
    }

    .iteration-card {
        background: linear-gradient(135deg, #ffffff, #f8fafc);
        border: 1px solid #dbe7ef;
        border-radius: 16px;
        padding: 18px;
        height: 100%;
    }

    .iteration-card h4 {
        color: #0b1b3a;
        font-size: 16px;
        font-weight: 900;
        margin: 0 0 10px;
    }

    .iteration-card p {
        color: #475569;
        font-size: 14px;
        line-height: 1.6;
        margin: 0;
    }

    .application-grid-advanced {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 14px;
    }

    .application-box {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 15px;
        padding: 18px;
        color: #334155;
        font-size: 14px;
        line-height: 1.55;
        min-height: 86px;
    }

    .application-box strong {
        display: block;
        color: #0b1b3a;
        margin-bottom: 5px;
        font-size: 14px;
    }

    .advantage-box,
    .limitation-box {
        border-radius: 16px;
        padding: 20px;
        min-height: 190px;
    }

    .advantage-box {
        background: #f0fdfa;
        border: 1px solid #99f6e4;
    }

    .limitation-box {
        background: #fff7ed;
        border: 1px solid #fed7aa;
    }

    .advantage-box h3,
    .limitation-box h3 {
        color: #0b1b3a;
        font-size: 18px;
        font-weight: 900;
        margin: 0 0 10px;
    }

    .advantage-box li,
    .limitation-box li {
        color: #475569;
        margin-bottom: 8px;
        line-height: 1.5;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 18px !important;
        border: 1px solid rgba(15, 61, 62, 0.10) !important;
        box-shadow: 0 10px 24px rgba(15, 61, 62, 0.06) !important;
    }

    div[data-testid="stExpander"] {
        border-radius: 14px !important;
        border-color: rgba(15, 61, 62, 0.12) !important;
        overflow: hidden !important;
    }

    @media (max-width: 1000px) {
        .summary-grid,
        .application-grid-advanced,
        .algorithm-grid {
            grid-template-columns: 1fr;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# Hero
# =========================================================
st.html(
    """
    <section class="method-hero">
        <div>
            <div class="page-label">ROOT FINDING METHOD</div>

            <h1>Bisection Method</h1>

            <p>
                Learn the mathematical basis, error behavior, convergence,
                stopping criteria, and implementation of one of the most
                dependable methods for solving nonlinear equations.
            </p>

            <div class="method-actions">
                <a
                    href="/Bisection_Solver"
                    target="_self"
                    class="btn-primary-ui"
                >
                    Open Solver →
                </a>

                <a
                    href="/Bisection_Quiz"
                    target="_self"
                    class="btn-outline-ui"
                >
                    Take Quiz →
                </a>
            </div>
        </div>
    </section>
    """
)


# =========================================================
# Main page
# =========================================================
left_margin, content, right_margin = st.columns([0.035, 0.93, 0.035])

with content:
    st.markdown('<div class="bisection-page">', unsafe_allow_html=True)

    # =====================================================
    # Quick summary
    # =====================================================
    st.html(
        """
        <div class="section-kicker">Quick reference</div>
        <h2 class="section-title">Method at a Glance</h2>
        <p class="section-intro">
            These four properties summarize the Bisection Method before
            studying the details.
        </p>

        <div class="summary-grid">
            <div class="summary-card">
                <span>METHOD TYPE</span>
                <strong>Bracketing root-finding method</strong>
            </div>

            <div class="summary-card">
                <span>REQUIREMENT</span>
                <strong>Continuous function with a sign change</strong>
            </div>

            <div class="summary-card">
                <span>CONVERGENCE</span>
                <strong>Linear and guaranteed</strong>
            </div>

            <div class="summary-card">
                <span>DERIVATIVE</span>
                <strong>Not required</strong>
            </div>
        </div>
        """
    )

    # =====================================================
    # Overview and foundation
    # =====================================================
    st.html(
        """
        <div class="section-kicker">Core concept</div>
        <h2 class="section-title">Overview and Mathematical Foundation</h2>
        <p class="section-intro">
            The method works because a continuous function cannot change
            sign without crossing the x-axis.
        </p>
        """
    )

    overview_col, foundation_col = st.columns(2)

    with overview_col:
        with st.container(border=True):
            st.subheader("Overview")

            st.write(
                "The Bisection Method solves nonlinear equations of the form:"
            )

            st.latex(r"f(x)=0")

            st.write(
                "It begins with an interval that contains a root. At each "
                "iteration, the interval is divided into two equal parts. "
                "The half that still contains a sign change is kept."
            )

    with foundation_col:
        with st.container(border=True):
            st.subheader("Intermediate Value Theorem")

            st.write(
                "If a function is continuous on an interval and its endpoint "
                "values have opposite signs, then at least one real root lies "
                "inside the interval."
            )

            st.latex(r"f(a)\,f(b)<0")

            st.write(
                "The selected interval always preserves this sign change, "
                "so the root remains bracketed."
            )

    # =====================================================
    # Conditions and midpoint
    # =====================================================
    st.html(
        """
        <div class="section-kicker">Requirements</div>
        <h2 class="section-title">Conditions and Core Formula</h2>
        <p class="section-intro">
            The method requires a valid interval and repeatedly uses its midpoint.
        </p>
        """
    )

    conditions_col, midpoint_col = st.columns([1.15, 0.85])

    with conditions_col:
        with st.container(border=True):
            st.subheader("Required Conditions")

            st.html(
                """
                <div class="condition-list">
                    <div class="condition-item">
                        <div class="condition-number">1</div>
                        <div>
                            The function must be continuous throughout the
                            selected interval.
                        </div>
                    </div>

                    <div class="condition-item">
                        <div class="condition-number">2</div>
                        <div>
                            The endpoint values must have opposite signs,
                            unless one endpoint is already a root.
                        </div>
                    </div>
                </div>
                """
            )

            st.latex(r"f(a)\,f(b)<0")

    with midpoint_col:
        with st.container(border=True):
            st.subheader("Midpoint Formula")

            st.write(
                "The midpoint becomes the next approximation of the root."
            )

            st.latex(r"c=\frac{a+b}{2}")

            st.info(
                "The sign of f(c) determines which half of the interval is kept."
            )

    # =====================================================
    # Error and convergence
    # =====================================================
    st.html(
        """
        <div class="section-kicker">Accuracy</div>
        <h2 class="section-title">Error and Convergence Analysis</h2>
        <p class="section-intro">
            The interval and the maximum possible error are reduced by one-half
            during every iteration.
        </p>
        """
    )

    error_col, convergence_col = st.columns(2)

    with error_col:
        with st.container(border=True):
            st.subheader("Error Bound")

            st.write(
                "After n iterations, the error is bounded by:"
            )

            st.latex(
                r"\left|x^\ast-c_n\right|\leq"
                r"\frac{b_0-a_0}{2^n}"
            )

            st.caption(
                "x* is the exact root, cₙ is the current midpoint, and "
                "[a₀, b₀] is the initial interval."
            )

    with convergence_col:
        with st.container(border=True):
            st.subheader("Convergence")

            st.write(
                "The Bisection Method has linear convergence."
            )

            st.latex(r"p=1")

            st.write(
                "It is not the fastest method, but it is highly reliable "
                "because the root remains inside the interval."
            )

    # =====================================================
    # Stopping criteria
    # =====================================================
    with st.container(border=True):
        st.subheader("Stopping Criteria")

        st.write(
            "The algorithm may stop when any of these conditions is satisfied:"
        )

        stop1, stop2, stop3 = st.columns(3)

        with stop1:
            st.markdown("**Function value test**")
            st.latex(r"|f(c)|<\varepsilon")
            st.caption("The midpoint is sufficiently close to a root.")

        with stop2:
            st.markdown("**Interval test**")
            st.latex(r"\frac{b-a}{2}<\varepsilon")
            st.caption("The remaining interval is sufficiently small.")

        with stop3:
            st.markdown("**Safety limit**")
            st.markdown("### Maximum iterations")
            st.caption("Stops the algorithm if the tolerance is not reached.")

    # =====================================================
    # Algorithm
    # =====================================================
    st.html(
        """
        <div class="section-kicker">Procedure</div>
        <h2 class="section-title">Algorithm</h2>
        <p class="section-intro">
            Each step keeps the root inside a smaller interval.
        </p>
        """
    )

    st.html(
        """
        <div class="algorithm-grid">
            <div class="algorithm-step">
                <span>STEP 1</span>
                <p>Choose an interval [a, b].</p>
            </div>

            <div class="algorithm-step">
                <span>STEP 2</span>
                <p>Check continuity and verify the sign change.</p>
            </div>

            <div class="algorithm-step">
                <span>STEP 3</span>
                <p>Calculate the midpoint of the interval.</p>
            </div>

            <div class="algorithm-step">
                <span>STEP 4</span>
                <p>Evaluate the function at the midpoint.</p>
            </div>

            <div class="algorithm-step">
                <span>STEP 5</span>
                <p>Keep the half that still contains the sign change.</p>
            </div>

            <div class="algorithm-step">
                <span>STEP 6</span>
                <p>Repeat until a stopping condition is satisfied.</p>
            </div>
        </div>
        """
    )

    with st.container(border=True):
        interval_left, interval_right = st.columns(2)

        with interval_left:
            st.markdown("**Keep the left half when:**")
            st.latex(r"f(a)\,f(c)<0")
            st.latex(r"[a,c]")

        with interval_right:
            st.markdown("**Otherwise, keep the right half:**")
            st.latex(r"[c,b]")

    # =====================================================
    # Worked example
    # =====================================================
    st.html(
        """
        <div class="section-kicker">Application</div>
        <h2 class="section-title">Worked Example</h2>
        <p class="section-intro">
            Observe how the interval becomes smaller while continuing to
            contain the root.
        </p>
        """
    )

    with st.container(border=True):
        equation_col, interval_col = st.columns(2)

        with equation_col:
            st.markdown("**Equation**")
            st.latex(r"f(x)=x^3-x-2")

        with interval_col:
            st.markdown("**Initial interval**")
            st.latex(r"[a,b]=[1,2]")

        st.markdown("**Endpoint check**")

        endpoint1, endpoint2 = st.columns(2)

        with endpoint1:
            st.metric("f(1)", "-2")

        with endpoint2:
            st.metric("f(2)", "4")

        st.success(
            "The endpoint values have opposite signs, so the interval contains a root."
        )

    iteration1, iteration2 = st.columns(2)

    with iteration1:
        with st.container(border=True):
            st.subheader("Iteration 1")
            st.latex(r"c_1=\frac{1+2}{2}=1.5")
            st.latex(r"f(1.5)=-0.125")
            st.write("The next interval is:")
            st.latex(r"[1.5,2]")

    with iteration2:
        with st.container(border=True):
            st.subheader("Iteration 2")
            st.latex(r"c_2=\frac{1.5+2}{2}=1.75")
            st.latex(r"f(1.75)=1.609375")
            st.write("The next interval is:")
            st.latex(r"[1.5,1.75]")

    with st.container(border=True):
        st.subheader("Approximate Root")

        result_col, note_col = st.columns([0.45, 0.55])

        with result_col:
            st.latex(r"x\approx1.52138")

        with note_col:
            st.write(
                "Repeating the same interval-reduction process gives a "
                "more accurate approximation of the root."
            )

    # =====================================================
    # Implementations
    # =====================================================
    st.html(
        """
        <div class="section-kicker">Programming</div>
        <h2 class="section-title">Implementation</h2>
        <p class="section-intro">
            Expand either language to examine a complete implementation.
        </p>
        """
    )

    with st.container(border=True):
        python_column, matlab_column = st.columns(2)

        with python_column:
            with st.expander("🐍 Python Implementation", expanded=False):
                st.code(
                    """
def bisection_method(
    f,
    a,
    b,
    tolerance=1e-6,
    max_iterations=100
):
    if a >= b:
        raise ValueError(
            "The left endpoint must be smaller than the right endpoint."
        )

    if tolerance <= 0:
        raise ValueError(
            "Tolerance must be greater than zero."
        )

    f_a = f(a)
    f_b = f(b)

    if abs(f_a) <= tolerance:
        return a

    if abs(f_b) <= tolerance:
        return b

    if f_a * f_b > 0:
        raise ValueError(
            "f(a) and f(b) must have opposite signs."
        )

    for iteration in range(max_iterations):
        c = (a + b) / 2
        f_c = f(c)
        error_bound = abs(b - a) / 2

        if abs(f_c) <= tolerance or error_bound <= tolerance:
            return c

        if f_a * f_c < 0:
            b = c
            f_b = f_c
        else:
            a = c
            f_a = f_c

    return c


def f(x):
    return x**3 - x - 2


root = bisection_method(
    f=f,
    a=1,
    b=2,
    tolerance=1e-6,
    max_iterations=100
)

print("Approximate root:", root)
                    """,
                    language="python",
                )

        with matlab_column:
            with st.expander("🟠 MATLAB Implementation", expanded=False):
                st.code(
                    """
function root = BisectionMethod(f, a, b, tol, maxIter)

    if a >= b
        error('The left endpoint must be smaller than the right endpoint.');
    end

    if tol <= 0
        error('Tolerance must be greater than zero.');
    end

    fa = f(a);
    fb = f(b);

    if abs(fa) <= tol
        root = a;
        return;
    end

    if abs(fb) <= tol
        root = b;
        return;
    end

    if fa * fb > 0
        error('f(a) and f(b) must have opposite signs.');
    end

    for iteration = 1:maxIter
        c = (a + b) / 2;
        fc = f(c);
        errorBound = abs(b - a) / 2;

        if abs(fc) <= tol || errorBound <= tol
            root = c;
            return;
        end

        if fa * fc < 0
            b = c;
            fb = fc;
        else
            a = c;
            fa = fc;
        end
    end

    root = c;
end
                    """,
                    language="matlab",
                )

    # =====================================================
    # Complexity
    # =====================================================
    st.html(
        """
        <div class="section-kicker">Performance</div>
        <h2 class="section-title">Computational Complexity</h2>
        <p class="section-intro">
            The method uses very little memory and has a predictable number
            of iterations.
        </p>
        """
    )

    complexity1, complexity2, complexity3, complexity4 = st.columns(4)

    with complexity1:
        st.metric("Time Complexity", "O(n)")

    with complexity2:
        st.metric("Memory Complexity", "O(1)")

    with complexity3:
        st.metric("Convergence", "Guaranteed")

    with complexity4:
        st.metric("Order", "Linear")

    with st.container(border=True):
        st.write("The required number of iterations can be estimated by:")

        st.latex(
            r"n\geq\log_2\left(\frac{b_0-a_0}{\varepsilon}\right)"
        )

    # =====================================================
    # Applications
    # =====================================================
    st.html(
        """
        <div class="section-kicker">Engineering context</div>
        <h2 class="section-title">Applications</h2>
        <p class="section-intro">
            Bisection is useful when reliability is more important than speed.
        </p>

        <div class="application-grid-advanced">
            <div class="application-box">
                <strong>Structural Engineering</strong>
                Solving nonlinear equilibrium and deflection equations.
            </div>

            <div class="application-box">
                <strong>Fluid Mechanics</strong>
                Finding pressure, friction, and flow-rate relationships.
            </div>

            <div class="application-box">
                <strong>Heat Transfer</strong>
                Determining temperatures from nonlinear energy equations.
            </div>

            <div class="application-box">
                <strong>Electrical Circuits</strong>
                Locating operating points in nonlinear components.
            </div>

            <div class="application-box">
                <strong>Orbital Mechanics</strong>
                Solving nonlinear position and trajectory equations.
            </div>

            <div class="application-box">
                <strong>Scientific Computing</strong>
                Providing a robust initial root before using faster methods.
            </div>
        </div>
        """
    )

    # =====================================================
    # Advantages and limitations
    # =====================================================
    st.html(
        """
        <div class="section-kicker">Evaluation</div>
        <h2 class="section-title">Advantages and Limitations</h2>
        <p class="section-intro">
            The method is dependable and easy to analyze, but its convergence
            is relatively slow.
        </p>
        """
    )

    advantages, limitations = st.columns(2)

    with advantages:
        st.html(
            """
            <div class="advantage-box">
                <h3>Advantages</h3>
                <ul>
                    <li>Simple to understand and implement.</li>
                    <li>Does not require derivatives.</li>
                    <li>Guaranteed to converge when the conditions hold.</li>
                    <li>Provides a clear theoretical error bound.</li>
                </ul>
            </div>
            """
        )

    with limitations:
        st.html(
            """
            <div class="limitation-box">
                <h3>Limitations</h3>
                <ul>
                    <li>Converges relatively slowly.</li>
                    <li>Requires an interval that brackets a root.</li>
                    <li>Cannot be used safely across discontinuities.</li>
                    <li>May isolate only one of several possible roots.</li>
                </ul>
            </div>
            """
        )

    st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# Footer
# =========================================================
st.html(
    """
    <footer class="footer-ui">
        <div>NM • © 2026 Numerical Methods</div>
        <div>Bisection Method • Root Finding</div>
    </footer>
    """
)