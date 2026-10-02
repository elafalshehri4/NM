import streamlit as st

from components.navigation import navbar
from utilities.ui import load_css

st.set_page_config(page_title='Richardson Extrapolation | Numerical Methods', page_icon="📘", layout="wide")
load_css()
navbar(active_page="learn")

st.markdown('\n<style>\n.numerical-method-page { padding-top: 26px; padding-bottom: 12px; }\n.section-kicker { color: #0f766e; font-size: 12px; font-weight: 900; letter-spacing: 1.2px; text-transform: uppercase; margin-bottom: 5px; }\n.section-title { color: #0b1b3a; font-size: 25px; font-weight: 900; margin: 0 0 7px; }\n.section-intro { color: #475569; font-size: 15px; line-height: 1.65; margin: 0 0 18px; }\n.summary-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin: 4px 0 22px; }\n.summary-card { min-height: 112px; padding: 18px; border-radius: 16px; border: 1px solid rgba(15,61,62,.10); box-shadow: 0 8px 20px rgba(15,61,62,.06); }\n.summary-card:nth-child(1) { background: linear-gradient(135deg,#f0fdfa,#ecfeff); }\n.summary-card:nth-child(2) { background: linear-gradient(135deg,#eff6ff,#e0f2fe); }\n.summary-card:nth-child(3) { background: linear-gradient(135deg,#f5f3ff,#faf5ff); }\n.summary-card:nth-child(4) { background: linear-gradient(135deg,#fff7ed,#fffbeb); }\n.summary-card span { display:block; color:#64748b; font-size:12px; font-weight:800; margin-bottom:8px; }\n.summary-card strong { display:block; color:#0b1b3a; font-size:18px; font-weight:900; line-height:1.25; }\n.condition-list { display:grid; gap:12px; margin-top:10px; }\n.condition-item { display:flex; gap:12px; align-items:flex-start; padding:14px 16px; border-radius:14px; background:#f8fafc; border:1px solid #e2e8f0; }\n.condition-number { width:28px; height:28px; min-width:28px; border-radius:50%; background:linear-gradient(135deg,#14b8a6,#0f766e); color:white; display:grid; place-items:center; font-size:13px; font-weight:900; }\n.condition-item div:last-child { color:#334155; font-size:14px; line-height:1.6; }\n.algorithm-grid { display:grid; grid-template-columns:repeat(2,1fr); gap:12px; margin-top:8px; }\n.algorithm-step { background:#f8fafc; border:1px solid #e2e8f0; border-radius:14px; padding:15px 16px; }\n.algorithm-step span { display:inline-block; color:#0f766e; font-size:12px; font-weight:900; margin-bottom:6px; }\n.algorithm-step p { color:#334155; font-size:14px; line-height:1.55; margin:0; }\n.application-grid-advanced { display:grid; grid-template-columns:repeat(3,1fr); gap:14px; }\n.application-box { background:#f8fafc; border:1px solid #e2e8f0; border-radius:15px; padding:18px; color:#334155; font-size:14px; line-height:1.55; min-height:86px; }\n.application-box strong { display:block; color:#0b1b3a; margin-bottom:5px; font-size:14px; }\n.advantage-box,.limitation-box { border-radius:16px; padding:20px; min-height:190px; }\n.advantage-box { background:#f0fdfa; border:1px solid #99f6e4; }\n.limitation-box { background:#fff7ed; border:1px solid #fed7aa; }\n.advantage-box h3,.limitation-box h3 { color:#0b1b3a; font-size:18px; font-weight:900; margin:0 0 10px; }\n.advantage-box li,.limitation-box li { color:#475569; margin-bottom:8px; line-height:1.5; }\ndiv[data-testid="stVerticalBlockBorderWrapper"] { border-radius:18px !important; border:1px solid rgba(15,61,62,.10) !important; box-shadow:0 10px 24px rgba(15,61,62,.06) !important; }\ndiv[data-testid="stExpander"] { border-radius:14px !important; border-color:rgba(15,61,62,.12) !important; overflow:hidden !important; }\n@media (max-width:1000px) { .summary-grid,.application-grid-advanced,.algorithm-grid { grid-template-columns:1fr; } }\n</style>\n', unsafe_allow_html=True)

st.html(
    """
    <section class="method-hero">
        <div>
            <div class="page-label">ERROR IMPROVEMENT METHOD</div>
            <h1>Richardson Extrapolation</h1>
            <p>Combine two approximations computed with different step sizes to cancel the leading error term and obtain a more accurate result.</p>
            <div class="method-actions">
                <a href="/Richardson_Extrapolation_Solver" target="_self" class="btn-primary-ui">Open Solver →</a>
                <a href="/Richardson_Extrapolation_Quiz" target="_self" class="btn-outline-ui">Take Quiz →</a>
            </div>
        </div>
    </section>
    """
)

left_margin, content, right_margin = st.columns([0.035, 0.93, 0.035])
with content:
    st.markdown('<div class="numerical-method-page">', unsafe_allow_html=True)

    st.html(
        """
        <div class="section-kicker">Quick reference</div>
        <h2 class="section-title">Method at a Glance</h2>
        <p class="section-intro">These four properties summarize Richardson Extrapolation before studying the details.</p>
        <div class="summary-grid">
            <div class="summary-card"><span>METHOD TYPE</span><strong>Extrapolation and error cancellation</strong></div><div class="summary-card"><span>DATA REQUIRED</span><strong>Approximations at h and h/2</strong></div><div class="summary-card"><span>ACCURACY GAIN</span><strong>Raises the formal order</strong></div><div class="summary-card"><span>NEW EXPERIMENTS</span><strong>Not required</strong></div>
        </div>
        """
    )

    st.html(
        """
        <div class="section-kicker">Core concept</div>
        <h2 class="section-title">Overview and Mathematical Foundation</h2>
        <p class="section-intro">Understand the idea first, then connect it to the formula.</p>
        """
    )
    overview_col, foundation_col = st.columns(2)
    with overview_col:
        with st.container(border=True):
            st.subheader("Overview")
            st.write('Richardson Extrapolation improves an existing numerical approximation rather than replacing the original numerical method. It combines results at two step sizes to estimate and remove the dominant truncation error.')
    with foundation_col:
        with st.container(border=True):
            st.subheader('Asymptotic Error Model')
            st.write('Assume an approximation has the form A(h)=L+Ch^p+O(h^{p+1}), where L is the exact value. The two approximations at h and h/2 contain related leading errors that can be eliminated algebraically.')
            st.latex('A(h)=L+Ch^p+\\text{higher-order terms}')

    st.html(
        """
        <div class="section-kicker">Requirements</div>
        <h2 class="section-title">Conditions and Core Formula</h2>
        <p class="section-intro">Use the method only when its data and smoothness requirements are satisfied.</p>
        """
    )
    conditions_col, formula_col = st.columns([1.15, 0.85])
    with conditions_col:
        with st.container(border=True):
            st.subheader("Required Conditions")
            st.html('<div class="condition-list"><div class="condition-item"><div class="condition-number">1</div><div>Two approximations must be available using the same numerical method.</div></div><div class="condition-item"><div class="condition-number">2</div><div>The step sizes should have a known ratio, commonly h and h/2.</div></div><div class="condition-item"><div class="condition-number">3</div><div>The leading error order p must be known or estimated.</div></div><div class="condition-item"><div class="condition-number">4</div><div>The function and approximation should be in the asymptotic convergence region.</div></div></div>')
    with formula_col:
        with st.container(border=True):
            st.subheader('General Richardson Formula')
            st.latex('R=A(h/2)+\\frac{A(h/2)-A(h)}{2^p-1}')
            st.info('For central difference, p=2, so R=[4D(h/2)-D(h)]/3.')

    st.html(
        """
        <div class="section-kicker">Accuracy</div>
        <h2 class="section-title">Error and Method Selection</h2>
        <p class="section-intro">The error order explains how accuracy changes as h becomes smaller.</p>
        """
    )
    error_col, comparison_col = st.columns(2)
    with error_col:
        with st.container(border=True):
            st.subheader('Order Improvement')
            st.latex('O(h^p)\\longrightarrow O(h^{p+1})\\ \\text{or better}')
            st.write('For the centered first-derivative formula, symmetry causes the improved result to be fourth order: O(h²) becomes O(h⁴).')
    with comparison_col:
        with st.container(border=True):
            st.subheader('Why Two Approximations Are Used')
            st.write('A single approximation cannot reveal its leading error coefficient. The predictable relationship between errors at h and h/2 allows that term to be cancelled.')

    st.html(
        """
        <div class="section-kicker">Procedure</div>
        <h2 class="section-title">Algorithm</h2>
        <p class="section-intro">Follow these steps in order to obtain and interpret the approximation.</p>
        <div class="algorithm-grid"><div class="algorithm-step"><span>STEP 1</span><p>Choose a base numerical method and identify its error order p.</p></div><div class="algorithm-step"><span>STEP 2</span><p>Compute the first approximation A(h).</p></div><div class="algorithm-step"><span>STEP 3</span><p>Compute the same approximation using h/2.</p></div><div class="algorithm-step"><span>STEP 4</span><p>Calculate the correction [A(h/2)-A(h)]/(2^p-1).</p></div><div class="algorithm-step"><span>STEP 5</span><p>Add the correction to A(h/2).</p></div><div class="algorithm-step"><span>STEP 6</span><p>Use the extrapolated value and compare successive refinements when needed.</p></div></div>
        """
    )

    st.html(
        """
        <div class="section-kicker">Worked example</div>
        <h2 class="section-title">Numerical Example</h2>
        <p class="section-intro">The example shows the calculation and its relationship to the exact derivative.</p>
        """
    )
    with st.container(border=True):
        st.markdown("**Problem setup**")
        st.latex('f(x)=e^x,\\quad x=0,\\quad \\text{central difference}')
        prediction = st.radio(
            'Why are two step sizes used?',
            ['To cancel the leading error term', 'To make the error larger', 'To avoid evaluating the function'],
            index=None,
            key='richardson_extrapolation_prediction',
        )
        if prediction is not None:
            if prediction == 'To cancel the leading error term':
                st.success("Correct. " + 'The related errors at h and h/2 are combined so the dominant term cancels.')
            else:
                st.warning("Review the data points and error behavior before continuing.")

        step_columns = st.columns(2)
        
        with step_columns[0]:
            with st.container(border=True):
                st.latex('D(0.2)\\approx1.006680')
                st.caption('Central difference with h=0.2.')


        with step_columns[1]:
            with st.container(border=True):
                st.latex('D(0.1)\\approx1.001668')
                st.caption('Central difference with h=0.1.')


        with step_columns[0]:
            with st.container(border=True):
                st.latex('R=\\frac{4D(0.1)-D(0.2)}{3}\\approx0.999997')
                st.caption('Richardson-improved estimate.')


        with step_columns[1]:
            with st.container(border=True):
                st.latex("\\text{Exact }f^{\\prime}(0)=1")
                st.caption('The extrapolated result is extremely close.')


    st.html(
        """
        <div class="section-kicker">Implementation</div>
        <h2 class="section-title">Python and MATLAB Code</h2>
        <p class="section-intro">Both implementations validate the step size and apply the same mathematical formula.</p>
        """
    )
    with st.container(border=True):
        python_column, matlab_column = st.columns(2)
        with python_column:
            with st.expander("🐍 Python Implementation", expanded=False):
                st.code('def richardson(approx_h, approx_half, order):\n    if order <= 0:\n        raise ValueError("order must be positive.")\n    factor = 2**order\n    return approx_half + (approx_half - approx_h) / (factor - 1)\n\nD_h = 1.0066800127\nD_half = 1.0016675002\nR = richardson(D_h, D_half, order=2)\nprint(R)', language="python")
        with matlab_column:
            with st.expander("🟠 MATLAB Implementation", expanded=False):
                st.code("function improved = Richardson(approxH, approxHalf, order)\n    if order <= 0\n        error('order must be positive.');\n    end\n    factor = 2^order;\n    improved = approxHalf + ...\n        (approxHalf - approxH) / (factor - 1);\nend", language="matlab")

    st.html(
        """
        <div class="section-kicker">Performance</div>
        <h2 class="section-title">Computational Characteristics</h2>
        <p class="section-intro">The calculation is inexpensive; accuracy is controlled mainly by the formula and step size.</p>
        """
    )
    metric_columns = st.columns(4)
    
    with metric_columns[0]:
        st.metric('Approximations Needed', '2')


    with metric_columns[1]:
        st.metric('Extra Arithmetic', 'O(1)')


    with metric_columns[2]:
        st.metric('Memory Complexity', 'O(1)')


    with metric_columns[3]:
        st.metric('Typical Gain', 'Higher order')


    st.html(
        """
        <div class="section-kicker">Engineering context</div>
        <h2 class="section-title">Applications</h2>
        <p class="section-intro">These are common situations where the method is useful.</p>
        <div class="application-grid-advanced"><div class="application-box"><strong>Numerical Differentiation</strong>Improve finite-difference derivative estimates.</div><div class="application-box"><strong>Numerical Integration</strong>Build Romberg-style integration tables.</div><div class="application-box"><strong>ODE Solvers</strong>Improve approximations obtained at different step sizes.</div><div class="application-box"><strong>Grid Refinement</strong>Estimate discretization error across mesh sizes.</div><div class="application-box"><strong>Scientific Computing</strong>Increase accuracy without changing the base algorithm.</div><div class="application-box"><strong>Convergence Studies</strong>Estimate observed order and extrapolated limits.</div></div>
        """
    )

    st.html(
        """
        <div class="section-kicker">Evaluation</div>
        <h2 class="section-title">Advantages and Limitations</h2>
        <p class="section-intro">Choose the method by balancing data availability, accuracy, and numerical stability.</p>
        """
    )
    advantages_col, limitations_col = st.columns(2)
    with advantages_col:
        st.html('<div class="advantage-box"><h3>Advantages</h3><ul><li>Substantially improves accuracy with simple arithmetic.</li><li>Reuses an existing numerical method.</li><li>Provides an error-correction interpretation.</li><li>Works across differentiation, integration, and ODE methods.</li></ul></div>')
    with limitations_col:
        st.html('<div class="limitation-box"><h3>Limitations</h3><ul><li>Requires a valid asymptotic error model.</li><li>Needs at least two approximations.</li><li>Can amplify round-off or noisy-data effects.</li><li>The error order p must be known or estimated.</li></ul></div>')

    with st.container(border=True):
        st.subheader("Continue Learning")
        nav1, nav2, nav3 = st.columns(3)
        with nav1:
            if st.button("Open Solver", type="primary", use_container_width=True):
                st.switch_page("pages/Richardson_Extrapolation_Solver.py")
        with nav2:
            if st.button("Take Quiz", use_container_width=True):
                st.switch_page("pages/Richardson_Extrapolation_Quiz.py")
        with nav3:
            if st.button("Back to Learn", use_container_width=True):
                st.switch_page("pages/Learn.py")

    st.markdown("</div>", unsafe_allow_html=True)

st.html(
    """
    <footer class="footer-ui">
        <div>NM • © 2026 Numerical Methods</div>
        <div>Richardson Extrapolation • Numerical Differentiation</div>
    </footer>
    """
)
