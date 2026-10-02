import streamlit as st

from components.navigation import navbar
from utilities.ui import load_css

st.set_page_config(page_title='Central Difference | Numerical Methods', page_icon="📘", layout="wide")
load_css()
navbar(active_page="learn")

st.markdown('\n<style>\n.numerical-method-page { padding-top: 26px; padding-bottom: 12px; }\n.section-kicker { color: #0f766e; font-size: 12px; font-weight: 900; letter-spacing: 1.2px; text-transform: uppercase; margin-bottom: 5px; }\n.section-title { color: #0b1b3a; font-size: 25px; font-weight: 900; margin: 0 0 7px; }\n.section-intro { color: #475569; font-size: 15px; line-height: 1.65; margin: 0 0 18px; }\n.summary-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin: 4px 0 22px; }\n.summary-card { min-height: 112px; padding: 18px; border-radius: 16px; border: 1px solid rgba(15,61,62,.10); box-shadow: 0 8px 20px rgba(15,61,62,.06); }\n.summary-card:nth-child(1) { background: linear-gradient(135deg,#f0fdfa,#ecfeff); }\n.summary-card:nth-child(2) { background: linear-gradient(135deg,#eff6ff,#e0f2fe); }\n.summary-card:nth-child(3) { background: linear-gradient(135deg,#f5f3ff,#faf5ff); }\n.summary-card:nth-child(4) { background: linear-gradient(135deg,#fff7ed,#fffbeb); }\n.summary-card span { display:block; color:#64748b; font-size:12px; font-weight:800; margin-bottom:8px; }\n.summary-card strong { display:block; color:#0b1b3a; font-size:18px; font-weight:900; line-height:1.25; }\n.condition-list { display:grid; gap:12px; margin-top:10px; }\n.condition-item { display:flex; gap:12px; align-items:flex-start; padding:14px 16px; border-radius:14px; background:#f8fafc; border:1px solid #e2e8f0; }\n.condition-number { width:28px; height:28px; min-width:28px; border-radius:50%; background:linear-gradient(135deg,#14b8a6,#0f766e); color:white; display:grid; place-items:center; font-size:13px; font-weight:900; }\n.condition-item div:last-child { color:#334155; font-size:14px; line-height:1.6; }\n.algorithm-grid { display:grid; grid-template-columns:repeat(2,1fr); gap:12px; margin-top:8px; }\n.algorithm-step { background:#f8fafc; border:1px solid #e2e8f0; border-radius:14px; padding:15px 16px; }\n.algorithm-step span { display:inline-block; color:#0f766e; font-size:12px; font-weight:900; margin-bottom:6px; }\n.algorithm-step p { color:#334155; font-size:14px; line-height:1.55; margin:0; }\n.application-grid-advanced { display:grid; grid-template-columns:repeat(3,1fr); gap:14px; }\n.application-box { background:#f8fafc; border:1px solid #e2e8f0; border-radius:15px; padding:18px; color:#334155; font-size:14px; line-height:1.55; min-height:86px; }\n.application-box strong { display:block; color:#0b1b3a; margin-bottom:5px; font-size:14px; }\n.advantage-box,.limitation-box { border-radius:16px; padding:20px; min-height:190px; }\n.advantage-box { background:#f0fdfa; border:1px solid #99f6e4; }\n.limitation-box { background:#fff7ed; border:1px solid #fed7aa; }\n.advantage-box h3,.limitation-box h3 { color:#0b1b3a; font-size:18px; font-weight:900; margin:0 0 10px; }\n.advantage-box li,.limitation-box li { color:#475569; margin-bottom:8px; line-height:1.5; }\ndiv[data-testid="stVerticalBlockBorderWrapper"] { border-radius:18px !important; border:1px solid rgba(15,61,62,.10) !important; box-shadow:0 10px 24px rgba(15,61,62,.06) !important; }\ndiv[data-testid="stExpander"] { border-radius:14px !important; border-color:rgba(15,61,62,.12) !important; overflow:hidden !important; }\n@media (max-width:1000px) { .summary-grid,.application-grid-advanced,.algorithm-grid { grid-template-columns:1fr; } }\n</style>\n', unsafe_allow_html=True)

st.html(
    """
    <section class="method-hero">
        <div>
            <div class="page-label">NUMERICAL DIFFERENTIATION</div>
            <h1>Central Difference</h1>
            <p>Use function values on both sides of a point to obtain a balanced, second-order approximation of the first derivative.</p>
            <div class="method-actions">
                <a href="/Central_Difference_Solver" target="_self" class="btn-primary-ui">Open Solver →</a>
                <a href="/Central_Difference_Quiz" target="_self" class="btn-outline-ui">Take Quiz →</a>
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
        <p class="section-intro">These four properties summarize Central Difference before studying the details.</p>
        <div class="summary-grid">
            <div class="summary-card"><span>METHOD TYPE</span><strong>Two-sided finite difference</strong></div><div class="summary-card"><span>DATA REQUIRED</span><strong>f(x - h) and f(x + h)</strong></div><div class="summary-card"><span>ACCURACY</span><strong>Second order, O(h²)</strong></div><div class="summary-card"><span>BEST LOCATION</span><strong>Interior points</strong></div>
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
            st.write('The Central Difference Method estimates the derivative at x by using one point before and one point after x. The symmetric arrangement cancels the leading first-order error term.')
    with foundation_col:
        with st.container(border=True):
            st.subheader('Symmetric Taylor Expansions')
            st.write('Subtracting the Taylor expansions of f(x+h) and f(x-h) cancels all even-powered terms through the leading order, producing a more accurate derivative formula.')
            st.latex("f(x+h)-f(x-h)=2hf^{\\prime}(x)+\\frac{h^3}{3}f^{(3)}(\\xi)")

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
            st.html('<div class="condition-list"><div class="condition-item"><div class="condition-number">1</div><div>Function values must be available on both sides of x.</div></div><div class="condition-item"><div class="condition-number">2</div><div>The points x-h and x+h should be equally spaced from x.</div></div><div class="condition-item"><div class="condition-number">3</div><div>The function should be sufficiently smooth near the target point.</div></div><div class="condition-item"><div class="condition-number">4</div><div>The method is normally used at interior points rather than boundaries.</div></div></div>')
    with formula_col:
        with st.container(border=True):
            st.subheader('Central Difference Formula')
            st.latex("f^{\\prime}(x)\\approx\\frac{f(x+h)-f(x-h)}{2h}")
            st.info('Using a symmetric interval balances the approximation and removes the leading O(h) truncation term.')

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
            st.subheader('Truncation Error')
            st.latex('E_T=-\\frac{h^2}{6}f^{(3)}(\\xi)=O(h^2)')
            st.write('Halving h reduces the leading truncation error by approximately a factor of four.')
    with comparison_col:
        with st.container(border=True):
            st.subheader('Why It Is More Accurate')
            st.write('Forward and backward differences use one side and are first order. Central difference uses both sides and is second order, although it cannot be applied directly at an endpoint without extra data.')

    st.html(
        """
        <div class="section-kicker">Procedure</div>
        <h2 class="section-title">Algorithm</h2>
        <p class="section-intro">Follow these steps in order to obtain and interpret the approximation.</p>
        <div class="algorithm-grid"><div class="algorithm-step"><span>STEP 1</span><p>Choose an interior point x.</p></div><div class="algorithm-step"><span>STEP 2</span><p>Select a positive step size h.</p></div><div class="algorithm-step"><span>STEP 3</span><p>Evaluate or obtain f(x-h).</p></div><div class="algorithm-step"><span>STEP 4</span><p>Evaluate or obtain f(x+h).</p></div><div class="algorithm-step"><span>STEP 5</span><p>Compute [f(x+h)-f(x-h)]/(2h).</p></div><div class="algorithm-step"><span>STEP 6</span><p>Report the approximation and compare it with a smaller h when possible.</p></div></div>
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
        st.latex('f(x)=x^3,\\quad x=2,\\quad h=0.1')
        prediction = st.radio(
            'Why is central difference generally more accurate?',
            ['It uses only f(x)', 'It uses balanced values on both sides', 'It always uses a larger h'],
            index=None,
            key='central_difference_prediction',
        )
        if prediction is not None:
            if prediction == 'It uses balanced values on both sides':
                st.success("Correct. " + 'The symmetric values cancel the leading first-order truncation error.')
            else:
                st.warning("Review the data points and error behavior before continuing.")

        step_columns = st.columns(2)
        
        with step_columns[0]:
            with st.container(border=True):
                st.latex('f(2.1)=9.261')
                st.caption('Value to the right of x.')


        with step_columns[1]:
            with st.container(border=True):
                st.latex('f(1.9)=6.859')
                st.caption('Value to the left of x.')


        with step_columns[0]:
            with st.container(border=True):
                st.latex("f^{\\prime}(2)\\approx\\frac{9.261-6.859}{0.2}=12.01")
                st.caption('Central-difference estimate.')


        with step_columns[1]:
            with st.container(border=True):
                st.latex("\\text{Exact }f^{\\prime}(2)=12,\\quad |E|=0.01")
                st.caption('Small second-order error.')


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
                st.code('def central_difference(f, x, h):\n    if h <= 0:\n        raise ValueError("h must be positive.")\n    return (f(x + h) - f(x - h)) / (2 * h)\n\n\ndef f(x):\n    return x**3\n\napproximation = central_difference(f, x=2.0, h=0.1)\nprint(approximation)', language="python")
        with matlab_column:
            with st.expander("🟠 MATLAB Implementation", expanded=False):
                st.code("function derivative = CentralDifference(f, x, h)\n    if h <= 0\n        error('h must be positive.');\n    end\n    derivative = (f(x + h) - f(x - h)) / (2 * h);\nend", language="matlab")

    st.html(
        """
        <div class="section-kicker">Performance</div>
        <h2 class="section-title">Computational Characteristics</h2>
        <p class="section-intro">The calculation is inexpensive; accuracy is controlled mainly by the formula and step size.</p>
        """
    )
    metric_columns = st.columns(4)
    
    with metric_columns[0]:
        st.metric('Function Evaluations', '2')


    with metric_columns[1]:
        st.metric('Time Complexity', 'O(1)')


    with metric_columns[2]:
        st.metric('Memory Complexity', 'O(1)')


    with metric_columns[3]:
        st.metric('Formal Order', '2')


    st.html(
        """
        <div class="section-kicker">Engineering context</div>
        <h2 class="section-title">Applications</h2>
        <p class="section-intro">These are common situations where the method is useful.</p>
        <div class="application-grid-advanced"><div class="application-box"><strong>Experimental Data</strong>Estimate slopes from measurements surrounding a point.</div><div class="application-box"><strong>Physics Simulations</strong>Compute velocity, acceleration, and spatial gradients.</div><div class="application-box"><strong>Heat Transfer</strong>Approximate interior temperature gradients.</div><div class="application-box"><strong>Fluid Mechanics</strong>Estimate pressure and velocity derivatives inside a grid.</div><div class="application-box"><strong>Signal Processing</strong>Measure local rates of change in sampled signals.</div><div class="application-box"><strong>PDE Discretization</strong>Build accurate centered spatial derivatives.</div></div>
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
        st.html('<div class="advantage-box"><h3>Advantages</h3><ul><li>Second-order accurate with only two evaluations.</li><li>Usually more accurate than one-sided differences.</li><li>Symmetric and easy to implement.</li><li>Works well for smooth interior data.</li></ul></div>')
    with limitations_col:
        st.html('<div class="limitation-box"><h3>Limitations</h3><ul><li>Requires data on both sides of x.</li><li>Not directly suitable at boundaries.</li><li>Very small h can cause cancellation error.</li><li>Accuracy decreases near discontinuities or noisy data.</li></ul></div>')

    with st.container(border=True):
        st.subheader("Continue Learning")
        nav1, nav2, nav3 = st.columns(3)
        with nav1:
            if st.button("Open Solver", type="primary", use_container_width=True):
                st.switch_page("pages/Central_Difference_Solver.py")
        with nav2:
            if st.button("Take Quiz", use_container_width=True):
                st.switch_page("pages/Central_Difference_Quiz.py")
        with nav3:
            if st.button("Back to Learn", use_container_width=True):
                st.switch_page("pages/Learn.py")

    st.markdown("</div>", unsafe_allow_html=True)

st.html(
    """
    <footer class="footer-ui">
        <div>NM • © 2026 Numerical Methods</div>
        <div>Central Difference • Numerical Differentiation</div>
    </footer>
    """
)
