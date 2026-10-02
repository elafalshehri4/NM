import streamlit as st

from components.navigation import navbar
from utilities.ui import load_css

st.set_page_config(page_title='Backward Difference | Numerical Methods', page_icon="📘", layout="wide")
load_css()
navbar(active_page="learn")

st.markdown('\n<style>\n.numerical-method-page { padding-top: 26px; padding-bottom: 12px; }\n.section-kicker { color: #0f766e; font-size: 12px; font-weight: 900; letter-spacing: 1.2px; text-transform: uppercase; margin-bottom: 5px; }\n.section-title { color: #0b1b3a; font-size: 25px; font-weight: 900; margin: 0 0 7px; }\n.section-intro { color: #475569; font-size: 15px; line-height: 1.65; margin: 0 0 18px; }\n.summary-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin: 4px 0 22px; }\n.summary-card { min-height: 112px; padding: 18px; border-radius: 16px; border: 1px solid rgba(15,61,62,.10); box-shadow: 0 8px 20px rgba(15,61,62,.06); }\n.summary-card:nth-child(1) { background: linear-gradient(135deg,#f0fdfa,#ecfeff); }\n.summary-card:nth-child(2) { background: linear-gradient(135deg,#eff6ff,#e0f2fe); }\n.summary-card:nth-child(3) { background: linear-gradient(135deg,#f5f3ff,#faf5ff); }\n.summary-card:nth-child(4) { background: linear-gradient(135deg,#fff7ed,#fffbeb); }\n.summary-card span { display:block; color:#64748b; font-size:12px; font-weight:800; margin-bottom:8px; }\n.summary-card strong { display:block; color:#0b1b3a; font-size:18px; font-weight:900; line-height:1.25; }\n.condition-list { display:grid; gap:12px; margin-top:10px; }\n.condition-item { display:flex; gap:12px; align-items:flex-start; padding:14px 16px; border-radius:14px; background:#f8fafc; border:1px solid #e2e8f0; }\n.condition-number { width:28px; height:28px; min-width:28px; border-radius:50%; background:linear-gradient(135deg,#14b8a6,#0f766e); color:white; display:grid; place-items:center; font-size:13px; font-weight:900; }\n.condition-item div:last-child { color:#334155; font-size:14px; line-height:1.6; }\n.algorithm-grid { display:grid; grid-template-columns:repeat(2,1fr); gap:12px; margin-top:8px; }\n.algorithm-step { background:#f8fafc; border:1px solid #e2e8f0; border-radius:14px; padding:15px 16px; }\n.algorithm-step span { display:inline-block; color:#0f766e; font-size:12px; font-weight:900; margin-bottom:6px; }\n.algorithm-step p { color:#334155; font-size:14px; line-height:1.55; margin:0; }\n.application-grid-advanced { display:grid; grid-template-columns:repeat(3,1fr); gap:14px; }\n.application-box { background:#f8fafc; border:1px solid #e2e8f0; border-radius:15px; padding:18px; color:#334155; font-size:14px; line-height:1.55; min-height:86px; }\n.application-box strong { display:block; color:#0b1b3a; margin-bottom:5px; font-size:14px; }\n.advantage-box,.limitation-box { border-radius:16px; padding:20px; min-height:190px; }\n.advantage-box { background:#f0fdfa; border:1px solid #99f6e4; }\n.limitation-box { background:#fff7ed; border:1px solid #fed7aa; }\n.advantage-box h3,.limitation-box h3 { color:#0b1b3a; font-size:18px; font-weight:900; margin:0 0 10px; }\n.advantage-box li,.limitation-box li { color:#475569; margin-bottom:8px; line-height:1.5; }\ndiv[data-testid="stVerticalBlockBorderWrapper"] { border-radius:18px !important; border:1px solid rgba(15,61,62,.10) !important; box-shadow:0 10px 24px rgba(15,61,62,.06) !important; }\ndiv[data-testid="stExpander"] { border-radius:14px !important; border-color:rgba(15,61,62,.12) !important; overflow:hidden !important; }\n@media (max-width:1000px) { .summary-grid,.application-grid-advanced,.algorithm-grid { grid-template-columns:1fr; } }\n</style>\n', unsafe_allow_html=True)

st.html(
    """
    <section class="method-hero">
        <div>
            <div class="page-label">NUMERICAL DIFFERENTIATION</div>
            <h1>Backward Difference</h1>
            <p>Estimate a first derivative from the current value and the previous point when future data is unavailable.</p>
            <div class="method-actions">
                <a href="/Backward_Difference_Solver" target="_self" class="btn-primary-ui">Open Solver →</a>
                <a href="/Backward_Difference_Quiz" target="_self" class="btn-outline-ui">Take Quiz →</a>
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
        <p class="section-intro">These four properties summarize Backward Difference before studying the details.</p>
        <div class="summary-grid">
            <div class="summary-card"><span>METHOD TYPE</span><strong>One-sided finite difference</strong></div><div class="summary-card"><span>DATA REQUIRED</span><strong>f(x) and f(x - h)</strong></div><div class="summary-card"><span>ACCURACY</span><strong>First order, O(h)</strong></div><div class="summary-card"><span>BEST LOCATION</span><strong>Right boundary or final data</strong></div>
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
            st.write('The Backward Difference Method approximates the slope at x using the secant line through the current point and the previous point. It is especially useful near the end of a data set.')
    with foundation_col:
        with st.container(border=True):
            st.subheader('Taylor-Series Foundation')
            st.write('Expanding f(x-h) about x and rearranging for the first derivative gives the backward formula. The omitted higher-order terms produce a first-order truncation error.')
            st.latex("f(x-h)=f(x)-hf^{\\prime}(x)+\\frac{h^2}{2}f^{\\prime\\prime}(\\xi)")

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
            st.html('<div class="condition-list"><div class="condition-item"><div class="condition-number">1</div><div>The function should be continuous and sufficiently smooth near x.</div></div><div class="condition-item"><div class="condition-number">2</div><div>Function values must be available at x and x-h.</div></div><div class="condition-item"><div class="condition-number">3</div><div>The step size h must be positive and reasonably small.</div></div><div class="condition-item"><div class="condition-number">4</div><div>For tabulated data, neighboring points should be equally spaced.</div></div></div>')
    with formula_col:
        with st.container(border=True):
            st.subheader('Backward Difference Formula')
            st.latex("f^{\\prime}(x)\\approx\\frac{f(x)-f(x-h)}{h}")
            st.info('The change from the previous point to the current point is divided by the step size to estimate the local slope.')

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
            st.latex("E_T=-\\frac{h}{2}f^{\\prime\\prime}(\\xi)=O(h)")
            st.write('The sign of the leading error differs from forward difference, but its magnitude is still first order in h.')
    with comparison_col:
        with st.container(border=True):
            st.subheader('When It Is Appropriate')
            st.write('Backward difference is suitable at the final point of a data set or a right boundary, where a value at x+h is not available.')

    st.html(
        """
        <div class="section-kicker">Procedure</div>
        <h2 class="section-title">Algorithm</h2>
        <p class="section-intro">Follow these steps in order to obtain and interpret the approximation.</p>
        <div class="algorithm-grid"><div class="algorithm-step"><span>STEP 1</span><p>Choose the point x where the derivative is needed.</p></div><div class="algorithm-step"><span>STEP 2</span><p>Select a positive step size h.</p></div><div class="algorithm-step"><span>STEP 3</span><p>Evaluate or obtain f(x).</p></div><div class="algorithm-step"><span>STEP 4</span><p>Evaluate or obtain f(x-h).</p></div><div class="algorithm-step"><span>STEP 5</span><p>Compute [f(x)-f(x-h)]/h.</p></div><div class="algorithm-step"><span>STEP 6</span><p>Report the approximation and assess its error when possible.</p></div></div>
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
        st.latex('f(x)=x^2,\\quad x=2,\\quad h=0.1')
        prediction = st.radio(
            'Which additional function value does the method require?',
            ['f(x-h)', 'f(x+h)', 'Only f(x)'],
            index=None,
            key='backward_difference_prediction',
        )
        if prediction is not None:
            if prediction == 'f(x-h)':
                st.success("Correct. " + 'Backward difference uses the current point and the previous point.')
            else:
                st.warning("Review the data points and error behavior before continuing.")

        step_columns = st.columns(2)
        
        with step_columns[0]:
            with st.container(border=True):
                st.latex('f(2)=4')
                st.caption('Current function value.')


        with step_columns[1]:
            with st.container(border=True):
                st.latex('f(1.9)=3.61')
                st.caption('Previous function value.')


        with step_columns[0]:
            with st.container(border=True):
                st.latex("f^{\\prime}(2)\\approx\\frac{4-3.61}{0.1}=3.9")
                st.caption('Backward-difference estimate.')


        with step_columns[1]:
            with st.container(border=True):
                st.latex("\\text{Exact }f^{\\prime}(2)=4,\\quad |E|=0.1")
                st.caption('Comparison with the analytical derivative.')


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
                st.code('def backward_difference(f, x, h):\n    if h <= 0:\n        raise ValueError("h must be positive.")\n    return (f(x) - f(x - h)) / h\n\n\ndef f(x):\n    return x**2\n\napproximation = backward_difference(f, x=2.0, h=0.1)\nprint(approximation)', language="python")
        with matlab_column:
            with st.expander("🟠 MATLAB Implementation", expanded=False):
                st.code("function derivative = BackwardDifference(f, x, h)\n    if h <= 0\n        error('h must be positive.');\n    end\n    derivative = (f(x) - f(x - h)) / h;\nend", language="matlab")

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
        st.metric('Formal Order', '1')


    st.html(
        """
        <div class="section-kicker">Engineering context</div>
        <h2 class="section-title">Applications</h2>
        <p class="section-intro">These are common situations where the method is useful.</p>
        <div class="application-grid-advanced"><div class="application-box"><strong>Final Measurements</strong>Estimate rates using the latest and previous readings.</div><div class="application-box"><strong>Right Boundaries</strong>Approximate derivatives at the end of a grid.</div><div class="application-box"><strong>Time-Series Data</strong>Estimate a present rate without future observations.</div><div class="application-box"><strong>Control Systems</strong>Use current and past samples in real-time calculations.</div><div class="application-box"><strong>Engineering Tests</strong>Estimate velocity or gradients at the final measurement.</div><div class="application-box"><strong>Finite-Difference Schemes</strong>Construct one-sided right-boundary formulas.</div></div>
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
        st.html('<div class="advantage-box"><h3>Advantages</h3><ul><li>Simple and computationally inexpensive.</li><li>Uses only current and previous data.</li><li>Useful at a right boundary.</li><li>Works with measured data without an explicit derivative.</li></ul></div>')
    with limitations_col:
        st.html('<div class="limitation-box"><h3>Limitations</h3><ul><li>Only first-order accurate.</li><li>Usually less accurate than central difference.</li><li>Sensitive to step-size selection.</li><li>Cannot use information ahead of the current point.</li></ul></div>')

    with st.container(border=True):
        st.subheader("Continue Learning")
        nav1, nav2, nav3 = st.columns(3)
        with nav1:
            if st.button("Open Solver", type="primary", use_container_width=True):
                st.switch_page("pages/Backward_Difference_Solver.py")
        with nav2:
            if st.button("Take Quiz", use_container_width=True):
                st.switch_page("pages/Backward_Difference_Quiz.py")
        with nav3:
            if st.button("Back to Learn", use_container_width=True):
                st.switch_page("pages/Learn.py")

    st.markdown("</div>", unsafe_allow_html=True)

st.html(
    """
    <footer class="footer-ui">
        <div>NM • © 2026 Numerical Methods</div>
        <div>Backward Difference • Numerical Differentiation</div>
    </footer>
    """
)
