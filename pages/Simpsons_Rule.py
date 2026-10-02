
import html
import streamlit as st

from components.navigation import navbar
from utilities.ui import load_css

DATA = {'title': "Simpson's 1/3 Rule",
 'label': 'HIGHER-ORDER NUMERICAL INTEGRATION METHOD',
 'solver': 'Simpsons_Solver',
 'quiz': 'Simpsons_Quiz',
 'hero': 'Approximate a definite integral with quadratic interpolation and the repeating weights '
         '1, 4, 2, 4, …, 1.',
 'summary': [('METHOD TYPE', 'Closed Newton-Cotes rule'),
             ('SUBINTERVALS', 'Must be even'),
             ('GLOBAL ORDER', 'Fourth order, O(h⁴)'),
             ('EXACTNESS', 'Polynomials through degree 3')],
 'overview': "Composite Simpson's 1/3 Rule fits a quadratic polynomial through each group of three "
             'equally spaced nodes and integrates the quadratic exactly. The panels are then added '
             'across the interval.',
 'foundation': 'Every Simpson panel covers two subintervals. Integrating the quadratic interpolant '
               'produces weights 1,4,1 for one panel; adjacent panels combine to give the '
               'composite weights 1,4,2,4,…,2,4,1.',
 'foundation_formula': '\\int_{x_{2j}}^{x_{2j+2}} f(x)\\,dx\\approx\\frac '
                       'h3\\left[f(x_{2j})+4f(x_{2j+1})+f(x_{2j+2})\\right]',
 'conditions': ['The integration limits must be finite and ordered.',
                'The interval must be divided into an even positive number n of equal '
                'subintervals.',
                'The function should be finite and sufficiently smooth on [a,b].',
                'For tabulated data, all nodes must be equally spaced.'],
 'formula_title': "Composite Simpson's 1/3 Rule",
 'formula': 'S_n=\\frac h3\\left[f(x_0)+4\\sum_{\\text{odd '
            '}i}f(x_i)+2\\sum_{\\substack{\\text{even }i\\\\i\\ne0,n}}f(x_i)+f(x_n)\\right]',
 'formula_note': 'Odd interior indices receive weight 4; even interior indices receive weight 2.',
 'accuracy_title': 'Error Order',
 'accuracy_formula': 'E_S=-\\frac{(b-a)}{180}h^4 f^{(4)}(\\xi)\\quad\\text{for some }\\xi\\in(a,b)',
 'accuracy_text': 'For smooth functions, the composite error is O(h⁴). Halving h usually reduces '
                  'the dominant error by about sixteen.',
 'comparison_title': 'Compared with Trapezoidal',
 'comparison_text': "Simpson's rule is usually much more accurate for smooth curves, but it "
                    'requires an even number of equally spaced subintervals.',
 'steps': ['Define f and [a,b].',
           'Choose an even positive number n of subintervals.',
           'Compute h=(b-a)/n and all nodes.',
           'Evaluate f at the nodes.',
           'Apply weights 1,4,2,4,…,2,4,1.',
           'Multiply the weighted sum by h/3.'],
 'example_title': 'Composite Simpson Example',
 'example_setup': '\\int_0^2 x^2\\,dx,\\qquad n=4,\\qquad h=0.5',
 'example_lines': ['f(x_i)=0,0.25,1,2.25,4',
                   'S_4=\\frac{0.5}{3}[0+4(0.25)+2(1)+4(2.25)+4]',
                   'S_4=\\frac{16}{6}=\\frac83\\approx2.666667',
                   '\\text{The result is exact because }x^2\\text{ has degree below 4.}'],
 'question': "Why must n be even for composite Simpson's 1/3 rule?",
 'options': ['Each quadratic panel spans two subintervals.',
             'The rule uses only endpoints.',
             'Odd n guarantees exactness.'],
 'answer': 'Each quadratic panel spans two subintervals.',
 'answer_explanation': 'The interval is grouped into pairs of subintervals, with one quadratic '
                       'interpolant per pair.',
 'python': 'import numpy as np\n'
           '\n'
           'def composite_simpson(f, a, b, n):\n'
           '    if n <= 0 or n % 2 != 0:\n'
           '        raise ValueError("n must be a positive even integer")\n'
           '\n'
           '    x = np.linspace(a, b, n + 1)\n'
           '    y = np.asarray(f(x), dtype=float)\n'
           '    h = (b - a) / n\n'
           '\n'
           '    weighted_sum = (\n'
           '        y[0]\n'
           '        + y[-1]\n'
           '        + 4 * np.sum(y[1:-1:2])\n'
           '        + 2 * np.sum(y[2:-1:2])\n'
           '    )\n'
           '    return (h / 3) * weighted_sum',
 'matlab': 'function I = CompositeSimpson(f,a,b,n)\n'
           'if n <= 0 || floor(n) ~= n || mod(n,2) ~= 0\n'
           "    error('n must be a positive even integer.')\n"
           'end\n'
           '\n'
           'h = (b-a)/n;\n'
           'x = linspace(a,b,n+1);\n'
           'y = f(x);\n'
           '\n'
           'I = (h/3)*( ...\n'
           '    y(1) + y(end) ...\n'
           '    + 4*sum(y(2:2:end-1)) ...\n'
           '    + 2*sum(y(3:2:end-2)) );\n'
           'end',
 'applications': [('Smooth Definite Integrals',
                   'Obtain high accuracy with a moderate number of nodes.'),
                  ('Experimental Data',
                   'Integrate equally spaced smooth measurements when n is even.'),
                  ('Scientific Models',
                   'Approximate accumulated quantities in physics and engineering.')],
 'advantages': ['Fourth-order accuracy for smooth functions.',
                'Exact for all polynomials of degree at most three.',
                'Simple fixed weights.',
                'Often far more accurate than trapezoidal for the same spacing.'],
 'limitations': ['Requires an even number of subintervals.',
                 'Requires equally spaced nodes.',
                 'Less reliable near discontinuities, singularities, or sharp changes.',
                 'It does not provide an error estimate unless additional calculations are made.'],
 'footer': "Simpson's 1/3 Rule • Numerical Integration"}

st.set_page_config(
    page_title=f"{DATA['title']} | Numerical Methods",
    page_icon="📘",
    layout="wide",
)

load_css()
navbar(active_page="learn")

st.markdown('\n<style>\n.method-page { padding-top: 26px; padding-bottom: 12px; }\n.section-kicker { color:#0f766e; font-size:12px; font-weight:900; letter-spacing:1.2px; text-transform:uppercase; margin:26px 0 5px; }\n.section-title { color:#0b1b3a; font-size:25px; font-weight:900; margin:0 0 7px; }\n.section-intro { color:#475569; font-size:15px; line-height:1.65; margin:0 0 18px; }\n.summary-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:14px; margin:4px 0 22px; }\n.summary-card { min-height:112px; padding:18px; border-radius:16px; border:1px solid rgba(15,61,62,.10); box-shadow:0 8px 20px rgba(15,61,62,.06); }\n.summary-card:nth-child(1){background:linear-gradient(135deg,#f0fdfa,#ecfeff);}\n.summary-card:nth-child(2){background:linear-gradient(135deg,#eff6ff,#e0f2fe);}\n.summary-card:nth-child(3){background:linear-gradient(135deg,#f5f3ff,#faf5ff);}\n.summary-card:nth-child(4){background:linear-gradient(135deg,#fff7ed,#fffbeb);}\n.summary-card span{display:block;color:#64748b;font-size:12px;font-weight:800;margin-bottom:8px;}\n.summary-card strong{display:block;color:#0b1b3a;font-size:18px;font-weight:900;line-height:1.25;}\n.condition-list{display:grid;gap:12px;margin-top:10px;}\n.condition-item{display:flex;gap:12px;align-items:flex-start;padding:14px 16px;border-radius:14px;background:#f8fafc;border:1px solid #e2e8f0;}\n.condition-number{width:28px;height:28px;min-width:28px;border-radius:50%;background:linear-gradient(135deg,#14b8a6,#0f766e);color:white;display:grid;place-items:center;font-size:13px;font-weight:900;}\n.condition-item div:last-child{color:#334155;font-size:14px;line-height:1.6;}\n.algorithm-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:12px;margin-top:8px;}\n.algorithm-step{background:#f8fafc;border:1px solid #e2e8f0;border-radius:14px;padding:15px 16px;}\n.algorithm-step span{display:inline-block;color:#0f766e;font-size:12px;font-weight:900;margin-bottom:6px;}\n.algorithm-step p{color:#334155;font-size:14px;line-height:1.55;margin:0;}\n.application-grid-advanced{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;}\n.application-box{background:#f8fafc;border:1px solid #e2e8f0;border-radius:15px;padding:18px;color:#334155;font-size:14px;line-height:1.55;min-height:94px;}\n.application-box strong{display:block;color:#0b1b3a;margin-bottom:5px;font-size:14px;}\n.advantage-box,.limitation-box{border-radius:16px;padding:20px;min-height:215px;}\n.advantage-box{background:#f0fdfa;border:1px solid #99f6e4;}\n.limitation-box{background:#fff7ed;border:1px solid #fed7aa;}\n.advantage-box h3,.limitation-box h3{color:#0b1b3a;font-size:18px;font-weight:900;margin:0 0 10px;}\n.advantage-box li,.limitation-box li{color:#475569;margin-bottom:8px;line-height:1.5;}\ndiv[data-testid="stVerticalBlockBorderWrapper"]{border-radius:18px!important;border:1px solid rgba(15,61,62,.10)!important;box-shadow:0 10px 24px rgba(15,61,62,.06)!important;}\ndiv[data-testid="stExpander"]{border-radius:14px!important;border-color:rgba(15,61,62,.12)!important;overflow:hidden!important;}\n@media(max-width:1000px){.summary-grid,.application-grid-advanced,.algorithm-grid{grid-template-columns:1fr;}}\n</style>\n', unsafe_allow_html=True)

st.html(
    f"""
    <section class="method-hero">
        <div>
            <div class="page-label">{html.escape(DATA['label'])}</div>
            <h1>{html.escape(DATA['title'])}</h1>
            <p>{html.escape(DATA['hero'])}</p>
            <div class="method-actions">
                <a href="/{DATA['solver']}" target="_self" class="btn-primary-ui">Open Solver →</a>
                <a href="/{DATA['quiz']}" target="_self" class="btn-outline-ui">Take Quiz →</a>
            </div>
        </div>
    </section>
    """
)

left_margin, content, right_margin = st.columns([0.035, 0.93, 0.035])

with content:
    st.markdown('<div class="method-page">', unsafe_allow_html=True)

    summary_cards = "".join(
        f'<div class="summary-card"><span>{html.escape(label)}</span>'
        f'<strong>{html.escape(value)}</strong></div>'
        for label, value in DATA["summary"]
    )
    st.html(
        f"""
        <div class="section-kicker">Quick reference</div>
        <h2 class="section-title">Method at a Glance</h2>
        <p class="section-intro">Review the essential properties before studying the derivation and algorithm.</p>
        <div class="summary-grid">{summary_cards}</div>
        """
    )

    st.html(
        """
        <div class="section-kicker">Core concept</div>
        <h2 class="section-title">Overview and Mathematical Foundation</h2>
        <p class="section-intro">Connect the intuitive idea to the mathematical approximation.</p>
        """
    )
    overview_col, foundation_col = st.columns(2)
    with overview_col:
        with st.container(border=True):
            st.subheader("Overview")
            st.write(DATA["overview"])
    with foundation_col:
        with st.container(border=True):
            st.subheader("Mathematical Foundation")
            st.write(DATA["foundation"])
            st.latex(DATA["foundation_formula"])

    st.html(
        """
        <div class="section-kicker">Requirements</div>
        <h2 class="section-title">Conditions and Core Formula</h2>
        <p class="section-intro">Use the method only when its mathematical and data requirements are satisfied.</p>
        """
    )
    conditions_col, formula_col = st.columns([1.15, 0.85])
    with conditions_col:
        with st.container(border=True):
            st.subheader("Required Conditions")
            condition_html = "".join(
                f'<div class="condition-item"><div class="condition-number">{index}</div>'
                f'<div>{html.escape(condition)}</div></div>'
                for index, condition in enumerate(DATA["conditions"], start=1)
            )
            st.html(f'<div class="condition-list">{condition_html}</div>')
    with formula_col:
        with st.container(border=True):
            st.subheader(DATA["formula_title"])
            st.latex(DATA["formula"])
            st.info(DATA["formula_note"])

    st.html(
        """
        <div class="section-kicker">Accuracy</div>
        <h2 class="section-title">Error and Method Selection</h2>
        <p class="section-intro">The order of accuracy explains how the approximation improves as the step or spacing is refined.</p>
        """
    )
    accuracy_col, comparison_col = st.columns(2)
    with accuracy_col:
        with st.container(border=True):
            st.subheader(DATA["accuracy_title"])
            st.latex(DATA["accuracy_formula"])
            st.write(DATA["accuracy_text"])
    with comparison_col:
        with st.container(border=True):
            st.subheader(DATA["comparison_title"])
            st.write(DATA["comparison_text"])

    st.html(
        """
        <div class="section-kicker">Procedure</div>
        <h2 class="section-title">Algorithm</h2>
        <p class="section-intro">Follow the steps in order and validate the input before performing the calculation.</p>
        """
    )
    algorithm_html = "".join(
        f'<div class="algorithm-step"><span>STEP {index}</span><p>{html.escape(step)}</p></div>'
        for index, step in enumerate(DATA["steps"], start=1)
    )
    st.html(f'<div class="algorithm-grid">{algorithm_html}</div>')

    st.html(
        """
        <div class="section-kicker">Worked example</div>
        <h2 class="section-title">Numerical Example</h2>
        <p class="section-intro">The example applies the formula carefully and checks the meaning of the result.</p>
        """
    )
    with st.container(border=True):
        st.markdown(f"**{DATA['example_title']}**")
        st.latex(DATA["example_setup"])
        for line in DATA["example_lines"]:
            st.latex(line)

        st.divider()
        prediction = st.radio(
            DATA["question"],
            DATA["options"],
            index=None,
            key=f"{DATA['solver']}_lesson_prediction",
        )
        reveal = st.checkbox(
            "Reveal answer",
            key=f"{DATA['solver']}_lesson_reveal",
        )
        if reveal:
            if prediction is None:
                st.info("Select an answer first, then compare it with the explanation.")
            elif prediction == DATA["answer"]:
                st.success("Correct.")
            else:
                st.warning("Review the explanation below.")
            st.markdown(f"**Correct answer:** {DATA['answer']}")
            st.write(DATA["answer_explanation"])

    st.html(
        """
        <div class="section-kicker">Programming</div>
        <h2 class="section-title">Implementation</h2>
        <p class="section-intro">Expand either language to examine a complete, validated implementation.</p>
        """
    )
    with st.container(border=True):
        python_col, matlab_col = st.columns(2)
        with python_col:
            with st.expander("🐍 Python Implementation", expanded=False):
                st.code(DATA["python"], language="python")
        with matlab_col:
            with st.expander("🟠 MATLAB Implementation", expanded=False):
                st.code(DATA["matlab"], language="matlab")

    st.html(
        """
        <div class="section-kicker">Use cases</div>
        <h2 class="section-title">Applications</h2>
        <p class="section-intro">These examples show where the method is commonly useful.</p>
        """
    )
    application_html = "".join(
        f'<div class="application-box"><strong>{html.escape(title)}</strong>{html.escape(text)}</div>'
        for title, text in DATA["applications"]
    )
    st.html(f'<div class="application-grid-advanced">{application_html}</div>')

    st.html(
        """
        <div class="section-kicker">Evaluation</div>
        <h2 class="section-title">Advantages and Limitations</h2>
        <p class="section-intro">Choose a numerical method by balancing accuracy, cost, assumptions, and stability.</p>
        """
    )
    advantages_col, limitations_col = st.columns(2)
    with advantages_col:
        advantage_items = "".join(f"<li>{html.escape(item)}</li>" for item in DATA["advantages"])
        st.html(f'<div class="advantage-box"><h3>Advantages</h3><ul>{advantage_items}</ul></div>')
    with limitations_col:
        limitation_items = "".join(f"<li>{html.escape(item)}</li>" for item in DATA["limitations"])
        st.html(f'<div class="limitation-box"><h3>Limitations</h3><ul>{limitation_items}</ul></div>')

    with st.container(border=True):
        st.subheader("Continue Learning")
        navigation_left, navigation_right = st.columns(2)
        with navigation_left:
            if st.button("Open Interactive Solver", use_container_width=True):
                st.switch_page(f"pages/{DATA['solver']}.py")
        with navigation_right:
            if st.button("Take the Quiz", use_container_width=True):
                st.switch_page(f"pages/{DATA['quiz']}.py")

    st.markdown("</div>", unsafe_allow_html=True)

st.html(
    f"""
    <footer class="footer-ui">
        <div>NM • © 2026 Numerical Methods</div>
        <div>{html.escape(DATA['footer'])}</div>
    </footer>
    """
)
