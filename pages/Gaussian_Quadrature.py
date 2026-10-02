
import html
import streamlit as st

from components.navigation import navbar
from utilities.ui import load_css

DATA = {'title': 'Gauss-Legendre Quadrature',
 'label': 'OPTIMAL-NODE NUMERICAL INTEGRATION METHOD',
 'solver': 'Gaussian_Quadrature_Solver',
 'quiz': 'Gaussian_Quadrature_Quiz',
 'hero': 'Use specially selected nodes and weights to achieve high polynomial exactness with few '
         'function evaluations.',
 'summary': [('METHOD TYPE', 'Gaussian quadrature'),
             ('STANDARD INTERVAL', '[-1, 1]'),
             ('n-POINT EXACTNESS', 'Degree ≤ 2n−1'),
             ('NODES', 'Roots of Legendre Pₙ')],
 'overview': 'Gauss-Legendre quadrature approximates an integral by a weighted sum of function '
             'values at non-equally spaced nodes. On [-1,1], the nodes are roots of the Legendre '
             'polynomial Pₙ and the weights are chosen for maximum polynomial exactness.',
 'foundation': 'An n-point Gauss-Legendre rule has 2n free node-and-weight parameters. These are '
               'selected so the rule integrates every polynomial through degree 2n−1 exactly.',
 'foundation_formula': '\\int_{-1}^{1}g(t)\\,dt\\approx\\sum_{i=1}^{n}w_i\\,g(t_i),\\qquad '
                       'P_n(t_i)=0',
 'conditions': ['The integrand should be finite at the Gaussian nodes.',
                'For an interval [a,b], transform the standard nodes from [-1,1].',
                'The selected node and weight table must match the chosen number of points.',
                'Discontinuities or singularities should be split or transformed before applying '
                'the rule.'],
 'formula_title': 'Transformation to [a,b]',
 'formula': '\\int_a^b '
            'f(x)\\,dx\\approx\\frac{b-a}{2}\\sum_{i=1}^{n}w_i\\,f\\!\\left(\\frac{a+b}{2}+\\frac{b-a}{2}t_i\\right)',
 'formula_note': 'The factor (b−a)/2 and transformed nodes are essential; omitting them is '
                 'incorrect for a general interval.',
 'accuracy_title': 'Polynomial Exactness',
 'accuracy_formula': '\\deg(p)\\le 2n-1\\quad\\Longrightarrow\\quad Q_n[p]=\\int_{-1}^{1}p(t)\\,dt',
 'accuracy_text': 'For smooth non-polynomial functions, accuracy can improve rapidly as n '
                  'increases, although the exact rate depends on analyticity and interval '
                  'behavior.',
 'comparison_title': 'Compared with Newton-Cotes',
 'comparison_text': 'Gaussian nodes are not equally spaced and generally achieve greater exactness '
                    'than trapezoidal or Simpson rules with the same number of evaluations.',
 'steps': ['Choose the number n of Gauss-Legendre points.',
           'Obtain the standard nodes tᵢ and weights wᵢ on [-1,1].',
           'Map each tᵢ to xᵢ=(a+b)/2+(b-a)tᵢ/2.',
           'Evaluate f at every transformed node.',
           'Compute each weighted contribution wᵢf(xᵢ).',
           'Multiply the sum by (b-a)/2.'],
 'example_title': 'Two-Point Gauss-Legendre Example',
 'example_setup': '\\int_0^2 x^3\\,dx,\\qquad t_{1,2}=\\pm\\frac1{\\sqrt3},\\qquad w_1=w_2=1',
 'example_lines': ['x_{1,2}=1\\pm\\frac1{\\sqrt3}',
                   'Q_2=f(x_1)+f(x_2)',
                   'Q_2=\\left(1-\\frac1{\\sqrt3}\\right)^3+\\left(1+\\frac1{\\sqrt3}\\right)^3=4',
                   '\\int_0^2x^3\\,dx=4'],
 'question': 'What is the highest polynomial degree integrated exactly by an n-point '
             'Gauss-Legendre rule?',
 'options': ['2n−1', 'n−1', '2n+1'],
 'answer': '2n−1',
 'answer_explanation': 'The nodes and weights satisfy moment conditions through polynomial degree '
                       '2n−1.',
 'python': 'import numpy as np\n'
           '\n'
           'def gauss_legendre(f, a, b, n):\n'
           '    if n <= 0:\n'
           '        raise ValueError("n must be a positive integer")\n'
           '\n'
           '    standard_nodes, standard_weights = np.polynomial.legendre.leggauss(n)\n'
           '    midpoint = (a + b) / 2\n'
           '    half_width = (b - a) / 2\n'
           '\n'
           '    transformed_nodes = midpoint + half_width * standard_nodes\n'
           '    values = np.asarray(f(transformed_nodes), dtype=float)\n'
           '\n'
           '    return half_width * np.dot(standard_weights, values)',
 'matlab': 'function I = GaussLegendre(f,a,b,n)\n'
           '% Requires a routine that returns n-point Gauss-Legendre nodes and weights.\n'
           '[t,w] = lgwt(n,-1,1);\n'
           '\n'
           'midpoint = (a+b)/2;\n'
           'halfWidth = (b-a)/2;\n'
           'x = midpoint + halfWidth*t;\n'
           '\n'
           'I = halfWidth*sum(w.*f(x));\n'
           'end',
 'applications': [('Finite Element Analysis',
                   'Evaluate element matrices and load vectors efficiently.'),
                  ('Computational Physics',
                   'Integrate smooth functions with few expensive evaluations.'),
                  ('Spectral and Approximation Methods',
                   'Exploit orthogonal-polynomial nodes and weights.')],
 'advantages': ['Maximum polynomial exactness for n weighted evaluations.',
                'Often very accurate for smooth functions.',
                'No equally spaced grid is required.',
                'Generalizes to weighted Gaussian rules and multidimensional products.'],
 'limitations': ['Nodes and weights must be generated or tabulated.',
                 'Not directly suited to arbitrary pre-existing equally spaced data.',
                 'Discontinuities and endpoint singularities reduce performance.',
                 'A correct interval transformation is required.'],
 'footer': 'Gauss-Legendre Quadrature • Numerical Integration'}

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
