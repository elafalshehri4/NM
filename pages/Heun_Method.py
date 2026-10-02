
import html
import streamlit as st

from components.navigation import navbar
from utilities.ui import load_css

DATA = {'title': 'Heun Method',
 'label': 'IMPROVED EULER / EXPLICIT TRAPEZOIDAL METHOD',
 'solver': 'Heun_Solver',
 'quiz': 'Heun_Quiz',
 'hero': 'Predict the endpoint with Euler, then correct it using the average of the starting and '
         'predicted endpoint slopes.',
 'summary': [('METHOD TYPE', 'Explicit RK2 predictor-corrector'),
             ('SLOPE EVALUATIONS', 'Two per step'),
             ('GLOBAL ORDER', 'Second order, O(h²)'),
             ('OTHER NAME', 'Improved Euler method')],
 'overview': "Heun's Method solves a first-order initial-value problem using a predictor-corrector "
             'idea. An Euler prediction estimates the endpoint; the starting slope and predicted '
             'endpoint slope are then averaged.',
 'foundation': 'The average of the two endpoint slopes approximates the integral of f over the '
               'step using the trapezoidal rule. Because the endpoint slope is evaluated at a '
               'predicted state, the method remains explicit.',
 'foundation_formula': 'k_1=f(x_n,y_n),\\quad y^{(p)}_{n+1}=y_n+h k_1,\\quad '
                       'k_2=f(x_n+h,y^{(p)}_{n+1})',
 'conditions': ['A first-order ODE and initial condition must be known.',
                'The right-hand side should be continuous and sufficiently smooth.',
                'The Euler predictor must remain in a meaningful region of the model.',
                'A suitable step size is required; the method is not automatically stable for '
                'stiff equations.'],
 'formula_title': 'Heun Corrector',
 'formula': 'y_{n+1}=y_n+\\frac h2(k_1+k_2)',
 'formula_note': 'This is the two-stage explicit trapezoidal Runge-Kutta method.',
 'accuracy_title': 'Error Order',
 'accuracy_formula': '\\text{local truncation error}=O(h^3),\\qquad \\text{global error}=O(h^2)',
 'accuracy_text': 'For smooth problems, halving h usually reduces the dominant global error by '
                  'approximately four.',
 'comparison_title': 'Compared with Midpoint',
 'comparison_text': 'Heun and explicit midpoint are both second-order RK2 methods. They use '
                    'different intermediate slopes and generally produce different approximations.',
 'steps': ['Record y′=f(x,y), the initial value, and h.',
           'Compute the starting slope k₁=f(xₙ,yₙ).',
           'Predict yₙ₊₁ using Euler: yᵖ=yₙ+h k₁.',
           'Compute the predicted endpoint slope k₂=f(xₙ+h,yᵖ).',
           'Correct with the average slope: yₙ₊₁=yₙ+(h/2)(k₁+k₂).',
           'Update x and repeat.'],
 'example_title': 'One Heun Step',
 'example_setup': "y'=x+y,\\quad y(0)=1,\\quad h=0.1",
 'example_lines': ['k_1=f(0,1)=1',
                   'y_1^{(p)}=1+0.1(1)=1.1',
                   'k_2=f(0.1,1.1)=1.2',
                   'y_1=1+\\frac{0.1}{2}(1+1.2)=1.11'],
 'question': 'Why does Heun evaluate a second slope?',
 'options': ['To correct the Euler prediction using an average slope.',
             'To avoid using the ODE.',
             'To make the method first-order.'],
 'answer': 'To correct the Euler prediction using an average slope.',
 'answer_explanation': 'The second slope estimates behavior near the step endpoint, improving the '
                       'average slope over the interval.',
 'python': 'def heun_method(f, x0, y0, h, steps):\n'
           '    x = float(x0)\n'
           '    y = float(y0)\n'
           '    values = [(0, x, y)]\n'
           '\n'
           '    for step in range(1, steps + 1):\n'
           '        k1 = f(x, y)\n'
           '        predictor = y + h * k1\n'
           '        k2 = f(x + h, predictor)\n'
           '\n'
           '        y = y + (h / 2) * (k1 + k2)\n'
           '        x = x + h\n'
           '        values.append((step, x, y))\n'
           '\n'
           '    return values',
 'matlab': 'function values = HeunMethod(f,x0,y0,h,steps)\n'
           'x = x0;\n'
           'y = y0;\n'
           'values = zeros(steps+1,3);\n'
           'values(1,:) = [0,x,y];\n'
           '\n'
           'for k = 1:steps\n'
           '    k1 = f(x,y);\n'
           '    predictor = y + h*k1;\n'
           '    k2 = f(x+h,predictor);\n'
           '\n'
           '    y = y + (h/2)*(k1+k2);\n'
           '    x = x + h;\n'
           '    values(k+1,:) = [k,x,y];\n'
           'end\n'
           'end',
 'applications': [('Engineering Dynamics',
                   'Approximate smooth first-order models with moderate effort.'),
                  ('Predictor-Corrector Teaching',
                   'Demonstrate how a preliminary estimate can improve a step.'),
                  ('Control and Simulation',
                   'Use a simple second-order explicit integrator for non-stiff systems.')],
 'advantages': ['Second-order accuracy.',
                'Clear predictor-corrector interpretation.',
                'Only two function evaluations per step.',
                'Often much more accurate than Euler at the same h.'],
 'limitations': ['The predictor can be poor for large steps.',
                 'Not A-stable and generally unsuitable for stiff systems.',
                 'More expensive than Euler.',
                 'Different methods are also called improved Euler, so the formula should be '
                 'stated explicitly.'],
 'footer': 'Heun Method • Ordinary Differential Equations'}

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
