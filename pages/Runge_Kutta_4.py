
import html
import streamlit as st

from components.navigation import navbar
from utilities.ui import load_css

DATA = {'title': 'Fourth-Order Runge-Kutta Method',
 'label': 'CLASSICAL RK4 ODE METHOD',
 'solver': 'Runge_Kutta_4_Solver',
 'quiz': 'Runge_Kutta_4_Quiz',
 'hero': 'Combine four carefully placed slope evaluations to obtain a highly accurate explicit '
         'update.',
 'summary': [('METHOD TYPE', 'Classical explicit RK4'),
             ('SLOPE EVALUATIONS', 'Four per step'),
             ('GLOBAL ORDER', 'Fourth order, O(h⁴)'),
             ('DERIVATIVES OF f', 'Not required')],
 'overview': 'The classical fourth-order Runge-Kutta method (RK4) approximates first-order '
             'initial-value problems. It samples the slope at the beginning, twice near the '
             'midpoint, and at the endpoint, then combines them with weights 1,2,2,1.',
 'foundation': 'The weighted combination is chosen so that the numerical update agrees with the '
               'Taylor expansion of the exact solution through terms of order h⁴, without '
               'explicitly calculating higher derivatives.',
 'foundation_formula': 'k_1=f(x_n,y_n),\\quad k_2=f(x_n+\\tfrac h2,y_n+\\tfrac h2k_1),\\quad '
                       'k_3=f(x_n+\\tfrac h2,y_n+\\tfrac h2k_2),\\quad k_4=f(x_n+h,y_n+h k_3)',
 'conditions': ['A first-order initial-value problem must be available.',
                'The right-hand side should be sufficiently smooth over each step.',
                'The step size must be chosen for both accuracy and stability.',
                'Classical RK4 is explicit and is not generally appropriate for strongly stiff '
                'systems.'],
 'formula_title': 'Classical RK4 Update',
 'formula': 'y_{n+1}=y_n+\\frac h6(k_1+2k_2+2k_3+k_4)',
 'formula_note': 'The two midpoint slopes receive double weight.',
 'accuracy_title': 'Error Order',
 'accuracy_formula': '\\text{local truncation error}=O(h^5),\\qquad \\text{global error}=O(h^4)',
 'accuracy_text': 'For smooth non-stiff problems, halving h often reduces the dominant global '
                  'error by about a factor of sixteen.',
 'comparison_title': 'Cost versus Accuracy',
 'comparison_text': 'RK4 uses four function evaluations per step, but its accuracy often allows '
                    'much larger steps than low-order methods.',
 'steps': ['Record y′=f(x,y), y(x₀)=y₀, and h.',
           'Compute k₁ at the beginning of the step.',
           'Compute k₂ using a half-step based on k₁.',
           'Compute k₃ using another half-step based on k₂.',
           'Compute k₄ at the endpoint using k₃.',
           'Apply the weighted update and repeat.'],
 'example_title': 'One RK4 Step',
 'example_setup': "y'=x+y,\\quad y(0)=1,\\quad h=0.1",
 'example_lines': ['k_1=1',
                   'k_2=f(0.05,1.05)=1.10',
                   'k_3=f(0.05,1.055)=1.105',
                   'k_4=f(0.1,1.1105)=1.2105',
                   'y_1=1+\\frac{0.1}{6}(1+2(1.10)+2(1.105)+1.2105)\\approx1.11034167'],
 'question': 'Why are the RK4 slopes weighted 1, 2, 2, 1?',
 'options': ['The weights make the update fourth-order accurate.',
             "They reduce RK4 to Euler's method.",
             'They guarantee stability for every ODE.'],
 'answer': 'The weights make the update fourth-order accurate.',
 'answer_explanation': 'The stages and weights satisfy order conditions that match the exact '
                       'Taylor expansion through fourth order.',
 'python': 'def runge_kutta_4(f, x0, y0, h, steps):\n'
           '    x = float(x0)\n'
           '    y = float(y0)\n'
           '    values = [(0, x, y)]\n'
           '\n'
           '    for step in range(1, steps + 1):\n'
           '        k1 = f(x, y)\n'
           '        k2 = f(x + h/2, y + h*k1/2)\n'
           '        k3 = f(x + h/2, y + h*k2/2)\n'
           '        k4 = f(x + h, y + h*k3)\n'
           '\n'
           '        y = y + (h/6) * (k1 + 2*k2 + 2*k3 + k4)\n'
           '        x = x + h\n'
           '        values.append((step, x, y))\n'
           '\n'
           '    return values',
 'matlab': 'function values = RungeKutta4(f,x0,y0,h,steps)\n'
           'x = x0;\n'
           'y = y0;\n'
           'values = zeros(steps+1,3);\n'
           'values(1,:) = [0,x,y];\n'
           '\n'
           'for k = 1:steps\n'
           '    k1 = f(x,y);\n'
           '    k2 = f(x+h/2,y+h*k1/2);\n'
           '    k3 = f(x+h/2,y+h*k2/2);\n'
           '    k4 = f(x+h,y+h*k3);\n'
           '\n'
           '    y = y + (h/6)*(k1+2*k2+2*k3+k4);\n'
           '    x = x + h;\n'
           '    values(k+1,:) = [k,x,y];\n'
           'end\n'
           'end',
 'applications': [('Orbital and Mechanical Models',
                   'Integrate smooth dynamical systems with high accuracy.'),
                  ('Electrical and Control Simulation', 'Advance non-stiff state equations.'),
                  ('Scientific Computing',
                   'Use a dependable general-purpose fixed-step integrator.')],
 'advantages': ['Fourth-order global accuracy.',
                'No higher derivatives are required.',
                'Excellent accuracy-to-complexity balance for smooth non-stiff problems.',
                'Widely understood and easy to verify.'],
 'limitations': ['Four function evaluations per step.',
                 'No built-in error estimate or automatic step-size control.',
                 'Can be inefficient or unstable for stiff systems.',
                 'Very long integrations may require adaptive or structure-preserving methods.'],
 'footer': 'Fourth-Order Runge-Kutta • Ordinary Differential Equations'}

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
