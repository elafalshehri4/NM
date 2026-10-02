DATA = {'title': 'Multiple Roots Method',
 'category': 'ROOT FINDING METHOD',
 'hero': 'Learn how a multiplicity-aware Newton update restores fast convergence at repeated roots.',
 'solver': 'Multiple_Roots_Solver',
 'quiz': 'Multiple_Roots_Quiz',
 'footer': 'Multiple Roots Method • Root Finding',
 'summary': [('METHOD TYPE', 'Modified Newton method'),
             ('REQUIREMENT', 'Initial guess and root multiplicity'),
             ('CONVERGENCE', 'Quadratic when multiplicity is correct'),
             ('DERIVATIVE', 'Required')],
 'core_intro': 'Repeated roots slow the standard Newton method because both the function and its derivative '
               'vanish at the root.',
 'overview': {'title': 'Overview',
              'text': 'If r is a root of multiplicity m, the function contains the factor (x−r)^m. The '
                      'multiplicity-aware update compensates for the flattened function shape.',
              'latex': ['f(x)=(x-r)^m g(x),\\qquad g(r)\\neq0']},
 'foundation': {'title': 'Why Standard Newton Slows',
                'text': 'For a repeated root, ordinary Newton-Raphson usually drops from quadratic to linear '
                        'convergence. Multiplying the correction by m restores the expected local speed.',
                'latex': ["x_{n+1}=x_n-\\frac{f(x_n)}{f'(x_n)}"]},
 'conditions': ['The multiplicity m must be known or estimated correctly.',
                'The function and derivative should be continuous near the root.',
                'The derivative must be nonzero away from the root during updates.',
                'The initial guess should be close enough to the repeated root.'],
 'formula_title': 'Multiplicity-Corrected Update',
 'formula_intro': 'Multiply the Newton correction by the known multiplicity.',
 'formulas': ["x_{n+1}=x_n-m\\frac{f(x_n)}{f'(x_n)}"],
 'formula_note': 'When m=1, the formula reduces to the standard Newton-Raphson method.',
 'analysis': [{'title': 'Convergence',
               'text': 'With the correct multiplicity and a suitable starting value, quadratic convergence '
                       'is recovered.',
               'latex': ['|e_{n+1}|\\approx C|e_n|^2']},
              {'title': 'Multiplicity Sensitivity',
               'text': 'Using an incorrect multiplicity can slow convergence or cause unstable updates.',
               'latex': ['m\\in\\{2,3,\\ldots\\}']}],
 'criteria': [('Step-size test', '|x_{n+1}-x_n|<\\varepsilon', 'The corrected update is sufficiently small.'),
              ('Residual test',
               '|f(x_{n+1})|<\\varepsilon',
               'The approximation nearly satisfies the equation.'),
              ('Safety limit', None, 'Stop at the maximum iteration count.')],
 'algorithm': ['Choose x₀ and the known multiplicity m.',
               'Evaluate f(xₙ) and f′(xₙ).',
               'Check that the derivative is safely nonzero.',
               'Apply the multiplicity-corrected Newton formula.',
               'Test the step size and residual.',
               'Update the approximation and repeat.'],
 'example': {'intro': 'Solve f(x)=(x−2)³, whose root r=2 has multiplicity m=3.',
             'top': [('Function', "f(x)=(x-2)^3,\\qquad f'(x)=3(x-2)^2"),
                     ('Initial data', 'x_0=3,\\qquad m=3')],
             'steps': [('Corrected update',
                        ['x_1=3-3\\frac{(3-2)^3}{3(3-2)^2}=2'],
                        'The multiplicity correction reaches the repeated root immediately.'),
                       ('Standard Newton comparison',
                        ['x_1=3-\\frac{1}{3}=2.666667'],
                        'Without the correction, convergence is only linear.')],
             'result': ['x=2', 'The modified method removes the slowdown caused by the repeated root.']},
 'complexity': [('Work per Iteration', 'O(1)'),
                ('Memory', 'O(1)'),
                ('Guarantee', 'Local only'),
                ('Order', 'Quadratic')],
 'applications': [('Repeated Polynomial Roots', 'Solving factors raised to powers greater than one.'),
                  ('Eigenvalue Problems', 'Refining repeated characteristic roots.'),
                  ('Control Systems', 'Handling repeated poles or equilibrium solutions.'),
                  ('Engineering Models', 'Solving equations with tangent contact at the axis.'),
                  ('Symbolic-Numeric Workflows', 'Using known algebraic multiplicity numerically.'),
                  ('Scientific Computing', 'Restoring fast convergence near repeated solutions.')],
 'advantages': ['Restores quadratic convergence when m is correct.',
                'Simple modification of Newton-Raphson.',
                'Efficient and memory-light.',
                'Especially effective for known repeated roots.'],
 'limitations': ['Requires the root multiplicity.',
                 'Still requires a derivative.',
                 'Remains sensitive to the initial guess.',
                 'Incorrect multiplicity may degrade convergence.']}

PYTHON_CODE = '\ndef multiple_roots_method(f, df, x0, m, tol, max_iter):\n    x = float(x0)\n\n    for i in range(max_iter):\n        derivative_value = df(x)\n\n        # Check if derivative is zero\n        if derivative_value == 0:\n            raise ValueError("Derivative is zero.")\n\n        # Multiple Roots formula\n        x_new = x - m * f(x) / derivative_value\n\n        # Check stopping condition\n        if abs(x_new - x) < tol or abs(f(x_new)) < tol:\n            return x_new\n\n        # Update value\n        x = x_new\n\n    # Return the last approximation\n    return x\n\n\n# Example\ndef f(x):\n    return (x - 2) ** 2\n\n\ndef df(x):\n    return 2 * (x - 2)\n\n\nroot = multiple_roots_method(\n    f=f,\n    df=df,\n    x0=3,\n    m=2,\n    tol=0.000001,\n    max_iter=20\n)\n\nprint("Approximate root:")\nprint(f"{root:.6f}")\n'

MATLAB_CODE = "\nfunction MultipleRootsExample()\n\nf = @(x) (x - 2)^2;\ndf = @(x) 2*(x - 2);\n\nx0 = 3;\nm = 2;\ntol = 0.000001;\nmaxIter = 20;\n\nroot = MultipleRootsMethod(f, df, x0, m, tol, maxIter);\n\nfprintf('Approximate root: %.6f\\n', root);\n\nend\n\n\nfunction root = MultipleRootsMethod(f, df, x0, m, tol, maxIter)\n\nx = x0;\n\nfor i = 1:maxIter\n\n    derivativeValue = df(x);\n\n    % Check if derivative is zero\n    if derivativeValue == 0\n        error('Derivative is zero.');\n    end\n\n    % Multiple Roots formula\n    xNew = x - m * f(x) / derivativeValue;\n\n    % Check stopping condition\n    if abs(xNew - x) < tol || abs(f(xNew)) < tol\n        root = xNew;\n        return;\n    end\n\n    % Update value\n    x = xNew;\nend\n\n% Return the last approximation\nroot = x;\n\nend\n"


import html

import streamlit as st

from components.navigation import navbar
from utilities.ui import load_css


st.set_page_config(
    page_title=f"{DATA['title']} | Numerical Methods",
    page_icon="📘",
    layout="wide",
)

load_css()
navbar(active_page="learn")


st.markdown(
    """
    <style>
    .method-page {
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
        min-height: 210px;
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


def section_header(kicker, title, intro):
    st.html(
        f"""
        <div class="section-kicker">{html.escape(kicker)}</div>
        <h2 class="section-title">{html.escape(title)}</h2>
        <p class="section-intro">{html.escape(intro)}</p>
        """
    )


def show_formulas(formulas):
    for formula in formulas:
        st.latex(formula)


hero_solver = DATA["solver"]
hero_quiz = DATA["quiz"]
st.html(
    f"""
    <section class="method-hero">
        <div>
            <div class="page-label">{html.escape(DATA['category'])}</div>
            <h1>{html.escape(DATA['title'])}</h1>
            <p>{html.escape(DATA['hero'])}</p>
            <div class="method-actions">
                <a href="/{hero_solver}" target="_self" class="btn-primary-ui">Open Solver →</a>
                <a href="/{hero_quiz}" target="_self" class="btn-outline-ui">Take Quiz →</a>
            </div>
        </div>
    </section>
    """
)


left_margin, content, right_margin = st.columns([0.035, 0.93, 0.035])

with content:
    st.markdown('<div class="method-page">', unsafe_allow_html=True)

    section_header(
        "Quick reference",
        "Method at a Glance",
        f"These four properties summarize the {DATA['title']} before studying the details.",
    )

    summary_cards = "".join(
        (
            '<div class="summary-card">'
            f'<span>{html.escape(label)}</span>'
            f'<strong>{html.escape(value)}</strong>'
            "</div>"
        )
        for label, value in DATA["summary"]
    )
    st.html(f'<div class="summary-grid">{summary_cards}</div>')

    section_header(
        "Core concept",
        "Overview and Mathematical Foundation",
        DATA["core_intro"],
    )

    overview_col, foundation_col = st.columns(2)
    for column, block in (
        (overview_col, DATA["overview"]),
        (foundation_col, DATA["foundation"]),
    ):
        with column:
            with st.container(border=True):
                st.subheader(block["title"])
                st.write(block["text"])
                show_formulas(block.get("latex", []))

    section_header(
        "Requirements",
        "Conditions and Core Formula",
        "The method depends on valid input conditions and a clearly defined computational update.",
    )

    conditions_col, formula_col = st.columns([1.05, 0.95])

    with conditions_col:
        with st.container(border=True):
            st.subheader("Required Conditions")
            conditions_html = "".join(
                (
                    '<div class="condition-item">'
                    f'<div class="condition-number">{index}</div>'
                    f'<div>{html.escape(condition)}</div>'
                    "</div>"
                )
                for index, condition in enumerate(DATA["conditions"], start=1)
            )
            st.html(f'<div class="condition-list">{conditions_html}</div>')

    with formula_col:
        with st.container(border=True):
            st.subheader(DATA["formula_title"])
            st.write(DATA["formula_intro"])
            show_formulas(DATA["formulas"])
            st.info(DATA["formula_note"])

    section_header(
        "Accuracy",
        "Behavior and Numerical Analysis",
        "Understanding performance, stability, and failure conditions is as important as knowing the main formula.",
    )

    analysis_left, analysis_right = st.columns(2)
    for column, block in (
        (analysis_left, DATA["analysis"][0]),
        (analysis_right, DATA["analysis"][1]),
    ):
        with column:
            with st.container(border=True):
                st.subheader(block["title"])
                st.write(block["text"])
                show_formulas(block.get("latex", []))

    with st.container(border=True):
        st.subheader("Checks and Completion Criteria")
        criterion_columns = st.columns(3)
        for column, criterion in zip(criterion_columns, DATA["criteria"]):
            title, formula, caption = criterion
            with column:
                st.markdown(f"**{title}**")
                if formula:
                    st.latex(formula)
                else:
                    st.markdown("### Validation")
                st.caption(caption)

    section_header(
        "Procedure",
        "Algorithm",
        "Follow the steps in order while checking validity and numerical safety.",
    )

    algorithm_html = "".join(
        (
            '<div class="algorithm-step">'
            f'<span>STEP {index}</span>'
            f'<p>{html.escape(step)}</p>'
            "</div>"
        )
        for index, step in enumerate(DATA["algorithm"], start=1)
    )
    st.html(f'<div class="algorithm-grid">{algorithm_html}</div>')

    section_header(
        "Application",
        "Worked Example",
        DATA["example"]["intro"],
    )

    with st.container(border=True):
        top_columns = st.columns(2)
        for column, (title, formula) in zip(top_columns, DATA["example"]["top"]):
            with column:
                st.markdown(f"**{title}**")
                st.latex(formula)

    example_steps = DATA["example"]["steps"]
    for start in range(0, len(example_steps), 2):
        step_columns = st.columns(2)
        for column, step in zip(step_columns, example_steps[start:start + 2]):
            step_title, formulas, note = step
            with column:
                with st.container(border=True):
                    st.subheader(step_title)
                    show_formulas(formulas)
                    st.write(note)

    with st.container(border=True):
        st.subheader("Result")
        result_col, note_col = st.columns([0.45, 0.55])
        with result_col:
            st.latex(DATA["example"]["result"][0])
        with note_col:
            st.write(DATA["example"]["result"][1])

    section_header(
        "Programming",
        "Implementation",
        "Expand either language to examine a complete implementation.",
    )

    with st.container(border=True):
        python_column, matlab_column = st.columns(2)

        with python_column:
            with st.expander("🐍 Python Implementation", expanded=False):
                st.code(PYTHON_CODE, language="python")

        with matlab_column:
            with st.expander("🟠 MATLAB Implementation", expanded=False):
                st.code(MATLAB_CODE, language="matlab")

    section_header(
        "Performance",
        "Computational Profile",
        "The values below summarize the method's typical work, memory, and numerical character.",
    )

    metric_columns = st.columns(4)
    for column, (label, value) in zip(metric_columns, DATA["complexity"]):
        with column:
            st.metric(label, value)

    section_header(
        "Engineering context",
        "Applications",
        "The method appears in numerical models across science, engineering, and data analysis.",
    )

    application_html = "".join(
        (
            '<div class="application-box">'
            f'<strong>{html.escape(title)}</strong>'
            f'{html.escape(description)}'
            "</div>"
        )
        for title, description in DATA["applications"]
    )
    st.html(f'<div class="application-grid-advanced">{application_html}</div>')

    section_header(
        "Evaluation",
        "Advantages and Limitations",
        "Choose the method based on its assumptions, numerical behavior, and the structure of the problem.",
    )

    advantages_col, limitations_col = st.columns(2)

    with advantages_col:
        advantage_items = "".join(
            f"<li>{html.escape(item)}</li>" for item in DATA["advantages"]
        )
        st.html(
            f'<div class="advantage-box"><h3>Advantages</h3><ul>{advantage_items}</ul></div>'
        )

    with limitations_col:
        limitation_items = "".join(
            f"<li>{html.escape(item)}</li>" for item in DATA["limitations"]
        )
        st.html(
            f'<div class="limitation-box"><h3>Limitations</h3><ul>{limitation_items}</ul></div>'
        )

    st.markdown("</div>", unsafe_allow_html=True)


st.html(
    f"""
    <footer class="footer-ui">
        <div>NM • © 2026 Numerical Methods</div>
        <div>{html.escape(DATA['footer'])}</div>
    </footer>
    """
)
