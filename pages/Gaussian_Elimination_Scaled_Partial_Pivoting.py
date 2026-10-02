DATA = {'title': 'Gaussian Elimination with Scaled Partial Pivoting',
 'category': 'SYSTEMS OF LINEAR EQUATIONS',
 'hero': 'Learn how scaled pivot selection improves Gaussian elimination for matrices whose rows have very '
         'different magnitudes.',
 'solver': 'Gaussian_Elimination_Scaled_Partial_Pivoting_Solver',
 'quiz': 'Gaussian_Elimination_Scaled_Partial_Pivoting_Quiz',
 'footer': 'Scaled Partial Pivoting • Linear Systems',
 'summary': [('METHOD TYPE', 'Pivoted direct solver'),
             ('REQUIREMENT', 'Square nonsingular matrix'),
             ('COMPLEXITY', 'Cubic time'),
             ('PIVOT RULE', 'Largest scaled ratio')],
 'core_intro': 'Scaled partial pivoting compares each pivot candidate with the largest coefficient in its '
               'own row before choosing the pivot row.',
 'overview': {'title': 'Overview',
              'text': 'The method performs Gaussian elimination but reorders rows at every pivot stage to '
                      'reduce the effect of small or poorly scaled pivots.',
              'latex': ['A\\mathbf{x}=\\mathbf{b}']},
 'foundation': {'title': 'Scale-Aware Pivoting',
                'text': 'A large absolute coefficient may still be small relative to the other values in its '
                        'row. Scaling makes the comparison dimensionless and more meaningful.',
                'latex': ['s_i=\\max_j|a_{ij}|']},
 'conditions': ['A must be square and compatible with b.',
                'No row may have a zero scaling factor.',
                'A usable nonzero pivot must exist at every stage.',
                'The system must have a unique solution.'],
 'formula_title': 'Pivot Ratio and Elimination',
 'formula_intro': 'Choose the candidate row with the largest scaled ratio, swap it into the pivot position, '
                  'and eliminate below it.',
 'formulas': ['r_i=\\frac{|a_{ik}|}{s_i}', 'p=\\arg\\max_{i\\ge k}r_i', 'm_{ik}=\\frac{a_{ik}}{a_{kk}}'],
 'formula_note': 'Scaling factors move with their rows whenever a row exchange occurs.',
 'analysis': [{'title': 'Stability Improvement',
               'text': 'The method reduces the chance of dividing by a coefficient that is tiny relative to '
                       'the rest of its row.',
               'latex': ['\\max_{i\\ge k}\\frac{|a_{ik}|}{s_i}']},
              {'title': 'Computational Cost',
               'text': 'Pivot comparisons add only lower-order work; dense elimination remains cubic '
                       'overall.',
               'latex': ['T(n)=O(n^3)']}],
 'criteria': [('Dimension check', None, 'A must be square and compatible with b.'),
              ('Scale/pivot check', 's_i>0,\\quad |a_{kk}|>0', 'Reject zero rows and unusable pivots.'),
              ('Residual check',
               '\\|A\\mathbf{x}-\\mathbf{b}\\|\\ \\text{small}',
               'Confirm the solution after back substitution.')],
 'algorithm': ['Form [A|b] and compute one scale factor per row.',
               'At pivot column k, calculate each candidate scaled ratio.',
               'Select the row with the largest ratio.',
               'Swap the selected row and its scale factor into position k.',
               'Eliminate all entries below the pivot.',
               'Repeat, then perform back substitution.'],
 'example': {'intro': 'Consider a poorly scaled first pivot in a 2×2 system.',
             'top': [('System', '\\left[\\begin{array}{cc|c}0.001&1&1.001\\\\1&1&2\\end{array}\\right]'),
                     ('Scale factors', 's_1=1,\\qquad s_2=1')],
             'steps': [('Scaled ratios', ['r_1=0.001', 'r_2=1'], 'The second row provides the safer pivot.'),
                       ('Row exchange',
                        ['R_1\\leftrightarrow R_2'],
                        'Elimination proceeds after placing the strongest scaled pivot first.')],
             'result': ['\\mathbf{x}=\\begin{bmatrix}1\\\\1\\end{bmatrix}',
                        'The row exchange avoids division by the tiny value 0.001.']},
 'complexity': [('Time Complexity', 'O(n³)'),
                ('Memory', 'O(n²)'),
                ('Method', 'Direct'),
                ('Stability', 'Improved')],
 'applications': [('Poorly Scaled Systems', 'Handling rows with very different coefficient magnitudes.'),
                  ('Engineering Simulation', 'Solving stable dense linear models.'),
                  ('Scientific Computing', 'Reducing round-off amplification.'),
                  ('Circuit and Network Models', 'Solving coupled equations with mixed units.'),
                  ('Structural Analysis', 'Improving reliability in stiffness systems.'),
                  ('General Linear Algebra', 'Serving as a safer default than naive elimination.')],
 'advantages': ['More stable than naive Gaussian elimination.',
                'Accounts for row scaling, not only absolute pivot size.',
                'Requires only modest extra work.',
                'Retains the familiar elimination/back-substitution structure.'],
 'limitations': ['Still costs O(n³) for dense systems.',
                 'Does not fix singular or severely ill-conditioned systems.',
                 'More bookkeeping is required.',
                 'Numerical accuracy still depends on floating-point conditioning.']}

PYTHON_CODE = 'def scaled_partial_pivoting(A, b):\n    A = [[float(value) for value in row] for row in A]\n    b = [float(value) for value in b]\n\n    n = len(b)\n\n    if len(A) != n:\n        raise ValueError("Matrix A and vector b dimensions do not match.")\n\n    for row in A:\n        if len(row) != n:\n            raise ValueError("Matrix A must be square.")\n\n    s = []\n\n    for i in range(n):\n        scale = max(abs(value) for value in A[i])\n        if scale == 0:\n            raise ValueError("A row has all zero elements.")\n        s.append(scale)\n\n    for k in range(n - 1):\n        pivot_row = k\n        max_ratio = abs(A[k][k]) / s[k]\n\n        for i in range(k + 1, n):\n            ratio = abs(A[i][k]) / s[i]\n            if ratio > max_ratio:\n                max_ratio = ratio\n                pivot_row = i\n\n        if abs(A[pivot_row][k]) < 1e-12:\n            raise ValueError("Zero pivot encountered. The system may have no unique solution.")\n\n        if pivot_row != k:\n            A[k], A[pivot_row] = A[pivot_row], A[k]\n            b[k], b[pivot_row] = b[pivot_row], b[k]\n            s[k], s[pivot_row] = s[pivot_row], s[k]\n\n        for i in range(k + 1, n):\n            factor = A[i][k] / A[k][k]\n            for j in range(k, n):\n                A[i][j] = A[i][j] - factor * A[k][j]\n            b[i] = b[i] - factor * b[k]\n\n    x = [0.0 for _ in range(n)]\n\n    for i in range(n - 1, -1, -1):\n        if abs(A[i][i]) < 1e-12:\n            raise ValueError("Zero pivot encountered during back substitution.")\n        total = 0\n        for j in range(i + 1, n):\n            total += A[i][j] * x[j]\n        x[i] = (b[i] - total) / A[i][i]\n\n    return x\n\n\n# Example\nA = [[2, 1, -1], [-3, -1, 2], [-2, 1, 2]]\nb = [8, -11, -3]\n\nsolution = scaled_partial_pivoting(A, b)\nprint("Solution:")\nfor i in range(len(solution)):\n    print(f"x{i + 1} = {solution[i]:.6f}")'

MATLAB_CODE = "function ScaledPartialPivotingExample()\n\nA = [2 1 -1; -3 -1 2; -2 1 2];\nb = [8; -11; -3];\n\nx = ScaledPartialPivoting(A, b);\n\ndisp('Solution:')\ndisp(x)\n\nend\n\nfunction x = ScaledPartialPivoting(A, b)\n\nn = length(b);\ns = max(abs(A), [], 2);\n\nfor k = 1:n-1\n    [~, p] = max(abs(A(k:n,k))./s(k:n));\n    p = p + k - 1;\n    if p ~= k\n        A([k p],:) = A([p k],:);\n        b([k p]) = b([p k]);\n        s([k p]) = s([p k]);\n    end\n    for i = k+1:n\n        factor = A(i,k)/A(k,k);\n        A(i,:) = A(i,:) - factor*A(k,:);\n        b(i) = b(i) - factor*b(k);\n    end\nend\n\nx = zeros(n,1);\nfor i = n:-1:1\n    x(i) = (b(i) - A(i,i+1:n)*x(i+1:n))/A(i,i);\nend\n\nend"


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
