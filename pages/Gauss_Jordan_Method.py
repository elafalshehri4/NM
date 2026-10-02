DATA = {'title': 'Gauss-Jordan Method',
 'category': 'SYSTEMS OF LINEAR EQUATIONS',
 'hero': 'Explore how complete row reduction transforms an augmented matrix directly into reduced row '
         'echelon form.',
 'solver': 'Gauss_Jordan_Solver',
 'quiz': 'Gauss_Jordan_Quiz',
 'footer': 'Gauss-Jordan Method • Linear Systems',
 'summary': [('METHOD TYPE', 'Direct row-reduction solver'),
             ('REQUIREMENT', 'Consistent linear system'),
             ('COMPLEXITY', 'Cubic time'),
             ('OUTPUT FORM', 'Reduced row echelon form')],
 'core_intro': 'Gauss-Jordan eliminates entries both below and above each pivot, so the final augmented '
               'matrix reveals the solution directly.',
 'overview': {'title': 'Overview',
              'text': 'Starting from [A|b], each pivot is normalized to one and every other entry in its '
                      'column is eliminated.',
              'latex': ['[A\\mid\\mathbf{b}]\\longrightarrow[I\\mid\\mathbf{x}]']},
 'foundation': {'title': 'Reduced Row Echelon Form',
                'text': 'In RREF, each leading entry is 1 and is the only nonzero value in its column. Zero '
                        'rows appear at the bottom.',
                'latex': ['\\operatorname{rref}([A\\mid\\mathbf{b}])']},
 'conditions': ['Matrix and vector dimensions must be compatible.',
                'A unique solution requires a pivot in every variable column.',
                'A zero pivot should be replaced by a valid row exchange.',
                'Inconsistent or dependent rows must be identified from the final RREF.'],
 'formula_title': 'Pivot Normalization and Elimination',
 'formula_intro': 'Normalize the pivot row, then eliminate the pivot column in every other row.',
 'formulas': ['R_k\\leftarrow\\frac{R_k}{a_{kk}}', 'R_i\\leftarrow R_i-a_{ik}R_k,\\qquad i\\neq k'],
 'formula_note': 'Unlike ordinary Gaussian elimination, no separate back-substitution stage is required.',
 'analysis': [{'title': 'Direct Solution Reading',
               'text': 'For a nonsingular square system, the left side becomes the identity matrix and the '
                       'final column is the solution vector.',
               'latex': ['[I\\mid\\mathbf{x}]']},
              {'title': 'Computational Cost',
               'text': 'Eliminating above and below every pivot uses more arithmetic than forward '
                       'elimination alone, while remaining cubic.',
               'latex': ['T(n)=O(n^3)']}],
 'criteria': [('Pivot check', 'a_{kk}\\neq0', 'Swap rows whenever the current pivot is zero.'),
              ('Consistency check',
               '[0\\ \\cdots\\ 0\\mid c],\\ c\\neq0',
               'Such a row indicates no solution.'),
              ('Residual check',
               '\\|A\\mathbf{x}-\\mathbf{b}\\|\\ \\text{small}',
               'Verify the extracted solution.')],
 'algorithm': ['Form the augmented matrix [A|b].',
               'Choose or swap in a valid pivot row.',
               'Divide the pivot row so the pivot equals one.',
               'Eliminate the pivot-column entries in all other rows.',
               'Move to the next pivot column and repeat.',
               'Read the solution or classify the system from the RREF.'],
 'example': {'intro': 'Solve 2x+y=5 and x−y=1.',
             'top': [('Augmented matrix', '\\left[\\begin{array}{cc|c}2&1&5\\\\1&-1&1\\end{array}\\right]'),
                     ('Target form', '\\left[\\begin{array}{cc|c}1&0&x\\\\0&1&y\\end{array}\\right]')],
             'steps': [('Normalize and eliminate',
                        ['R_1\\leftarrow\\frac12R_1', 'R_2\\leftarrow R_2-R_1'],
                        'Create the first pivot and eliminate beneath it.'),
                       ('Complete reduction',
                        ['R_2\\leftarrow-\\frac23R_2', 'R_1\\leftarrow R_1-\\frac12R_2'],
                        'Eliminate above the second pivot.')],
             'result': ['\\left[\\begin{array}{cc|c}1&0&2\\\\0&1&1\\end{array}\\right]',
                        'Therefore x=2 and y=1.']},
 'complexity': [('Time Complexity', 'O(n³)'),
                ('Memory', 'O(n²)'),
                ('Back Substitution', 'Not required'),
                ('Output', 'RREF')],
 'applications': [('Linear Systems', 'Solving and classifying systems of equations.'),
                  ('Matrix Inversion', 'Reducing [A|I] to [I|A⁻¹].'),
                  ('Rank Calculation', 'Counting pivot rows in RREF.'),
                  ('Linear Dependence', 'Identifying dependent rows or columns.'),
                  ('Engineering Models', 'Solving small and medium dense systems.'),
                  ('Education', 'Visualizing elementary row operations.')],
 'advantages': ['Provides the solution directly.',
                'Can classify unique, infinite, or inconsistent systems.',
                'Also computes matrix inverses and rank.',
                'Produces a clear canonical matrix form.'],
 'limitations': ['More arithmetic than ordinary Gaussian elimination.',
                 'Requires pivot handling for zero or small pivots.',
                 'Can accumulate round-off error.',
                 'Not efficient for large sparse systems.']}

PYTHON_CODE = '\ndef gauss_jordan(A, b):\n    # Convert values to float\n    A = [[float(value) for value in row] for row in A]\n    b = [float(value) for value in b]\n\n    n = len(b)\n\n    # Form augmented matrix [A | b]\n    Aug = []\n\n    for i in range(n):\n        Aug.append(A[i] + [b[i]])\n\n    for k in range(n):\n        # Partial pivoting to avoid zero pivot\n        pivot_row = k\n\n        for i in range(k + 1, n):\n            if abs(Aug[i][k]) > abs(Aug[pivot_row][k]):\n                pivot_row = i\n\n        if abs(Aug[pivot_row][k]) < 1e-12:\n            raise ValueError("Zero pivot encountered. The system may have no unique solution.")\n\n        # Swap rows if needed\n        if pivot_row != k:\n            Aug[k], Aug[pivot_row] = Aug[pivot_row], Aug[k]\n\n        # Normalize pivot row\n        pivot = Aug[k][k]\n\n        for j in range(n + 1):\n            Aug[k][j] = Aug[k][j] / pivot\n\n        # Eliminate other rows\n        for i in range(n):\n            if i != k:\n                factor = Aug[i][k]\n\n                for j in range(n + 1):\n                    Aug[i][j] = Aug[i][j] - factor * Aug[k][j]\n\n    # Extract solution\n    x = []\n\n    for i in range(n):\n        x.append(Aug[i][-1])\n\n    return x\n\n\n# Example\nA = [\n    [2, 1, -1],\n    [-3, -1, 2],\n    [-2, 1, 2]\n]\n\nb = [8, -11, -3]\n\nsolution = gauss_jordan(A, b)\n\nprint("Solution:")\n\nfor i in range(len(solution)):\n    print(f"x{i + 1} = {solution[i]:.6f}")\n'

MATLAB_CODE = "\nfunction GaussJordanExample()\n\nA = [2 1 -1;\n    -3 -1 2;\n    -2 1 2];\n\nb = [8; -11; -3];\n\nx = GaussJordan(A, b);\n\ndisp('Solution:')\ndisp(x)\n\nend\n\n\nfunction x = GaussJordan(A, b)\n\nn = length(b);\n\n% Form augmented matrix\nAug = [A b];\n\nfor k = 1:n\n\n    % Check pivot\n    if Aug(k,k) == 0\n        error('Zero pivot encountered. Row swapping is needed.');\n    end\n\n    % Normalize pivot row\n    Aug(k,:) = Aug(k,:) / Aug(k,k);\n\n    % Eliminate other rows\n    for i = 1:n\n        if i ~= k\n            factor = Aug(i,k);\n            Aug(i,:) = Aug(i,:) - factor * Aug(k,:);\n        end\n    end\nend\n\n% Extract solution\nx = Aug(:,end);\n\nend\n"


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
