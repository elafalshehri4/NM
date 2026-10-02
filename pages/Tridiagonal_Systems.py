from __future__ import annotations

import html

import streamlit as st

from components.navigation import navbar
from utilities.ui import load_css


DATA = {'title': 'Tridiagonal Systems',
 'page_label': 'LINEAR SYSTEM METHOD',
 'hero_text': 'Learn how the Thomas Algorithm solves a tridiagonal linear system in linear time by '
              'specializing Gaussian elimination to the three nonzero diagonals.',
 'solver_page': 'Tridiagonal_Systems_Solver',
 'quiz_page': 'Tridiagonal_Systems_Quiz',
 'footer_text': 'Tridiagonal Systems • Thomas Algorithm',
 'summary_intro': 'These properties explain why tridiagonal systems can be solved much more efficiently '
                  'than dense systems.',
 'summary': [('METHOD TYPE', 'Direct linear-system method'),
             ('MATRIX STRUCTURE', 'Three nonzero diagonals'),
             ('TIME COMPLEXITY', 'O(n)'),
             ('PIVOTING', 'Not used in the standard form')],
 'core_intro': 'The Thomas Algorithm is Gaussian elimination adapted to a narrow banded matrix.',
 'overview': {'title': 'Overview',
              'text': 'A tridiagonal system contains nonzero coefficients only on the main diagonal, '
                      'the diagonal immediately below it, and the diagonal immediately above it.',
              'formulas': ['a_i x_{i-1}+b_i x_i+c_i x_{i+1}=d_i,\\qquad i=1,\\ldots,n',
                           'A_{ij}=0\\quad\\text{whenever}\\quad |i-j|>1']},
 'foundation': {'title': 'Thomas Algorithm',
                'text': 'The forward sweep removes the lower-diagonal terms while storing only modified '
                        'upper coefficients and modified right-hand-side values. Back substitution then '
                        'recovers the unknowns from the last row upward.',
                'formulas': ['p_i=b_i-a_i c^{\\prime}_{i-1}',
                             'x_i=d^{\\prime}_i-c^{\\prime}_i x_{i+1}']},
 'requirements_intro': 'The matrix structure and every modified pivot must be valid before the method '
                       'can continue.',
 'conditions': ['The coefficient matrix must be square and tridiagonal.',
                'The right-hand-side vector must have one entry for every row.',
                'Every modified pivot must be nonzero and not numerically near zero.',
                'The system must have a unique solution; strict diagonal dominance or symmetric '
                'positive definiteness is sufficient, but not necessary.'],
 'formula': {'title': 'Forward Sweep and Back Substitution',
             'text': 'Using one-based indexing, initialize the first modified coefficients, process '
                     'rows 2 through n, then solve backward.',
             'formulas': ['c^{\\prime}_1=\\frac{c_1}{b_1},\\qquad d^{\\prime}_1=\\frac{d_1}{b_1}',
                          'c^{\\prime}_i=\\frac{c_i}{p_i},\\qquad d^{\\prime}_i=\\frac{d_i-a_i '
                          'd^{\\prime}_{i-1}}{p_i}',
                          'x_n=d^{\\prime}_n,\\qquad x_i=d^{\\prime}_i-c^{\\prime}_i x_{i+1}'],
             'note': 'For the last row, c′ₙ is not required. The standard method stops if any modified '
                     'pivot pᵢ is zero or too small.'},
 'analysis_intro': 'A direct method has no iterative convergence tolerance, so reliability is checked '
                   'through pivots, conditioning, and the residual.',
 'analysis': [{'title': 'Stability Conditions',
               'text': 'The standard Thomas Algorithm is dependable for important classes such as '
                       'strictly diagonally dominant tridiagonal matrices and symmetric '
                       'positive-definite tridiagonal matrices. A general tridiagonal matrix may '
                       'require pivoting or another solver.',
               'formulas': ['|b_i|>|a_i|+|c_i|\\quad\\text{(sufficient condition)}']},
              {'title': 'Residual Verification',
               'text': 'After computing x, substitute it into the original system. A small residual '
                       'confirms that the calculated vector satisfies the equations to numerical '
                       'precision.',
               'formulas': ['r=Ax-d',
                            '\\text{relative '
                            'residual}=\\frac{\\|r\\|_2}{\\|A\\|_2\\|x\\|_2+\\|d\\|_2}']}],
 'checks_title': 'Validation and Completion Checks',
 'checks_intro': 'The solver should verify all three items instead of assuming that every tridiagonal '
                 'input is safe.',
 'checks': [{'title': 'Structure Check',
             'formula': 'A_{ij}=0\\ \\text{for}\\ |i-j|>1',
             'caption': 'Reject coefficients outside the three allowed diagonals.'},
            {'title': 'Modified Pivot Check',
             'formula': '|p_i|>\\tau',
             'caption': 'Stop safely when elimination would divide by a near-zero pivot.'},
            {'title': 'Residual Check',
             'formula': '\\|Ax-d\\|\\approx0',
             'caption': 'Verify the final result with the original, unmodified system.'}],
 'algorithm_intro': 'Only the three diagonal vectors and the right-hand side are needed.',
 'algorithm': ['Validate the dimensions, finite values, and tridiagonal structure.',
               'Extract the lower, main, and upper diagonal vectors.',
               'Initialize the first modified upper coefficient and modified right-hand side.',
               'Perform the forward sweep and check every modified pivot.',
               'Set the final unknown equal to the final modified right-hand-side value.',
               'Use back substitution to compute the remaining unknowns and verify the residual.'],
 'example': {'intro': 'Solve a four-equation tridiagonal system whose exact solution is easy to verify.',
             'setup': [{'label': 'Coefficient matrix',
                        'formulas': ['A=\\begin{bmatrix}2&-1&0&0\\\\-1&2&-1&0\\\\0&-1&2&-1\\\\0&0&-1&2\\end{bmatrix}']},
                       {'label': 'Right-hand side',
                        'formulas': ['d=\\begin{bmatrix}1\\\\0\\\\0\\\\1\\end{bmatrix}']}],
             'setup_note': 'The matrix is tridiagonal and strictly diagonally dominant at the boundary '
                           'rows, with a unique solution.',
             'steps': [{'title': 'Forward Sweep — Rows 1 and 2',
                        'formulas': ['c^{\\prime}_1=-\\frac12,\\qquad d^{\\prime}_1=\\frac12',
                                     'p_2=2-(-1)(-\\tfrac12)=\\frac32',
                                     'c^{\\prime}_2=-\\frac23,\\qquad d^{\\prime}_2=\\frac13'],
                        'text': 'The first subdiagonal entry is eliminated without storing a full new '
                                'matrix.'},
                       {'title': 'Forward Sweep — Rows 3 and 4',
                        'formulas': ['p_3=2-(-1)(-\\tfrac23)=\\frac43,\\quad '
                                     'c^{\\prime}_3=-\\frac34,\\quad d^{\\prime}_3=\\frac14',
                                     'p_4=2-(-1)(-\\tfrac34)=\\frac54,\\quad d^{\\prime}_4=1'],
                        'text': 'The final modified right-hand-side value immediately gives x₄.'},
                       {'title': 'Back Substitution',
                        'formulas': ['x_4=1',
                                     'x_3=\\frac14-(-\\tfrac34)(1)=1',
                                     'x_2=\\frac13-(-\\tfrac23)(1)=1',
                                     'x_1=\\frac12-(-\\tfrac12)(1)=1'],
                        'text': 'Each unknown uses only the next known value.'},
                       {'title': 'Verification',
                        'formulas': ['Ax=d', 'r=Ax-d=0'],
                        'text': 'Substituting the solution into all four equations reproduces the '
                                'original right-hand side.'}],
             'result_title': 'Final Solution',
             'result_formulas': ['x=\\begin{bmatrix}1&1&1&1\\end{bmatrix}^{T}'],
             'result_text': 'The solution satisfies every original equation, and the residual is zero '
                            'in exact arithmetic.'},
 'complexity_intro': 'The banded structure avoids the cubic cost and quadratic storage of a dense '
                     'direct solver.',
 'complexity': [('Time Complexity', 'O(n)'),
                ('Storage', 'O(n)'),
                ('Method Type', 'Direct'),
                ('Matrix Bandwidth', '3 diagonals')],
 'complexity_note': {'text': 'A dense Gaussian-elimination solve requires O(n³) work, whereas the '
                             'Thomas Algorithm performs a constant amount of work per row.',
                     'formulas': ['T(n)=O(n)']},
 'applications_intro': 'Tridiagonal systems appear whenever each unknown interacts mainly with its '
                       'nearest neighbors.',
 'applications': [('Heat Conduction',
                   'One-dimensional finite-difference models of steady and transient temperature.'),
                  ('Diffusion Equations',
                   'Implicit time stepping for one-dimensional diffusion problems.'),
                  ('Cubic Splines', 'Computing spline second derivatives or coefficients.'),
                  ('Structural Mechanics', 'Banded equilibrium equations for chain-like structures.'),
                  ('Electrical Networks', 'Nearest-neighbor circuit and transmission-line models.'),
                  ('Computational Finance',
                   'Finite-difference discretizations of one-dimensional pricing equations.')],
 'evaluation_intro': 'The method is exceptionally efficient, but its no-pivot structure must be '
                     'respected.',
 'advantages': ['Linear time and linear storage.',
                'Simple forward-sweep and back-substitution structure.',
                'Much faster than dense elimination for large tridiagonal systems.',
                'Produces a full residual-based verification of the solution.'],
 'limitations': ['Applies only to tridiagonal matrices.',
                 'The standard form does not perform row pivoting.',
                 'A zero or near-zero modified pivot causes breakdown.',
                 'General ill-conditioned systems may require a more stable banded solver.']}
PYTHON_CODE = 'import numpy as np\n\n\ndef thomas_algorithm(lower, main, upper, rhs, tolerance=1e-12):\n    lower = np.asarray(lower, dtype=float)\n    main = np.asarray(main, dtype=float).copy()\n    upper = np.asarray(upper, dtype=float)\n    rhs = np.asarray(rhs, dtype=float).copy()\n\n    n = main.size\n\n    if n < 2:\n        raise ValueError("At least a 2 × 2 system is required.")\n\n    if lower.size != n - 1 or upper.size != n - 1 or rhs.size != n:\n        raise ValueError("The diagonal and right-hand-side lengths are inconsistent.")\n\n    if not all(np.all(np.isfinite(values)) for values in (lower, main, upper, rhs)):\n        raise ValueError("All coefficients must be finite real numbers.")\n\n    # Forward elimination.\n    for i in range(1, n):\n        if abs(main[i - 1]) <= tolerance:\n            raise ValueError(f"Near-zero modified pivot at row {i}.")\n\n        multiplier = lower[i - 1] / main[i - 1]\n        main[i] -= multiplier * upper[i - 1]\n        rhs[i] -= multiplier * rhs[i - 1]\n\n    if abs(main[-1]) <= tolerance:\n        raise ValueError("Near-zero final modified pivot.")\n\n    # Back substitution.\n    solution = np.zeros(n)\n    solution[-1] = rhs[-1] / main[-1]\n\n    for i in range(n - 2, -1, -1):\n        if abs(main[i]) <= tolerance:\n            raise ValueError(f"Near-zero modified pivot at row {i + 1}.")\n        solution[i] = (rhs[i] - upper[i] * solution[i + 1]) / main[i]\n\n    return solution\n\n\nlower = [-1, -1, -1]\nmain = [2, 2, 2, 2]\nupper = [-1, -1, -1]\nrhs = [1, 0, 0, 1]\n\nx = thomas_algorithm(lower, main, upper, rhs)\nprint("Solution:", x)'
MATLAB_CODE = "function x = ThomasAlgorithm(lower, main, upper, rhs, tolerance)\n\n    if nargin < 5\n        tolerance = 1e-12;\n    end\n\n    lower = lower(:);\n    main = main(:);\n    upper = upper(:);\n    rhs = rhs(:);\n\n    n = length(main);\n\n    if length(lower) ~= n - 1 || length(upper) ~= n - 1 || length(rhs) ~= n\n        error('The diagonal and right-hand-side lengths are inconsistent.');\n    end\n\n    for i = 2:n\n        if abs(main(i - 1)) <= tolerance\n            error('A modified pivot is zero or near zero.');\n        end\n\n        multiplier = lower(i - 1) / main(i - 1);\n        main(i) = main(i) - multiplier * upper(i - 1);\n        rhs(i) = rhs(i) - multiplier * rhs(i - 1);\n    end\n\n    if abs(main(n)) <= tolerance\n        error('The final modified pivot is zero or near zero.');\n    end\n\n    x = zeros(n, 1);\n    x(n) = rhs(n) / main(n);\n\n    for i = n-1:-1:1\n        x(i) = (rhs(i) - upper(i) * x(i + 1)) / main(i);\n    end\nend\n\n% Example:\nlower = [-1; -1; -1];\nmain = [2; 2; 2; 2];\nupper = [-1; -1; -1];\nrhs = [1; 0; 0; 1];\nsolution = ThomasAlgorithm(lower, main, upper, rhs)"


# =========================================================
# Page configuration
# =========================================================
st.set_page_config(
    page_title=f"{DATA['title']} | Numerical Methods",
    page_icon="📘",
    layout="wide",
)

load_css()
navbar(active_page="learn")


# =========================================================
# Page-specific visual styling — aligned with Bisection
# =========================================================
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
        min-height: 220px;
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


# =========================================================
# Hero
# =========================================================
quiz_button = ""
if DATA.get("quiz_page"):
    quiz_button = f"""
        <a href="/{DATA['quiz_page']}" target="_self" class="btn-outline-ui">
            Take Quiz →
        </a>
    """

st.html(
    f"""
    <section class="method-hero">
        <div>
            <div class="page-label">{html.escape(DATA['page_label'])}</div>
            <h1>{html.escape(DATA['title'])}</h1>
            <p>{html.escape(DATA['hero_text'])}</p>
            <div class="method-actions">
                <a href="/{DATA['solver_page']}" target="_self" class="btn-primary-ui">
                    Open Solver →
                </a>
                {quiz_button}
            </div>
        </div>
    </section>
    """
)


# =========================================================
# Main page
# =========================================================
left_margin, content, right_margin = st.columns([0.035, 0.93, 0.035])

with content:
    st.markdown('<div class="method-page">', unsafe_allow_html=True)

    # Quick summary
    summary_cards = "".join(
        f"""
        <div class="summary-card">
            <span>{html.escape(label)}</span>
            <strong>{html.escape(value)}</strong>
        </div>
        """
        for label, value in DATA["summary"]
    )

    st.html(
        f"""
        <div class="section-kicker">Quick reference</div>
        <h2 class="section-title">Method at a Glance</h2>
        <p class="section-intro">{html.escape(DATA['summary_intro'])}</p>
        <div class="summary-grid">{summary_cards}</div>
        """
    )

    # Overview and foundation
    st.html(
        f"""
        <div class="section-kicker">Core concept</div>
        <h2 class="section-title">Overview and Mathematical Foundation</h2>
        <p class="section-intro">{html.escape(DATA['core_intro'])}</p>
        """
    )

    overview_col, foundation_col = st.columns(2)

    for column, section in zip(
        (overview_col, foundation_col),
        (DATA["overview"], DATA["foundation"]),
    ):
        with column:
            with st.container(border=True):
                st.subheader(section["title"])
                st.write(section["text"])
                for formula in section.get("formulas", []):
                    st.latex(formula)
                if section.get("note"):
                    st.info(section["note"])

    # Conditions and core formula
    st.html(
        f"""
        <div class="section-kicker">Requirements</div>
        <h2 class="section-title">Conditions and Core Formula</h2>
        <p class="section-intro">{html.escape(DATA['requirements_intro'])}</p>
        """
    )

    conditions_col, formula_col = st.columns([1.05, 0.95])

    with conditions_col:
        with st.container(border=True):
            st.subheader("Required Conditions")
            conditions_html = "".join(
                f"""
                <div class="condition-item">
                    <div class="condition-number">{index}</div>
                    <div>{html.escape(condition)}</div>
                </div>
                """
                for index, condition in enumerate(DATA["conditions"], start=1)
            )
            st.html(f'<div class="condition-list">{conditions_html}</div>')

    with formula_col:
        with st.container(border=True):
            st.subheader(DATA["formula"]["title"])
            st.write(DATA["formula"]["text"])
            for formula in DATA["formula"]["formulas"]:
                st.latex(formula)
            if DATA["formula"].get("note"):
                st.info(DATA["formula"]["note"])

    # Accuracy and numerical behavior
    st.html(
        f"""
        <div class="section-kicker">Accuracy</div>
        <h2 class="section-title">Error and Numerical Behavior</h2>
        <p class="section-intro">{html.escape(DATA['analysis_intro'])}</p>
        """
    )

    analysis_columns = st.columns(2)
    for column, section in zip(analysis_columns, DATA["analysis"]):
        with column:
            with st.container(border=True):
                st.subheader(section["title"])
                st.write(section["text"])
                for formula in section.get("formulas", []):
                    st.latex(formula)
                if section.get("note"):
                    st.caption(section["note"])

    # Validation checks
    with st.container(border=True):
        st.subheader(DATA["checks_title"])
        st.write(DATA["checks_intro"])
        check_columns = st.columns(len(DATA["checks"]))
        for column, check in zip(check_columns, DATA["checks"]):
            with column:
                st.markdown(f"**{check['title']}**")
                if check.get("formula"):
                    st.latex(check["formula"])
                else:
                    st.markdown(f"### {check['value']}")
                st.caption(check["caption"])

    # Algorithm
    algorithm_html = "".join(
        f"""
        <div class="algorithm-step">
            <span>STEP {index}</span>
            <p>{html.escape(step)}</p>
        </div>
        """
        for index, step in enumerate(DATA["algorithm"], start=1)
    )

    st.html(
        f"""
        <div class="section-kicker">Procedure</div>
        <h2 class="section-title">Algorithm</h2>
        <p class="section-intro">{html.escape(DATA['algorithm_intro'])}</p>
        <div class="algorithm-grid">{algorithm_html}</div>
        """
    )

    # Worked example
    example = DATA["example"]
    st.html(
        f"""
        <div class="section-kicker">Application</div>
        <h2 class="section-title">Worked Example</h2>
        <p class="section-intro">{html.escape(example['intro'])}</p>
        """
    )

    with st.container(border=True):
        setup_columns = st.columns(len(example["setup"]))
        for column, setup_item in zip(setup_columns, example["setup"]):
            with column:
                st.markdown(f"**{setup_item['label']}**")
                for formula in setup_item["formulas"]:
                    st.latex(formula)
        if example.get("setup_note"):
            st.success(example["setup_note"])

    for start in range(0, len(example["steps"]), 2):
        step_columns = st.columns(2)
        for offset, column in enumerate(step_columns):
            step_index = start + offset
            if step_index >= len(example["steps"]):
                continue
            step = example["steps"][step_index]
            with column:
                with st.container(border=True):
                    st.subheader(step["title"])
                    for formula in step.get("formulas", []):
                        st.latex(formula)
                    if step.get("text"):
                        st.write(step["text"])

    with st.container(border=True):
        st.subheader(example["result_title"])
        result_col, note_col = st.columns([0.46, 0.54])
        with result_col:
            for formula in example["result_formulas"]:
                st.latex(formula)
        with note_col:
            st.write(example["result_text"])

    # Implementations
    st.html(
        """
        <div class="section-kicker">Programming</div>
        <h2 class="section-title">Implementation</h2>
        <p class="section-intro">
            Expand either language to examine a complete, validated implementation.
        </p>
        """
    )

    with st.container(border=True):
        python_column, matlab_column = st.columns(2)
        with python_column:
            with st.expander("🐍 Python Implementation", expanded=False):
                st.code(PYTHON_CODE, language="python")
        with matlab_column:
            with st.expander("🟠 MATLAB Implementation", expanded=False):
                st.code(MATLAB_CODE, language="matlab")

    # Complexity
    st.html(
        f"""
        <div class="section-kicker">Performance</div>
        <h2 class="section-title">Computational Complexity</h2>
        <p class="section-intro">{html.escape(DATA['complexity_intro'])}</p>
        """
    )

    complexity_columns = st.columns(4)
    for column, metric in zip(complexity_columns, DATA["complexity"]):
        with column:
            st.metric(metric[0], metric[1])

    with st.container(border=True):
        st.write(DATA["complexity_note"]["text"])
        for formula in DATA["complexity_note"].get("formulas", []):
            st.latex(formula)

    # Applications
    application_html = "".join(
        f"""
        <div class="application-box">
            <strong>{html.escape(title)}</strong>
            {html.escape(description)}
        </div>
        """
        for title, description in DATA["applications"]
    )

    st.html(
        f"""
        <div class="section-kicker">Engineering context</div>
        <h2 class="section-title">Applications</h2>
        <p class="section-intro">{html.escape(DATA['applications_intro'])}</p>
        <div class="application-grid-advanced">{application_html}</div>
        """
    )

    # Advantages and limitations
    advantages_html = "".join(
        f"<li>{html.escape(item)}</li>" for item in DATA["advantages"]
    )
    limitations_html = "".join(
        f"<li>{html.escape(item)}</li>" for item in DATA["limitations"]
    )

    st.html(
        f"""
        <div class="section-kicker">Evaluation</div>
        <h2 class="section-title">Advantages and Limitations</h2>
        <p class="section-intro">{html.escape(DATA['evaluation_intro'])}</p>
        """
    )

    advantages_col, limitations_col = st.columns(2)
    with advantages_col:
        st.html(
            f"""
            <div class="advantage-box">
                <h3>Advantages</h3>
                <ul>{advantages_html}</ul>
            </div>
            """
        )
    with limitations_col:
        st.html(
            f"""
            <div class="limitation-box">
                <h3>Limitations</h3>
                <ul>{limitations_html}</ul>
            </div>
            """
        )

    st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# Footer
# =========================================================
st.html(
    f"""
    <footer class="footer-ui">
        <div>NM • © 2026 Numerical Methods</div>
        <div>{html.escape(DATA['footer_text'])}</div>
    </footer>
    """
)
