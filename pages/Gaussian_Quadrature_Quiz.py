import html
import streamlit as st

from components.navigation import navbar
from utilities.ui import load_css

DATA = {'title': 'Gaussian Quadrature Quiz',
 'label': 'GAUSSIAN QUADRATURE QUIZ',
 'lesson': 'Gaussian_Quadrature',
 'solver': 'Gaussian_Quadrature_Solver',
 'category': 'Numerical Integration',
 'key': 'gaussian_quadrature',
 'description': 'Test Gauss–Legendre nodes, weights, polynomial exactness, interval transformation, and practical '
                'limitations.',
 'questions': [('Gauss-Legendre quadrature approximates:',
                ['A) Definite integrals', 'B) Matrix inverses only', 'C) ODE initial values only', 'D) Roots only'],
                'A'),
               ('Its standard interval is:',
                ['A) [-1,1]', 'B) [0,1] only', 'C) (-∞,∞) only', 'D) [a,b] without transformation'],
                'A'),
               ('The nodes are roots of:',
                ['A) Legendre polynomials',
                 'B) Taylor polynomials',
                 'C) Characteristic polynomials only',
                 'D) Random polynomials'],
                'A'),
               ('An n-point Gauss-Legendre rule is exact through degree:',
                ['A) n-1', 'B) n', 'C) 2n-1', 'D) 2n+1'],
                'C'),
               ('The approximation has the form:',
                ['A) Σwᵢf(xᵢ)', 'B) Σf(xᵢ) without weights', 'C) f(a)+f(b) only', 'D) det(A)'],
                'A'),
               ('For [a,b], standard nodes are mapped using:',
                ['A) x=(a+b)/2+(b-a)t/2', 'B) x=a+t', 'C) x=bt', 'D) x=t²'],
                'A'),
               ('The transformed integral includes the factor:',
                ['A) (b-a)/2', 'B) (a+b)/2', 'C) 2/(b-a)', 'D) n²'],
                'A'),
               ('Gaussian nodes are generally:',
                ['A) Not equally spaced', 'B) Always equally spaced', 'C) Repeated', 'D) Integers'],
                'A'),
               ('A 2-point rule is exact for polynomials through degree:', ['A) 1', 'B) 2', 'C) 3', 'D) 4'], 'C'),
               ('Compared with Newton-Cotes rules using the same evaluations, Gaussian quadrature often has:',
                ['A) Higher polynomial exactness', 'B) Lower exactness always', 'C) No weights', 'D) No interval'],
                'A'),
               ('The weights:',
                ["A) Determine each node's contribution",
                 'B) Are always zero',
                 'C) Are random',
                 'D) Are unnecessary'],
                'A'),
               ('For arbitrary existing equally spaced tabulated data, Gaussian quadrature is:',
                ['A) Not directly applicable without interpolation',
                 'B) Always the natural direct rule',
                 'C) Exact with no changes',
                 'D) A derivative formula'],
                'A'),
               ('Endpoint singularities may require:',
                ['A) A transformation or interval splitting',
                 'B) Ignoring the singularity',
                 'C) Repeated nodes',
                 'D) Negative n'],
                'A'),
               ('A common implementation source for nodes and weights is:',
                ['A) A Legendre-Gauss routine such as leggauss',
                 'B) A matrix determinant',
                 "C) Euler's formula",
                 'D) Bisection'],
                'A'),
               ('A key advantage is:',
                ['A) High accuracy with relatively few evaluations for smooth functions',
                 'B) Exactness for every nonsmooth function',
                 'C) No function values',
                 'D) Equal spacing'],
                'A')]}

st.set_page_config(
    page_title=f"{DATA['title']} | Numerical Methods",
    page_icon="📝",
    layout="wide",
)

load_css()
navbar(active_page="quizzes")

st.markdown(
    """
    <style>
    .quiz-page-note { color:#64748b; font-size:14px; line-height:1.6; margin:-6px 0 18px; }
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius:18px!important;
        border:1px solid rgba(15,61,62,.10)!important;
        box-shadow:0 10px 24px rgba(15,61,62,.06)!important;
    }
    div[role="radiogroup"] { gap:.35rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.html(
    f"""
    <section class="quiz-hero">
        <div>
            <div class="page-label">{html.escape(DATA['label'])}</div>
            <h1>{html.escape(DATA['title'])}</h1>
            <p>{html.escape(DATA['description'])}</p>
            <div class="method-actions">
                <a href="/{DATA['lesson']}" target="_self" class="btn-outline-ui">Review Lesson →</a>
                <a href="/{DATA['solver']}" target="_self" class="btn-primary-ui">Open Solver →</a>
            </div>
        </div>
    </section>
    """
)


def student_level(score: int, total_questions: int) -> str:
    percentage = score / total_questions if total_questions else 0
    if percentage == 1:
        return "Excellent"
    if percentage >= 0.8:
        return "Very Good"
    if percentage >= 0.6:
        return "Good"
    if percentage >= 0.4:
        return "Acceptable"
    return "Needs Review"


left_margin, quiz_area, right_margin = st.columns([0.035, 0.93, 0.035])

with quiz_area:
    st.html(
        """
        <div class="section-header quiz-header-row">
            <div>
                <h2>Answer the Questions</h2>
                <p>Select one answer for every question, then submit the quiz.</p>
            </div>
        </div>
        """
    )
    st.markdown(
        '<p class="quiz-page-note">No option is selected automatically. '
        'All 15 questions must be answered before grading.</p>',
        unsafe_allow_html=True,
    )

    user_answers = []
    for question_index, (question, options, answer_key) in enumerate(DATA["questions"]):
        with st.container(border=True):
            st.html(
                f"""
                <div class="quiz-question-title">
                    <span>Question {question_index + 1} of {len(DATA['questions'])}</span>
                    <h3>{html.escape(question)}</h3>
                </div>
                """
            )
            selected = st.radio(
                "Choose your answer:",
                options,
                index=None,
                key=f"{DATA['key']}_question_{question_index}",
                label_visibility="collapsed",
            )
            user_answers.append(selected[0] if selected else None)

    submit_quiz = st.button(
        "Submit Quiz",
        type="primary",
        use_container_width=True,
        key=f"{DATA['key']}_submit",
    )

    if submit_quiz:
        unanswered = sum(answer is None for answer in user_answers)
        if unanswered:
            st.warning(f"Please answer all questions. {unanswered} question(s) remain.")
        else:
            score = sum(
                user_answers[index] == question_data[2]
                for index, question_data in enumerate(DATA["questions"])
            )
            level = student_level(score, len(DATA["questions"]))
            percentage = round(100 * score / len(DATA["questions"]))

            st.html(
                f"""
                <div class="quiz-result-card">
                    <span>Your Score</span>
                    <strong>{score} / {len(DATA['questions'])}</strong>
                    <p>{level} • {percentage}%</p>
                </div>
                """
            )

            if score == len(DATA["questions"]):
                st.balloons()

            with st.expander("Review Answers", expanded=False):
                for question_index, (question, options, answer_key) in enumerate(DATA["questions"]):
                    correct_option = next(option for option in options if option.startswith(answer_key))
                    user_option = next(option for option in options if option.startswith(user_answers[question_index]))
                    st.markdown(f"### Question {question_index + 1}")
                    st.write(question)
                    st.markdown(f"**Your answer:** {user_option}")
                    st.markdown(f"**Correct answer:** {correct_option}")
                    if user_answers[question_index] == answer_key:
                        st.success("Correct")
                    else:
                        st.error("Incorrect")
                    st.divider()

    with st.container(border=True):
        st.subheader("Continue Learning")
        left_navigation, right_navigation = st.columns(2)
        with left_navigation:
            if st.button("Review Lesson", use_container_width=True, key=f"{DATA['key']}_review_lesson"):
                st.switch_page(f"pages/{DATA['lesson']}.py")
        with right_navigation:
            if st.button("Back to Quizzes", use_container_width=True, key=f"{DATA['key']}_back_quizzes"):
                st.switch_page("pages/Quizzes.py")

st.html(
    f"""
    <footer class="footer-ui">
        <div>NM • © 2026 Numerical Methods</div>
        <div>{html.escape(DATA['title'])} • {html.escape(DATA['category'])}</div>
    </footer>
    """
)
