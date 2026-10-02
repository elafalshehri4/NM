import html
import streamlit as st

from components.navigation import navbar
from utilities.ui import load_css

DATA = {'title': 'Midpoint Method Quiz',
 'label': 'MIDPOINT METHOD QUIZ',
 'lesson': 'Midpoint_Method',
 'solver': 'Midpoint_Solver',
 'category': 'Ordinary Differential Equations',
 'key': 'midpoint',
 'description': 'Test midpoint stages, second-order accuracy, function evaluations, update formulas, and '
                'limitations.',
 'questions': [('The explicit Midpoint Method is used for:',
                ['A) ODE initial-value problems',
                 'B) Matrix inversion',
                 'C) Root bracketing',
                 'D) Fourier transforms'],
                'A'),
               ('The first stage is:', ['A) k₁=f(xₙ,yₙ)', 'B) k₁=f(xₙ+h,yₙ)', 'C) k₁=yₙ/h', 'D) k₁=0'], 'A'),
               ('The midpoint state is estimated as:',
                ['A) (xₙ+h/2, yₙ+h k₁/2)', 'B) (xₙ+h, yₙ+h k₁)', 'C) (xₙ, yₙ)', 'D) (h/2,h/2)'],
                'A'),
               ('The full-step update uses:',
                ['A) The starting slope only',
                 'B) The midpoint slope',
                 'C) The endpoint value only',
                 'D) A matrix determinant'],
                'B'),
               ('The global order of explicit midpoint is:', ['A) First', 'B) Second', 'C) Third', 'D) Fourth'], 'B'),
               ('Its local truncation error is:', ['A) O(h)', 'B) O(h²)', 'C) O(h³)', 'D) O(h⁵)'], 'C'),
               ('How many f evaluations are used per step?', ['A) One', 'B) Two', 'C) Three', 'D) Four'], 'B'),
               ('Halving h usually reduces the dominant global error by about:',
                ['A) 2', 'B) 4', 'C) 8', 'D) 16'],
                'B'),
               ('Compared with Euler, midpoint is usually:',
                ['A) More accurate for the same h', 'B) Less accurate', 'C) Identical', 'D) Zero-order'],
                'A'),
               ('The explicit Midpoint Method is a member of:',
                ['A) Runge-Kutta methods',
                 'B) Gaussian quadrature',
                 'C) Newton-Cotes rules',
                 'D) Direct linear solvers'],
                'A'),
               ('The midpoint slope is evaluated at:',
                ['A) An estimated midpoint state',
                 'B) The exact final solution',
                 'C) x=0 always',
                 'D) A random point'],
                'A'),
               ('Which statement is correct?',
                ['A) Explicit midpoint and implicit midpoint are different methods',
                 'B) They are always identical',
                 'C) Midpoint is a root-finding method',
                 'D) Midpoint needs no initial value'],
                'A'),
               ('A limitation of explicit midpoint is:',
                ['A) It may be unsuitable for stiff ODEs',
                 'B) It cannot solve nonlinear f',
                 'C) It requires matrix inversion every step',
                 'D) It is only zeroth-order'],
                'A'),
               ('For one step, yₙ₊₁ equals:', ['A) yₙ+h k₂', 'B) yₙ+h k₁/2', 'C) yₙ-k₂', 'D) h(k₁+k₂)'], 'A'),
               ('The main benefit of the midpoint calculation is:',
                ['A) A better representation of the slope over the interval',
                 'B) Fewer equations',
                 'C) Exactness for every ODE',
                 'D) No step-size choice'],
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
