import html
import streamlit as st

from components.navigation import navbar
from utilities.ui import load_css

DATA = {'title': 'Heun Method Quiz',
 'label': 'HEUN METHOD QUIZ',
 'lesson': 'Heun_Method',
 'solver': 'Heun_Solver',
 'category': 'Ordinary Differential Equations',
 'key': 'heun',
 'description': 'Test the predictor–corrector update, two slope evaluations, second-order accuracy, and step-size '
                'behavior.',
 'questions': [("Heun's Method is also commonly called:",
                ['A) Improved Euler Method', 'B) Bisection Method', 'C) Gauss-Jordan Method', "D) Simpson's Rule"],
                'A'),
               ('The predictor is normally computed with:',
                ['A) An Euler step', "B) Simpson's rule", 'C) A matrix inverse', 'D) A central difference'],
                'A'),
               ('The first slope is:', ['A) k₁=f(xₙ,yₙ)', 'B) k₁=f(xₙ+h,yₙ)', 'C) k₁=h²', 'D) k₁=0'], 'A'),
               ('The predicted endpoint value is:',
                ['A) yᵖ=yₙ+h k₁', 'B) yᵖ=yₙ-h k₁', 'C) yᵖ=k₁/h', 'D) yᵖ=yₙ'],
                'A'),
               ('The second slope is evaluated at:',
                ['A) The predicted endpoint',
                 'B) The exact midpoint only',
                 'C) The initial x with no prediction',
                 'D) Infinity'],
                'A'),
               ('The Heun corrector is:',
                ['A) yₙ₊₁=yₙ+(h/2)(k₁+k₂)', 'B) yₙ₊₁=yₙ+h k₁', 'C) yₙ₊₁=k₁-k₂', 'D) yₙ₊₁=h²'],
                'A'),
               ("Heun's global order is:", ['A) First', 'B) Second', 'C) Third', 'D) Fourth'], 'B'),
               ('The local truncation error is:', ['A) O(h)', 'B) O(h²)', 'C) O(h³)', 'D) O(h⁵)'], 'C'),
               ('How many function evaluations are used per step?', ['A) One', 'B) Two', 'C) Three', 'D) Four'], 'B'),
               ('Why are k₁ and k₂ averaged?',
                ['A) To approximate the average slope over the step',
                 'B) To remove the initial condition',
                 'C) To make the method first-order',
                 'D) To avoid evaluating f'],
                'A'),
               ("Heun's Method is:",
                ['A) Explicit', 'B) Fully implicit', 'C) A quadrature-only method', 'D) A direct matrix solver'],
                'A'),
               ('Compared with Euler at the same small h, Heun is usually:',
                ['A) More accurate', 'B) Less accurate', 'C) Identical', 'D) Unusable'],
                'A'),
               ("Halving h usually reduces Heun's dominant global error by about:",
                ['A) 2', 'B) 4', 'C) 8', 'D) 16'],
                'B'),
               ('A limitation is that Heun:',
                ['A) Is not generally suitable for stiff ODEs',
                 'B) Cannot use nonlinear f',
                 'C) Needs no step size',
                 'D) Is exact for every ODE'],
                'A'),
               ('Heun and explicit midpoint:',
                ['A) Are both second-order RK2 methods but use different stages',
                 'B) Are always the same formula',
                 'C) Are integration rules only',
                 'D) Require four slopes'],
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
