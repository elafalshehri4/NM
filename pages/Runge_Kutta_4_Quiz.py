import html
import streamlit as st

from components.navigation import navbar
from utilities.ui import load_css

DATA = {'title': 'Fourth-Order Runge–Kutta Quiz',
 'label': 'FOURTH-ORDER RUNGE–KUTTA QUIZ',
 'lesson': 'Runge_Kutta_4',
 'solver': 'Runge_Kutta_4_Solver',
 'category': 'Ordinary Differential Equations',
 'key': 'runge_kutta_4',
 'description': 'Test the four RK4 stages, weights, fourth-order accuracy, function evaluations, and fixed-step '
                'limitations.',
 'questions': [('Classical RK4 is used to solve:',
                ['A) ODE initial-value problems',
                 'B) Only linear systems',
                 'C) Definite integrals only',
                 'D) Eigenvalue problems only'],
                'A'),
               ('How many slope evaluations are used per step?', ['A) One', 'B) Two', 'C) Three', 'D) Four'], 'D'),
               ('The RK4 weights are:', ['A) 1,2,2,1', 'B) 1,1,1,1', 'C) 1,4,1', 'D) 2,1,2,1'], 'A'),
               ('The global order is:', ['A) First', 'B) Second', 'C) Third', 'D) Fourth'], 'D'),
               ('The local truncation error is:', ['A) O(h²)', 'B) O(h³)', 'C) O(h⁴)', 'D) O(h⁵)'], 'D'),
               ('k₁ is evaluated at:',
                ['A) The current point', 'B) The final exact point', 'C) A random point', 'D) x=1 always'],
                'A'),
               ('k₂ and k₃ are evaluated near:',
                ['A) The midpoint', 'B) The left boundary only', 'C) Infinity', 'D) The previous time step'],
                'A'),
               ('k₄ is evaluated near:',
                ['A) The step endpoint', 'B) The initial point only', 'C) The midpoint only', 'D) Zero'],
                'A'),
               ('The RK4 update is:',
                ['A) yₙ₊₁=yₙ+(h/6)(k₁+2k₂+2k₃+k₄)', 'B) yₙ₊₁=yₙ+h k₁', 'C) yₙ₊₁=k₄', 'D) yₙ₊₁=yₙ/h'],
                'A'),
               ('Halving h usually reduces the dominant global error by about:',
                ['A) 2', 'B) 4', 'C) 8', 'D) 16'],
                'D'),
               ('RK4 requires explicit higher derivatives of f:',
                ['A) Always', 'B) Never for the standard formula', 'C) Only for linear ODEs', 'D) Only when h=1'],
                'B'),
               ('A major advantage is:',
                ['A) High accuracy for smooth non-stiff problems',
                 'B) Zero function evaluations',
                 'C) Automatic exactness',
                 'D) Unconditional stability'],
                'A'),
               ('A limitation of fixed-step RK4 is:',
                ['A) No built-in local error estimate',
                 'B) It is only first-order',
                 'C) It cannot handle nonlinear equations',
                 'D) It needs a matrix inverse every step'],
                'A'),
               ('Classical RK4 is generally not ideal for:',
                ['A) Strongly stiff systems',
                 'B) Smooth non-stiff systems',
                 'C) Initial-value problems',
                 'D) Scalar ODEs'],
                'A'),
               ('The two midpoint stages help RK4:',
                ['A) Sample the slope behavior inside the step',
                 'B) Avoid choosing h',
                 'C) Remove y₀',
                 'D) Become implicit'],
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
