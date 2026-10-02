import html
import streamlit as st

from components.navigation import navbar
from utilities.ui import load_css

DATA = {'title': 'Secant Method Quiz',
 'label': 'SECANT METHOD QUIZ',
 'lesson': 'Secant_Method',
 'solver': 'Secant_Solver',
 'category': 'Root Finding',
 'key': 'secant',
 'description': 'Test secant-line slopes, derivative-free updates, superlinear convergence, stopping criteria, and '
                'failure conditions.',
 'questions': [('Which statement best describes the Secant Method?',
                ['A) A bracketing method that always halves an interval.',
                 'B) An open method that uses two previous points to approximate the slope.',
                 'C) A direct method for linear systems.',
                 'D) A method that requires the exact second derivative.'],
                'B'),
               ('How many initial approximations are required?', ['A) None', 'B) One', 'C) Two', 'D) Three'], 'C'),
               ('Which derivative approximation leads to the Secant formula?',
                ['A) [f(xₙ)−f(xₙ₋₁)]/[xₙ−xₙ₋₁]', 'B) f(xₙ)/xₙ', 'C) f(xₙ)+f(xₙ₋₁)', 'D) xₙ²'],
                'A'),
               ('What must be checked before dividing in the Secant update?',
                ['A) xₙ must equal zero.',
                 'B) f(xₙ)−f(xₙ₋₁) must not be zero or extremely small.',
                 'C) The two guesses must bracket a root.',
                 'D) The derivative must be positive.'],
                'B'),
               ('What is the typical local convergence order near a simple root?',
                ['A) Linear', 'B) Approximately 1.618 (superlinear)', 'C) Quadratic', 'D) Cubic'],
                'B'),
               ('Which is a valid stopping test?',
                ['A) |xₙ₊₁−xₙ| < tolerance', 'B) xₙ₊₁ > 100', 'C) f(xₙ)=f(xₙ₋₁)', 'D) The interval width doubles'],
                'A'),
               ('Does the Secant Method require an analytical derivative?',
                ['A) Always', 'B) Only for the first iteration', 'C) No', 'D) Only for polynomial equations'],
                'C'),
               ('Why can the Secant Method diverge?',
                ['A) It always preserves a bracket.',
                 'B) Poor starting values or a nearly zero denominator can produce bad steps.',
                 'C) It halves the interval too slowly.',
                 'D) It uses exact arithmetic.'],
                'B'),
               ('Compared with Bisection, the Secant Method is usually:',
                ['A) More reliable and slower',
                 'B) Faster locally but not guaranteed to converge',
                 'C) Identical in every way',
                 'D) Only usable for linear equations'],
                'B'),
               ('After computing xₙ₊₁, how are the points updated?',
                ['A) Keep both old points forever.',
                 'B) Set xₙ₋₁←xₙ and xₙ←xₙ₊₁.',
                 'C) Replace both points with zero.',
                 'D) Swap the function and derivative.'],
                'B'),
               ('Unlike Bisection, the Secant Method:',
                ['A) Does not necessarily maintain a root bracket',
                 'B) Always halves an interval',
                 'C) Requires f′′',
                 'D) Is a direct matrix method'],
                'A'),
               ('After the two starting evaluations, a typical Secant iteration requires:',
                ['A) One new function evaluation',
                 'B) No function evaluations',
                 'C) Four derivative evaluations',
                 'D) A matrix inverse'],
                'A'),
               ('If the two function values are nearly equal, the Secant step may be:',
                ['A) Numerically unstable or excessively large',
                 'B) Exactly zero in all cases',
                 'C) Guaranteed to bracket a root',
                 'D) Quadratically convergent'],
                'A'),
               ('The Secant Method can be interpreted as replacing f′(xₙ) by:',
                ['A) A finite-difference slope through the two latest points',
                 'B) The exact second derivative',
                 'C) An integral average',
                 'D) A Jacobian determinant'],
                'A'),
               ('A residual-based stopping test uses:',
                ['A) |f(xₙ₊₁)|<tolerance', 'B) |f′′(xₙ)|>1', 'C) xₙ=0 only', 'D) The interval width doubling'],
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
