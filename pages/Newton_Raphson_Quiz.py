import html
import streamlit as st

from components.navigation import navbar
from utilities.ui import load_css

DATA = {'title': 'Newton–Raphson Method Quiz',
 'label': 'NEWTON–RAPHSON METHOD QUIZ',
 'lesson': 'Newton_Raphson_Method',
 'solver': 'Newton_Raphson_Solver',
 'category': 'Root Finding',
 'key': 'newton_raphson',
 'description': 'Test tangent-line reasoning, the Newton update, derivative checks, convergence, stopping criteria, '
                'and failure cases.',
 'questions': [('Which assumption is essential for the Newton-Raphson Method to achieve quadratic convergence?',
                ['A) The function must be periodic.',
                 "B) The root must be simple and f'(x) ≠ 0 near the root.",
                 'C) The function must be linear.',
                 'D) The derivative must be constant.'],
                'B'),
               ('The Newton-Raphson iteration formula is:',
                ['A) xₙ₊₁ = (a+b)/2', "B) xₙ₊₁ = xₙ + f(xₙ)/f'(xₙ)", "C) xₙ₊₁ = xₙ - f(xₙ)/f'(xₙ)", 'D) xₙ₊₁ = xₙ²'],
                'C'),
               ('Newton-Raphson Method is derived from:',
                ['A) Fourier Series', 'B) Taylor Series Expansion', 'C) Laplace Transform', 'D) Euler Formula'],
                'B'),
               ('Which situation may cause Newton-Raphson Method to fail?',
                ["A) f'(x)=0 at an iteration",
                 'B) The root is simple',
                 'C) The function is continuous',
                 'D) The initial guess is close to the root'],
                'A'),
               ('The order of convergence of Newton-Raphson Method is:',
                ['A) Linear', 'B) Quadratic', 'C) Cubic', 'D) Logarithmic'],
                'B'),
               ('Newton-Raphson Method requires:',
                ['A) Numerical integration',
                 'B) The derivative of the function',
                 'C) Matrix inversion',
                 'D) Partial fractions'],
                'B'),
               ('Which statement is TRUE?',
                ['A) Newton-Raphson always converges.',
                 'B) Convergence depends on the initial guess.',
                 'C) It never diverges.',
                 'D) It requires two initial guesses.'],
                'B'),
               ('If the initial approximation is far from the root:',
                ['A) Convergence is always guaranteed.',
                 'B) Divergence may occur.',
                 'C) The derivative is ignored.',
                 'D) Quadratic convergence is guaranteed.'],
                'B'),
               ('Multiple roots generally make Newton-Raphson:',
                ['A) Converge faster.',
                 'B) Lose quadratic convergence.',
                 'C) Impossible to apply.',
                 'D) Independent of the derivative.'],
                'B'),
               ('Which method usually converges faster near the root?',
                ['A) Bisection Method', 'B) Newton-Raphson Method', 'C) Regula Falsi', 'D) Incremental Search'],
                'B'),
               ('Geometrically, the next Newton iterate is where the tangent at xₙ:',
                ['A) Intersects the x-axis',
                 'B) Intersects the y-axis',
                 'C) Reaches its maximum',
                 'D) Becomes horizontal'],
                'A'),
               ('A common stopping test is:',
                ['A) |xₙ₊₁-xₙ|<tolerance or |f(xₙ₊₁)|<tolerance',
                 'B) xₙ₊₁>xₙ always',
                 'C) f′(xₙ)=0',
                 'D) The interval doubles'],
                'A'),
               ('Quadratic convergence is normally expected when:',
                ['A) The initial guess is sufficiently close to a simple root and f′ does not vanish there',
                 'B) The function is discontinuous',
                 'C) The root has any multiplicity with no modification',
                 'D) No derivative is available'],
                'A'),
               ('Poor initial guesses can cause Newton’s Method to:',
                ['A) Diverge or converge to a different root',
                 'B) Preserve a bracket automatically',
                 'C) Become Bisection',
                 'D) Require no iterations'],
                'A'),
               ('For a multiple root, unmodified Newton convergence is typically reduced to:',
                ['A) Linear', 'B) Quadratic', 'C) Cubic', 'D) Finite-step exact convergence'],
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
