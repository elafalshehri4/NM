import html
import streamlit as st

from components.navigation import navbar
from utilities.ui import load_css

DATA = {'title': 'Euler Method Quiz',
 'label': 'EULER METHOD QUIZ',
 'lesson': 'Euler_Method',
 'solver': 'Euler_Solver',
 'category': 'Ordinary Differential Equations',
 'key': 'euler',
 'description': 'Test the Euler update, error order, step size, initial values, geometric meaning, and stability '
                'limitations.',
 'questions': [("Euler's Method is primarily used to approximate:",
                ['A) Definite integrals',
                 'B) Initial-value problems for ODEs',
                 'C) Matrix eigenvalues',
                 'D) Polynomial roots'],
                'B'),
               ('The Euler update is:',
                ['A) yₙ₊₁=yₙ+h f(xₙ,yₙ)', 'B) yₙ₊₁=yₙ-h f(xₙ,yₙ)', 'C) yₙ₊₁=f(xₙ,yₙ)/h', 'D) yₙ₊₁=yₙ+h²'],
                'A'),
               ("Euler's Method uses the slope evaluated at:",
                ['A) The midpoint only', 'B) The current point', 'C) The endpoint only', 'D) Two random points'],
                'B'),
               ('The global order of the standard Euler Method is:',
                ['A) O(h)', 'B) O(h²)', 'C) O(h³)', 'D) O(h⁴)'],
                'A'),
               ('The local truncation error per Euler step is:', ['A) O(1)', 'B) O(h)', 'C) O(h²)', 'D) O(h⁴)'], 'C'),
               ("If h is halved for a smooth problem, Euler's dominant global error is usually:",
                ['A) Roughly halved', 'B) Roughly quartered', 'C) Unchanged', 'D) Exactly zero'],
                'A'),
               ("Which information is required to start Euler's Method?",
                ['A) Only the final value',
                 'B) An initial condition and the ODE',
                 'C) A matrix inverse',
                 'D) A second derivative'],
                'B'),
               ("Euler's Method is classified as:",
                ['A) An explicit one-step method',
                 'B) An implicit multistep method',
                 'C) A quadrature rule',
                 'D) A root-bracketing method'],
                'A'),
               ("For y'=f(x,y), one Euler step from (xₙ,yₙ) first computes:",
                ['A) f(xₙ,yₙ)', "B) f''(xₙ)", 'C) det(A)', 'D) An integral'],
                'A'),
               ('A smaller step size generally:',
                ['A) Improves accuracy but increases work',
                 'B) Decreases accuracy and work',
                 'C) Has no effect',
                 'D) Removes all stability restrictions'],
                'A'),
               ("Euler's Method can perform poorly on:",
                ['A) Stiff ODEs',
                 'B) Constant functions only',
                 'C) Linear equations only',
                 'D) Every smooth problem'],
                'A'),
               ('The x-update is normally:', ['A) xₙ₊₁=xₙ+h', 'B) xₙ₊₁=xₙh', 'C) xₙ₊₁=xₙ/h', 'D) xₙ₊₁=h'], 'A'),
               ("Euler's Method follows which geometric object over one step?",
                ['A) The tangent line at the current point',
                 'B) A parabola through three points',
                 'C) A circle',
                 'D) A cubic spline'],
                'A'),
               ('Compared with midpoint or Heun at the same small h, Euler is usually:',
                ['A) More accurate', 'B) Less accurate', 'C) Identical', 'D) Fourth-order'],
                'B'),
               ("A key advantage of Euler's Method is:",
                ['A) Simplicity',
                 'B) Automatic exactness',
                 'C) Unconditional stability',
                 'D) No function evaluations'],
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
