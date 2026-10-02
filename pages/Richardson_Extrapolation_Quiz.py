import html
import streamlit as st

from components.navigation import navbar
from utilities.ui import load_css

DATA = {'title': 'Richardson Extrapolation Quiz',
 'label': 'RICHARDSON EXTRAPOLATION QUIZ',
 'lesson': 'Richardson_Extrapolation',
 'solver': 'Richardson_Extrapolation_Solver',
 'category': 'Numerical Differentiation',
 'key': 'richardson_extrapolation',
 'description': 'Test error expansions, step-size refinement, extrapolation formulas, method order, and cancellation '
                'of leading truncation error.',
 'questions': [('Richardson Extrapolation combines approximations computed with:',
                ['A) Different step sizes',
                 'B) Different functions only',
                 'C) Exact derivatives only',
                 'D) Random intervals'],
                'A'),
               ('Suppose A(h)=L+Chᵖ+O(hᵖ⁺¹). Richardson Extrapolation targets the term:',
                ['A) L', 'B) Chᵖ', 'C) O(1)', 'D) The function value only'],
                'B'),
               ('Using step sizes h and h/2, the standard extrapolated estimate is:',
                ['A) [2ᵖA(h/2)-A(h)]/(2ᵖ-1)', 'B) [A(h/2)+A(h)]/2', 'C) 2ᵖA(h)-A(h/2)', 'D) A(h)/h'],
                'A'),
               ('The value p in the Richardson formula is the:',
                ['A) Leading order of the base-method error',
                 'B) Number of variables',
                 'C) Polynomial degree of f',
                 'D) Iteration count'],
                'A'),
               ('For a first-order forward difference, p equals:', ['A) 1', 'B) 2', 'C) 3', 'D) 4'], 'A'),
               ('For the standard centered first difference, p equals:', ['A) 1', 'B) 2', 'C) 3', 'D) 4'], 'B'),
               ('The principal purpose of Richardson Extrapolation is to:',
                ['A) Cancel the leading truncation-error term',
                 'B) Eliminate all round-off error',
                 'C) Avoid function evaluations',
                 'D) Solve a matrix directly'],
                'A'),
               ('If the base method has error O(hᵖ), one Richardson level often raises the order when the error '
                'expansion is suitable to:',
                ['A) At least the next available power',
                 'B) O(1)',
                 'C) Exactly zero for every problem',
                 'D) A lower order'],
                'A'),
               ('For a centered-difference sequence with even error powers, one extrapolation commonly changes O(h²) '
                'to:',
                ['A) O(h)', 'B) O(h²)', 'C) O(h³)', 'D) O(h⁴)'],
                'D'),
               ('Which assumption is important for Richardson Extrapolation?',
                ['A) The approximations follow a predictable asymptotic error expansion',
                 'B) The exact answer is already known',
                 'C) The function is always linear',
                 'D) The step sizes are unrelated'],
                'A'),
               ('Why can very small h be harmful?',
                ['A) Round-off and cancellation may dominate',
                 'B) The truncation error becomes infinite',
                 'C) The method becomes a root solver',
                 'D) The function becomes discontinuous'],
                'A'),
               ('Richardson Extrapolation is commonly used with:',
                ['A) Finite differences and numerical integration',
                 'B) Sorting algorithms only',
                 'C) Database indexes',
                 'D) Matrix storage formats'],
                'A'),
               ('If A(h)=L+Ch²+O(h⁴), which combination estimates L with O(h⁴) error?',
                ['A) [4A(h/2)-A(h)]/3', 'B) [2A(h/2)-A(h)]', 'C) [A(h)+A(h/2)]/2', 'D) A(h)-A(h/2)'],
                'A'),
               ('Richardson Extrapolation does not automatically remove:',
                ['A) Round-off error and modeling error',
                 'B) The leading truncation term',
                 'C) Dependence on p',
                 'D) The need for multiple approximations'],
                'A'),
               ('A practical way to assess improvement is to:',
                ['A) Compare successive extrapolated estimates',
                 'B) Ignore all step-size changes',
                 'C) Use only one approximation',
                 'D) Replace the function with zero'],
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
