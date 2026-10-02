import html
import streamlit as st

from components.navigation import navbar
from utilities.ui import load_css

DATA = {'title': 'Tridiagonal Systems Quiz',
 'label': 'TRIDIAGONAL SYSTEMS QUIZ',
 'lesson': 'Tridiagonal_Systems',
 'solver': 'Tridiagonal_Systems_Solver',
 'category': 'Linear Systems',
 'key': 'tridiagonal_systems',
 'description': 'Test tridiagonal structure, the Thomas Algorithm, modified coefficients, pivot restrictions, '
                'complexity, and residual checks.',
 'questions': [('A tridiagonal matrix contains nonzero elements on:',
                ['A) Only the main diagonal',
                 'B) The main diagonal and two adjacent diagonals',
                 'C) All diagonals',
                 'D) Only the upper diagonal'],
                'B'),
               ('Which algorithm is commonly used to solve tridiagonal systems?',
                ['A) Newton-Raphson', 'B) Thomas Algorithm', 'C) Euler Method', 'D) Secant Method'],
                'B'),
               ('A tridiagonal system is a special case of:',
                ['A) Nonlinear systems',
                 'B) Differential equations',
                 'C) Linear systems',
                 'D) Optimization problems'],
                'C'),
               ('The Thomas Algorithm is based on:',
                ['A) Gaussian Elimination', 'B) Numerical Integration', 'C) Interpolation', 'D) Differentiation'],
                'A'),
               ('The main advantage of the Thomas Algorithm is:',
                ['A) High memory usage',
                 'B) Simplicity and efficiency',
                 'C) Works only for nonlinear equations',
                 'D) Requires iterations'],
                'B'),
               ('In a tridiagonal matrix, each row contains at most:',
                ['A) 1 nonzero element', 'B) 2 nonzero elements', 'C) 3 nonzero elements', 'D) 4 nonzero elements'],
                'C'),
               ('The Thomas Algorithm consists of:',
                ['A) Forward elimination and back substitution',
                 'B) Interpolation and fitting',
                 'C) Integration and differentiation',
                 'D) Root bracketing'],
                'A'),
               ('The computational cost of Thomas Algorithm is:',
                ['A) O(n³)', 'B) O(n²)', 'C) O(n)', 'D) O(log n)'],
                'C'),
               ('Tridiagonal systems often arise in:',
                ['A) Numerical solutions of differential equations',
                 'B) Sorting algorithms',
                 'C) Database systems',
                 'D) Image compression only'],
                'A'),
               ('The main diagonal of a tridiagonal matrix is:',
                ['A) Always zero',
                 'B) The diagonal from top-left to bottom-right',
                 'C) The upper diagonal',
                 'D) The lower diagonal'],
                'B'),
               ('The three coefficient vectors are commonly called:',
                ['A) Lower, main, and upper diagonals',
                 'B) Eigenvalue, eigenvector, and residual',
                 'C) Predictor, corrector, and slope',
                 'D) Node, weight, and interval'],
                'A'),
               ('The standard Thomas Algorithm performs:',
                ['A) A forward sweep followed by back substitution',
                 'B) Numerical integration followed by differentiation',
                 'C) Newton iteration followed by Bisection',
                 'D) Matrix inversion only'],
                'A'),
               ('The standard Thomas Algorithm without pivoting can fail when:',
                ['A) A modified pivot is zero or nearly zero',
                 'B) The system has three diagonals',
                 'C) n is larger than two',
                 'D) The right-hand side is nonzero'],
                'A'),
               ('The storage requirement for diagonal vectors is:',
                ['A) O(n)', 'B) O(n²)', 'C) O(n³)', 'D) O(2ⁿ)'],
                'A'),
               ('A commonly used sufficient condition for safe Thomas elimination is:',
                ['A) Strict diagonal dominance',
                 'B) A zero main diagonal',
                 'C) Repeated rows',
                 'D) A singular matrix'],
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
