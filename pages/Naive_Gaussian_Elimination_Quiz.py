import html
import streamlit as st

from components.navigation import navbar
from utilities.ui import load_css

DATA = {'title': 'Naive Gaussian Elimination Quiz',
 'label': 'NAIVE GAUSSIAN ELIMINATION QUIZ',
 'lesson': 'Naive_Gaussian_Elimination',
 'solver': 'Naive_Gaussian_Elimination_Solver',
 'category': 'Linear Systems',
 'key': 'naive_gaussian',
 'description': 'Test forward elimination, multipliers, pivots, upper-triangular form, back substitution, and '
                'stability limitations.',
 'questions': [('The primary objective of Naive Gaussian Elimination is to:',
                ['A) Compute eigenvalues.',
                 'B) Transform the coefficient matrix into an upper triangular matrix.',
                 'C) Find the determinant directly.',
                 'D) Perform matrix inversion.'],
                'B'),
               ('Naive Gaussian Elimination consists of two major phases:',
                ['A) Elimination and Back Substitution.',
                 'B) Pivoting and LU Decomposition.',
                 'C) Forward Difference and Backward Difference.',
                 'D) Iteration and Interpolation.'],
                'A'),
               ('During forward elimination, the elements below the pivot are:',
                ['A) Doubled.', 'B) Eliminated to become zero.', 'C) Normalized to one.', 'D) Ignored.'],
                'B'),
               ('The multiplier used in Gaussian Elimination is calculated as:',
                ['A) Pivot / Element', 'B) Element / Pivot', 'C) Sum of Row Elements', 'D) Determinant / Pivot'],
                'B'),
               ('A major weakness of Naive Gaussian Elimination is:',
                ['A) It requires interpolation.',
                 'B) It does not use pivoting.',
                 'C) It requires eigenvalues.',
                 'D) It only works for nonlinear systems.'],
                'B'),
               ('If a pivot element is zero during elimination:',
                ['A) The algorithm continues normally.',
                 'B) Division by zero occurs.',
                 'C) The determinant becomes one.',
                 'D) Back substitution is skipped.'],
                'B'),
               ('Which technique is commonly introduced to overcome the weakness of Naive Gaussian Elimination?',
                ['A) Euler Method', 'B) Partial Pivoting', 'C) Newton-Raphson', "D) Simpson's Rule"],
                'B'),
               ('Back substitution begins after:',
                ['A) The matrix becomes diagonal.',
                 'B) The coefficient matrix becomes upper triangular.',
                 'C) The determinant is computed.',
                 'D) The inverse matrix is found.'],
                'B'),
               ('The computational complexity of Naive Gaussian Elimination is approximately:',
                ['A) O(n)', 'B) O(n²)', 'C) O(n³)', 'D) O(2ⁿ)'],
                'C'),
               ('During forward elimination, row operations are performed to:',
                ['A) Increase the determinant.',
                 'B) Eliminate unknown variables systematically.',
                 'C) Compute eigenvectors.',
                 'D) Find numerical derivatives.'],
                'B'),
               ('The augmented matrix for Ax=b is written as:',
                ['A) [A|b]', 'B) [b|A⁻¹]', 'C) [A+b]', 'D) [A²|b]'],
                'A'),
               ('Elementary row operations preserve:',
                ['A) The solution set of the linear system',
                 'B) Every matrix entry',
                 'C) The original row order',
                 'D) The value of each pivot'],
                'A'),
               ('After forward elimination, the coefficient matrix should be:',
                ['A) Upper triangular',
                 'B) Lower triangular',
                 'C) Identity without further work',
                 'D) Antisymmetric'],
                'A'),
               ('A very small pivot can cause:',
                ['A) Large multipliers and amplified round-off error',
                 'B) Automatic exactness',
                 'C) Fewer operations',
                 'D) A larger determinant by definition'],
                'A'),
               ('After obtaining x, a useful accuracy check is the residual:',
                ['A) r=Ax-b', 'B) r=A+x', 'C) r=det(A)', 'D) r=x/b'],
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
