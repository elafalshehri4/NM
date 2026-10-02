import html
import streamlit as st

from components.navigation import navbar
from utilities.ui import load_css

DATA = {'title': 'Gauss–Jordan Method Quiz',
 'label': 'GAUSS–JORDAN METHOD QUIZ',
 'lesson': 'Gauss_Jordan_Method',
 'solver': 'Gauss_Jordan_Solver',
 'category': 'Linear Systems',
 'key': 'gauss_jordan',
 'description': 'Test reduced row-echelon form, pivot normalization, elimination above and below pivots, and '
                'solution interpretation.',
 'questions': [('What is the primary objective of the Gauss-Jordan Method?',
                ['A) Transform the coefficient matrix into a lower triangular matrix.',
                 'B) Transform the augmented matrix into reduced row echelon form (RREF).',
                 'C) Compute the determinant only.',
                 'D) Approximate nonlinear roots.'],
                'B'),
               ('Unlike Gaussian Elimination, the Gauss-Jordan Method eliminates:',
                ['A) Only the elements below each pivot.',
                 'B) Only the diagonal elements.',
                 'C) Both the elements above and below each pivot.',
                 'D) Only the last row.'],
                'C'),
               ('At the end of the Gauss-Jordan elimination process, the coefficient matrix becomes:',
                ['A) Upper triangular matrix.',
                 'B) Lower triangular matrix.',
                 'C) Identity matrix.',
                 'D) Diagonal matrix with arbitrary values.'],
                'C'),
               ('Which matrix form is obtained after completing the Gauss-Jordan Method?',
                ['A) Upper Row Echelon Form',
                 'B) Reduced Row Echelon Form (RREF)',
                 'C) Hessenberg Form',
                 'D) Triangular Form'],
                'B'),
               ('Which operation is performed before eliminating other entries in a pivot column?',
                ['A) Multiply all rows by zero.',
                 'B) Normalize the pivot so that it becomes 1.',
                 'C) Compute the determinant.',
                 'D) Exchange all rows.'],
                'B'),
               ('Which statement correctly compares Gaussian Elimination and Gauss-Jordan Method?',
                ['A) Both require back substitution.',
                 'B) Gauss-Jordan eliminates the need for back substitution.',
                 'C) Gaussian Elimination always produces the identity matrix.',
                 'D) Gauss-Jordan cannot solve linear systems.'],
                'B'),
               ('The Gauss-Jordan Method can also be used to:',
                ['A) Compute numerical integration.',
                 'B) Find the inverse of a matrix.',
                 'C) Solve differential equations.',
                 'D) Perform polynomial interpolation.'],
                'B'),
               ('Which elementary row operation is NOT allowed in Gauss-Jordan elimination?',
                ['A) Swapping two rows.',
                 'B) Multiplying a row by a nonzero constant.',
                 'C) Adding a multiple of one row to another.',
                 'D) Multiplying a row by zero.'],
                'D'),
               ('If a pivot element is zero and cannot be replaced by row swapping, the matrix is:',
                ['A) Always invertible.',
                 'B) Singular or does not have a unique solution.',
                 'C) Guaranteed to have infinitely many solutions.',
                 'D) Orthogonal.'],
                'B'),
               ('The computational complexity of the Gauss-Jordan Method is approximately:',
                ['A) O(n)', 'B) O(n²)', 'C) O(n³)', 'D) O(log n)'],
                'C'),
               ('Before eliminating a pivot column, the pivot row is usually scaled so that the pivot becomes:',
                ['A) 0', 'B) 1', 'C) -1', 'D) The determinant'],
                'B'),
               ('In the final augmented matrix [I|x], the solution is read from:',
                ['A) The rightmost column', 'B) The first row only', 'C) The diagonal of I', 'D) The row scales'],
                'A'),
               ('Which row operation is valid in Gauss–Jordan elimination?',
                ['A) Add a multiple of one row to another',
                 'B) Square one row without changing the other side',
                 'C) Delete any nonzero row',
                 'D) Change only the coefficient side'],
                'A'),
               ('A row [0 0 … 0 | c] with c≠0 indicates:',
                ['A) A unique solution',
                 'B) An inconsistent system',
                 'C) An identity matrix',
                 'D) A harmless zero row'],
                'B'),
               ('Compared with Gaussian elimination, Gauss–Jordan usually requires:',
                ['A) More arithmetic operations',
                 'B) No pivot operations',
                 'C) No augmented matrix',
                 'D) Fewer eliminations'],
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
