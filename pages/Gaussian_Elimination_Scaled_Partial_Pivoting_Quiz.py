import html
import streamlit as st

from components.navigation import navbar
from utilities.ui import load_css

DATA = {'title': 'Scaled Partial Pivoting Quiz',
 'label': 'SCALED PARTIAL PIVOTING QUIZ',
 'lesson': 'Gaussian_Elimination_Scaled_Partial_Pivoting',
 'solver': 'Gaussian_Elimination_Scaled_Partial_Pivoting_Solver',
 'category': 'Linear Systems',
 'key': 'scaled_partial_pivoting',
 'description': 'Test scale factors, pivot ratios, row exchanges, elimination, back substitution, and numerical '
                'stability.',
 'questions': [('What is the primary purpose of Scaled Partial Pivoting?',
                ['A) Reduce memory usage.',
                 'B) Improve numerical stability by selecting better pivots.',
                 'C) Increase the number of iterations.',
                 'D) Eliminate the need for back substitution.'],
                'B'),
               ('In Scaled Partial Pivoting, the pivot row is selected based on:',
                ['A) The largest absolute pivot only.',
                 'B) The smallest row sum.',
                 'C) The largest ratio of the pivot candidate to its scaling factor.',
                 'D) The determinant of the matrix.'],
                'C'),
               ('The scaling factor for each row is defined as:',
                ['A) The diagonal element.',
                 'B) The largest absolute value in that row.',
                 'C) The smallest element in the row.',
                 'D) The sum of all row elements.'],
                'B'),
               ('Scaled Partial Pivoting is mainly designed to reduce:',
                ['A) Round-off errors.',
                 'B) Matrix dimensions.',
                 'C) Computational complexity.',
                 'D) Number of unknowns.'],
                'A'),
               ('Compared with Naive Gaussian Elimination, Scaled Partial Pivoting is:',
                ['A) Less accurate.',
                 'B) More numerically stable.',
                 'C) Slower because it uses iterations.',
                 'D) Applicable only to nonlinear systems.'],
                'B'),
               ('Which ratio is used to determine the pivot row?',
                ['A) |aik| × si', 'B) |aik| / si', 'C) si / |aik|', 'D) aik + si'],
                'B'),
               ('If two rows have identical scaling factors, the pivot is usually chosen based on:',
                ['A) The larger absolute pivot element.',
                 'B) The row number.',
                 'C) The determinant.',
                 'D) The smallest multiplier.'],
                'A'),
               ('Scaled Partial Pivoting is especially useful when:',
                ['A) Matrix elements differ greatly in magnitude.',
                 'B) The matrix is diagonal.',
                 'C) The determinant equals one.',
                 'D) The system has only two variables.'],
                'A'),
               ('The scaling vector is computed:',
                ['A) During every elimination step from scratch.',
                 'B) Once before the elimination process begins.',
                 'C) Only after back substitution.',
                 'D) After each row operation.'],
                'B'),
               ('Which statement is TRUE regarding row exchanges?',
                ['A) They are never required.',
                 'B) They occur whenever another row has a larger scaled pivot ratio.',
                 'C) They are performed after back substitution.',
                 'D) They change the number of equations.'],
                'B'),
               ('At column k, the selected pivot row maximizes:',
                ['A) |aᵢₖ|/sᵢ', 'B) |aᵢₖ|sᵢ', 'C) sᵢ/|aᵢₖ| for all rows', 'D) The row sum without absolute values'],
                'A'),
               ('When two rows are exchanged, their stored scale factors must:',
                ['A) Be exchanged with them',
                 'B) Be discarded',
                 'C) Be squared',
                 'D) Stay attached to the old positions'],
                'A'),
               ('Why can ordinary partial pivoting be misleading when row magnitudes differ greatly?',
                ['A) A large absolute pivot may be small relative to its row scale',
                 'B) It always chooses a zero pivot',
                 'C) It changes the solution',
                 'D) It eliminates back substitution'],
                'A'),
               ('Scaled partial pivoting changes:',
                ['A) The row order, not the mathematical solution',
                 'B) The original equations into nonlinear equations',
                 'C) The number of unknowns',
                 'D) The right-hand side only'],
                'A'),
               ('The asymptotic cost of Gaussian elimination with scaled pivoting remains approximately:',
                ['A) O(n)', 'B) O(n²)', 'C) O(n³)', 'D) O(2ⁿ)'],
                'C')]}

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
