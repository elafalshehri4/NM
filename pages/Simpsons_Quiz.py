import html
import streamlit as st

from components.navigation import navbar
from utilities.ui import load_css

DATA = {'title': 'Simpson’s 1/3 Rule Quiz',
 'label': 'SIMPSON’S 1/3 RULE QUIZ',
 'lesson': 'Simpsons_Rule',
 'solver': 'Simpsons_Solver',
 'category': 'Numerical Integration',
 'key': 'simpsons_rule',
 'description': 'Test parabolic interpolation, composite weights, even-subinterval requirements, exactness, and '
                'fourth-order error.',
 'questions': [("Simpson's 1/3 Rule approximates:",
                ['A) Definite integrals', 'B) Matrix determinants', 'C) Roots only', 'D) ODE slopes only'],
                'A'),
               ('It is based on:',
                ['A) Quadratic interpolation', 'B) Constant interpolation', 'C) Circular arcs', 'D) Random sampling'],
                'A'),
               ('One Simpson panel spans:',
                ['A) One subinterval', 'B) Two subintervals', 'C) Three subintervals', 'D) Four subintervals'],
                'B'),
               ('The number n of composite subintervals must be:', ['A) Even', 'B) Odd', 'C) Prime', 'D) Zero'], 'A'),
               ('The repeating weights are:',
                ['A) 1,4,2,4,...,2,4,1', 'B) 1,2,2,...,1', 'C) 1,1,1,...,1', 'D) 2,3,2,...,3'],
                'A'),
               ('The composite result is multiplied by:', ['A) h/2', 'B) h/3', 'C) h/4', 'D) 3h'], 'B'),
               ('The global error order is:', ['A) O(h)', 'B) O(h²)', 'C) O(h³)', 'D) O(h⁴)'], 'D'),
               ("Simpson's 1/3 Rule is exact for polynomials through degree:", ['A) 1', 'B) 2', 'C) 3', 'D) 4'], 'C'),
               ('Odd-indexed interior values have weight:', ['A) 1', 'B) 2', 'C) 3', 'D) 4'], 'D'),
               ('Even-indexed interior values have weight:', ['A) 1', 'B) 2', 'C) 3', 'D) 4'], 'B'),
               ('Nodes must normally be:',
                ['A) Equally spaced', 'B) Randomly spaced', 'C) Repeated', 'D) Complex'],
                'A'),
               ('Halving h usually reduces the dominant error by about:', ['A) 2', 'B) 4', 'C) 8', 'D) 16'], 'D'),
               ('Compared with trapezoidal on smooth functions, Simpson is usually:',
                ['A) More accurate', 'B) Less accurate', 'C) Always identical', 'D) First-order'],
                'A'),
               ('If n is odd, the standard composite 1/3 rule should:',
                ['A) Reject or adjust the grid', 'B) Continue unchanged', 'C) Set h=0', 'D) Ignore the last point'],
                'A'),
               ('A limitation is:',
                ['A) It requires an even number of equally spaced subintervals',
                 'B) It needs no function values',
                 'C) It is exact for every function',
                 'D) It cannot integrate quadratics'],
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
