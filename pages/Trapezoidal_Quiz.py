import html
import streamlit as st

from components.navigation import navbar
from utilities.ui import load_css

DATA = {'title': 'Trapezoidal Rule Quiz',
 'label': 'TRAPEZOIDAL RULE QUIZ',
 'lesson': 'Trapezoidal_Rule',
 'solver': 'Trapezoidal_Solver',
 'category': 'Numerical Integration',
 'key': 'trapezoidal_rule',
 'description': 'Test composite weights, equally spaced nodes, polynomial exactness, second-order error, and '
                'practical use.',
 'questions': [('The Trapezoidal Rule approximates:',
                ['A) Definite integrals', 'B) Matrix eigenvalues', 'C) ODE roots only', 'D) Derivatives only'],
                'A'),
               ('It replaces the curve on each subinterval with:',
                ['A) A straight line', 'B) A cubic spline only', 'C) A circle', 'D) A constant zero function'],
                'A'),
               ('The composite step size is:', ['A) h=(b-a)/n', 'B) h=n/(b-a)', 'C) h=a+b', 'D) h=1/n²'], 'A'),
               ('The endpoint weights are:', ['A) 1 and 1', 'B) 2 and 2', 'C) 4 and 4', 'D) 0 and 0'], 'A'),
               ('Each interior value has weight:', ['A) 1', 'B) 2', 'C) 3', 'D) 4'], 'B'),
               ('The composite formula is multiplied by:', ['A) h/2', 'B) h/3', 'C) 2h', 'D) 1/h'], 'A'),
               ('The global error order for smooth functions is:',
                ['A) O(h)', 'B) O(h²)', 'C) O(h³)', 'D) O(h⁴)'],
                'B'),
               ('The rule is exact for polynomials up to degree:', ['A) 0', 'B) 1', 'C) 2', 'D) 3'], 'B'),
               ('For standard composite use, nodes should be:',
                ['A) Equally spaced', 'B) Random', 'C) Complex', 'D) Repeated'],
                'A'),
               ('Increasing n usually:',
                ['A) Improves accuracy for smooth functions',
                 'B) Always worsens accuracy',
                 'C) Has no effect',
                 'D) Removes all round-off'],
                'A'),
               ('Interior values are counted twice because:',
                ['A) They belong to two adjacent trapezoids',
                 'B) The method is fourth-order',
                 'C) Endpoints are ignored',
                 'D) n must be odd'],
                'A'),
               ('The Trapezoidal Rule can work directly with:',
                ['A) Equally spaced tabulated data',
                 'B) No function values',
                 'C) Only symbolic antiderivatives',
                 'D) Only matrices'],
                'A'),
               ("Compared with Simpson's Rule on smooth data, trapezoidal is usually:",
                ['A) Lower order', 'B) Higher order', 'C) Identical order', 'D) Exact for cubics'],
                'A'),
               ('A function discontinuity may require:',
                ['A) Splitting the interval',
                 'B) Ignoring the discontinuity',
                 'C) Setting h=0',
                 'D) Using repeated nodes'],
                'A'),
               ('A key advantage is:',
                ['A) Simplicity',
                 'B) Exactness for all functions',
                 'C) No input data',
                 'D) Unconditional fourth-order accuracy'],
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
