import html
import streamlit as st

from components.navigation import navbar
from utilities.ui import load_css

DATA = {'title': 'Bisection Method Quiz',
 'label': 'BISECTION METHOD QUIZ',
 'lesson': 'Bisection_Method',
 'solver': 'Bisection_Solver',
 'category': 'Root Finding',
 'key': 'bisection',
 'description': 'Test root bracketing, midpoint updates, convergence, error bounds, stopping criteria, and '
                'limitations.',
 'questions': [('For a continuous function on [a,b], which condition brackets at least one root in the interval?',
                ['A) f(a)=f(b)', 'B) f(a)f(b)<0', 'C) f(a)f(b)>0', 'D) a=b'],
                'B'),
               ('The midpoint used in each Bisection iteration is:',
                ['A) c=a+b', 'B) c=(a+b)/2', 'C) c=(b-a)/2', 'D) c=ab'],
                'B'),
               ('The Bisection Method is justified by the:',
                ['A) Intermediate Value Theorem',
                 'B) Mean Value Theorem only',
                 'C) Fundamental Theorem of Calculus',
                 'D) Binomial Theorem'],
                'A'),
               ('After computing c, which interval is retained?',
                ['A) The half on which the endpoint function values have opposite signs',
                 'B) Always [a,c]',
                 'C) Always [c,b]',
                 'D) The half with the larger endpoint value'],
                'A'),
               ('If f(c)=0 exactly, the algorithm should:',
                ['A) Continue for the maximum iterations',
                 'B) Stop because c is a root',
                 'C) Double the interval',
                 'D) Replace c with a'],
                'B'),
               ('After n bisections, the bracket width is:',
                ['A) (b-a)/n', 'B) (b-a)/2ⁿ', 'C) 2ⁿ(b-a)', 'D) (b-a)²'],
                'B'),
               ('A standard upper bound for the midpoint error after n bisections is:',
                ['A) (b-a)/2ⁿ⁺¹', 'B) n(b-a)', 'C) 2ⁿ/(b-a)', 'D) h²'],
                'A'),
               ('The convergence of the Bisection Method is:',
                ['A) Linear', 'B) Quadratic', 'C) Cubic', 'D) Superlinear of order 1.618'],
                'A'),
               ('Which quantity is not required by the Bisection Method?',
                ['A) Function values',
                 'B) A bracketing interval',
                 'C) An analytical derivative',
                 'D) A stopping tolerance'],
                'C'),
               ('Which is a valid stopping condition?',
                ['A) (b-a)/2 < tolerance',
                 'B) f(a)=f(b)',
                 'C) The interval width increases',
                 'D) a and b have the same sign forever'],
                'A'),
               ('A key advantage of Bisection is that it:',
                ['A) Is guaranteed to preserve a bracket under its assumptions',
                 'B) Always converges quadratically',
                 'C) Requires only one initial number',
                 'D) Finds every root in one step'],
                'A'),
               ('A key disadvantage of Bisection is that it:',
                ['A) Converges relatively slowly',
                 'B) Requires a second derivative',
                 'C) Cannot use tolerances',
                 'D) Never evaluates the function'],
                'A'),
               ('Why can Bisection miss an even-multiplicity root?',
                ['A) The function may touch the axis without changing sign',
                 'B) The function becomes discontinuous at every repeated root',
                 'C) The derivative is always infinite',
                 'D) The midpoint formula fails'],
                'A'),
               ('If the initial width is 8, what is the width after three bisections?',
                ['A) 4', 'B) 2', 'C) 1', 'D) 0.5'],
                'C'),
               ('What should be checked before starting the standard Bisection algorithm?',
                ['A) Continuity on the interval and a sign change at the endpoints',
                 'B) That f is a polynomial',
                 'C) That f′′ is positive',
                 'D) That a and b are both roots'],
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
