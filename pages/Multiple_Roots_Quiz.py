import html
import streamlit as st

from components.navigation import navbar
from utilities.ui import load_css

DATA = {'title': 'Multiple Roots Method Quiz',
 'label': 'MULTIPLE ROOTS METHOD QUIZ',
 'lesson': 'Multiple_Roots_Method',
 'solver': 'Multiple_Roots_Solver',
 'category': 'Root Finding',
 'key': 'multiple_roots',
 'description': 'Test multiplicity, repeated-root behavior, modified Newton formulas, derivative requirements, and '
                'convergence.',
 'questions': [('What is a multiple root?',
                ['A) A root that appears only once',
                 'B) A root with multiplicity greater than one',
                 'C) A complex root',
                 'D) A negative root'],
                'B'),
               ('Which equation has a multiple root?',
                ['A) x² - 4 = 0', 'B) (x - 2)² = 0', 'C) x + 5 = 0', 'D) x³ + 1 = 0'],
                'B'),
               ('What is the multiplicity of the root x = 3 in (x - 3)³ = 0?', ['A) 1', 'B) 2', 'C) 3', 'D) 4'], 'C'),
               ('At a multiple root r, which condition is true?',
                ['A) f(r) = 0 only', 'B) f(r) ≠ 0', "C) f(r) = 0 and f'(r) = 0", "D) f'(r) ≠ 0"],
                'C'),
               ("Why does Newton's Method converge slowly near a multiple root?",
                ['A) Because the derivative becomes small near the root',
                 'B) Because the function is discontinuous',
                 'C) Because the root does not exist',
                 'D) Because the interval is too large'],
                'A'),
               ('The Multiple Roots Method modifies which method?',
                ['A) Bisection Method', 'B) Newton-Raphson Method', 'C) Euler Method', 'D) Simpson Method'],
                'B'),
               ('What additional information is needed in the Multiple Roots Method?',
                ['A) Integral of the function',
                 'B) Root interval',
                 'C) Multiplicity of the root',
                 'D) Matrix determinant'],
                'C'),
               ('If a root has multiplicity m, the modified Newton formula contains:',
                ['A) Division by m', 'B) Multiplication by m', 'C) Squaring m', 'D) Ignoring m'],
                'B'),
               ('A root of multiplicity 1 is called:',
                ['A) Double root', 'B) Triple root', 'C) Simple root', 'D) Repeated root'],
                'C'),
               ('A double root has multiplicity:', ['A) 1', 'B) 2', 'C) 3', 'D) 4'], 'B'),
               ('If the multiplicity m is known, a modified Newton update is:',
                ['A) xₙ₊₁=xₙ-m f(xₙ)/f′(xₙ)', 'B) xₙ₊₁=xₙ+mf′(xₙ)', 'C) xₙ₊₁=(a+b)/2', 'D) xₙ₊₁=xₙ²'],
                'A'),
               ('A multiplicity-independent modified Newton formula is:',
                ['A) xₙ₊₁=xₙ-[f f′]/[(f′)²-f f′′]', 'B) xₙ₊₁=xₙ-f′/f', 'C) xₙ₊₁=xₙ+f/f′', 'D) xₙ₊₁=xₙ-f′′'],
                'A'),
               ('The multiplicity-independent formula requires evaluating:',
                ['A) f, f′, and f′′', 'B) Only f', 'C) Only f′′', 'D) An integral'],
                'A'),
               ('A potential failure occurs when:',
                ['A) (f′)²-f f′′ is zero or extremely small',
                 'B) The root is real',
                 'C) The initial guess is close',
                 'D) f is continuous'],
                'A'),
               ('For a known multiplicity and a sufficiently close starting value, the modified Newton method can '
                'often restore:',
                ['A) Quadratic convergence',
                 'B) No convergence',
                 'C) Cubic convergence for every function',
                 'D) Bisection convergence'],
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
