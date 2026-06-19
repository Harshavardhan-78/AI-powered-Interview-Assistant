import re
import json
import tempfile
import datetime

import streamlit as st

from src.rag import create_retriever, get_context
from src.evaluator import generate_questions, evaluate_answer

# ── Page Config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AI Interview Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

    .app-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 60%, #0f3460 100%);
        border-radius: 16px; padding: 2rem 2.5rem; margin-bottom: 1.5rem; color: white;
    }
    .app-header h1 { font-size: 2rem; font-weight: 700; margin: 0 0 0.25rem; color: white; }
    .app-header p  { font-size: 1rem; color: rgba(255,255,255,0.7); margin: 0; }
    .badge {
        display: inline-block; background: rgba(255,255,255,0.15);
        border: 1px solid rgba(255,255,255,0.2); color: white;
        font-size: 0.72rem; font-weight: 600; letter-spacing: 0.08em;
        text-transform: uppercase; padding: 4px 10px; border-radius: 20px; margin-bottom: 0.75rem;
    }

    /* Question card */
    .q-card {
        background: white; border: 1.5px solid #e2e8f0;
        border-left: 5px solid #6c63ff; border-radius: 14px;
        padding: 1.4rem 1.6rem; margin-bottom: 1rem;
        font-size: 1.05rem; line-height: 1.65; color: #1e293b;
        box-shadow: 0 2px 12px rgba(108,99,255,0.06);
    }
    .q-label {
        font-size: 0.7rem; font-weight: 700; letter-spacing: 0.12em;
        text-transform: uppercase; color: #6c63ff; margin-bottom: 6px;
    }

    /* Timer */
    .timer-wrap {
        display: flex; align-items: center; gap: 12px;
        background: #1a1a2e; border-radius: 12px;
        padding: 0.75rem 1.25rem; margin-bottom: 1.25rem;
    }
    .timer-digits {
        font-family: 'Courier New', monospace; font-size: 1.6rem;
        font-weight: 700; color: #a5f3fc; letter-spacing: 3px;
    }
    .timer-digits.warn   { color: #fde68a; }
    .timer-digits.danger { color: #fca5a5; }
    .timer-label { font-size: 0.78rem; color: rgba(255,255,255,0.5); text-transform: uppercase; letter-spacing: 0.08em; }

    /* Progress stepper */
    .stepper {
        display: flex; gap: 6px; align-items: center; margin-bottom: 1.5rem; flex-wrap: wrap;
    }
    .step-dot {
        width: 32px; height: 32px; border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        font-size: 0.78rem; font-weight: 700; border: 2px solid #e2e8f0;
        color: #94a3b8; background: white;
    }
    .step-dot.done    { background: #6c63ff; border-color: #6c63ff; color: white; }
    .step-dot.current { background: white; border-color: #6c63ff; color: #6c63ff; }
    .step-dot.skipped { background: #fee2e2; border-color: #fca5a5; color: #ef4444; }
    .step-line { flex: 1; height: 2px; background: #e2e8f0; min-width: 16px; }
    .step-line.done { background: #6c63ff; }

    /* Evaluation report */
    .report-header {
        background: linear-gradient(135deg, #0f3460 0%, #16213e 100%);
        border-radius: 14px; padding: 1.5rem 2rem; margin-bottom: 1.5rem; color: white;
        text-align: center;
    }
    .report-header .big-score {
        font-size: 3rem; font-weight: 800; color: white; line-height: 1;
    }
    .report-header .sub { font-size: 0.9rem; color: rgba(255,255,255,0.6); margin-top: 4px; }

    .eval-card {
        background: white; border: 1.5px solid #e2e8f0; border-radius: 14px;
        padding: 1.4rem 1.6rem; margin-bottom: 1.25rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
    .eval-card .score-pill {
        display: inline-block; font-size: 0.85rem; font-weight: 700;
        padding: 3px 12px; border-radius: 20px; margin-bottom: 0.6rem;
    }
    .score-pill.low  { background: #fee2e2; color: #b91c1c; }
    .score-pill.mid  { background: #fef9c3; color: #854d0e; }
    .score-pill.high { background: #dcfce7; color: #15803d; }

    /* Metric mini cards */
    .metric-row { display: flex; gap: 12px; margin-bottom: 1.5rem; flex-wrap: wrap; }
    .metric-card {
        flex: 1; min-width: 90px; background: #f8f9ff;
        border: 1px solid #e8eaf6; border-radius: 12px;
        padding: 0.9rem 1rem; text-align: center;
    }
    .metric-card .val { font-size: 1.6rem; font-weight: 700; color: #1a1a2e; line-height: 1; }
    .metric-card .lbl { font-size: 0.72rem; color: #64748b; margin-top: 4px; text-transform: uppercase; letter-spacing: 0.05em; }

    /* Tip card */
    .tip-card {
        background: #f8f9ff; border: 1px solid #e0e7ff; border-radius: 10px;
        padding: 0.8rem 1rem; margin-bottom: 0.6rem;
        font-size: 0.88rem; color: #334155;
        display: flex; align-items: flex-start; gap: 10px;
    }

    [data-testid="stSidebar"] { background: #fafafa; border-right: 1px solid #f1f5f9; }
    .stButton button { border-radius: 10px !important; font-weight: 600 !important; }
    .stProgress > div > div { background: #6c63ff !important; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)

# ── Helpers ────────────────────────────────────────────────────────────────────

def extract_score(text: str):
    for pat in [r"Score[:\s]*(\d+)\s*/\s*10", r"(\d+)\s*/\s*10", r"Score[:\s]*(\d+)"]:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return min(int(m.group(1)), 10)
    return None

def score_class(s):
    if s is None: return "mid"
    return "low" if s <= 4 else ("mid" if s <= 7 else "high")

def avg(lst): return round(sum(lst) / len(lst), 1) if lst else 0

def parse_questions(raw: str) -> list[str]:
    """Return list of individual question strings."""
    lines = [l.strip() for l in raw.strip().splitlines() if l.strip()]
    questions = []
    for line in lines:
        m = re.match(r"^\d+[.)]\s*(.+)", line)
        if m:
            questions.append(m.group(1).strip())
        elif questions:
            questions[-1] += " " + line  # continuation
        else:
            questions.append(line)
    return [q for q in questions if len(q) > 10]

# ── Session State ──────────────────────────────────────────────────────────────

defaults = {
    "phase":       "setup",   # setup | interview | report
    "questions":   [],
    "current_q":   0,
    "answers":     {},        # {idx: {"text": str, "elapsed": int, "skipped": bool}}
    "evaluations": {},        # {idx: {"feedback": str, "score": int}}
    "q_start":     None,
    "history":     [],        # past completed sessions
    "pdf_path":    None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ Settings")

    difficulty = st.selectbox("Difficulty", ["Easy", "Medium", "Hard"], index=1)
    question_count = st.slider("Number of questions", 3, 10, 5)
    time_limit = st.slider("Time per question (seconds)", 30, 300, 120, step=15,
                            help="Timer counts down; question auto-advances when time runs out.")

    st.divider()

    uploaded_file = st.file_uploader("📄 Resume PDF", type=["pdf"])
    if uploaded_file:
        st.success(f"✓ {uploaded_file.name}")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            st.session_state.pdf_path = tmp.name

    st.divider()

    if st.session_state.history:
        st.markdown("### 📁 Past Sessions")
        for i, sess in enumerate(reversed(st.session_state.history[-5:]), 1):
            st.markdown(
                f"**Session {len(st.session_state.history)-i+1}** · "
                f"Avg {sess['avg_score']}/10 · {sess['date']}"
            )

# ── Header ─────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="app-header">
    <div class="badge">AI-Powered</div>
    <h1>🤖 Interview Assistant</h1>
    <p>Answer each question within the time limit · Get a full evaluation report at the end</p>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
#  PHASE: SETUP
# ══════════════════════════════════════════════════════════════════════

if st.session_state.phase == "setup":

    col_info, col_btn = st.columns([3, 1])
    with col_info:
        st.markdown(
            f"**How it works:** Your resume will be analysed to generate **{question_count} {difficulty}** "
            f"questions. You'll have **{time_limit} seconds** per question to answer. "
            f"After all questions, you'll see a full AI evaluation report."
        )
    with col_btn:
        start_btn = st.button(
            "🚀 Start Interview",
            type="primary",
            use_container_width=True,
            disabled=st.session_state.pdf_path is None
        )

    if not st.session_state.pdf_path:
        st.info("👈 Upload your resume PDF in the sidebar to begin.")

    if start_btn and st.session_state.pdf_path:
        with st.spinner("Analysing your resume and generating questions…"):
            retriever = create_retriever(st.session_state.pdf_path)
            context   = get_context(
                retriever,
                "Extract skills, projects, technologies and experience from the resume."
            )
            raw = generate_questions(context, difficulty)
            qs  = parse_questions(raw)
            if not qs:
                st.error("Could not parse questions. Please try again.")
                st.stop()

            st.session_state.questions   = qs[:question_count]
            st.session_state.current_q   = 0
            st.session_state.answers     = {}
            st.session_state.evaluations = {}
            st.session_state.q_start     = datetime.datetime.now()
            st.session_state.phase       = "interview"
            st.rerun()

# ══════════════════════════════════════════════════════════════════════
#  PHASE: INTERVIEW  (one question at a time)
# ══════════════════════════════════════════════════════════════════════

elif st.session_state.phase == "interview":

    questions  = st.session_state.questions
    idx        = st.session_state.current_q
    total      = len(questions)
    elapsed    = int((datetime.datetime.now() - st.session_state.q_start).total_seconds())
    remaining  = max(0, time_limit - elapsed)
    timed_out  = remaining == 0

    # ── Stepper ────────────────────────────────────────────────────────

    stepper_html = '<div class="stepper">'
    for i in range(total):
        if i < idx:
            ans = st.session_state.answers.get(i, {})
            cls = "skipped" if ans.get("skipped") else "done"
            stepper_html += f'<div class="step-dot {cls}">{"✕" if cls=="skipped" else "✓"}</div>'
        elif i == idx:
            stepper_html += f'<div class="step-dot current">{i+1}</div>'
        else:
            stepper_html += f'<div class="step-dot">{i+1}</div>'
        if i < total - 1:
            line_cls = "done" if i < idx else ""
            stepper_html += f'<div class="step-line {line_cls}"></div>'
    stepper_html += '</div>'
    st.markdown(stepper_html, unsafe_allow_html=True)

    # ── Question card ──────────────────────────────────────────────────

    st.markdown(
        f'<div class="q-card"><div class="q-label">Question {idx+1} of {total}</div>'
        f'{questions[idx]}</div>',
        unsafe_allow_html=True
    )

    # ── Timer ──────────────────────────────────────────────────────────

    mins, secs = divmod(remaining, 60)
    t_cls = "danger" if remaining <= 15 else ("warn" if remaining <= 45 else "")
    bar_pct = remaining / time_limit

    st.markdown(
        f'<div class="timer-wrap">'
        f'<div><div class="timer-label">Time remaining</div>'
        f'<div class="timer-digits {t_cls}">{mins:02d}:{secs:02d}</div></div>'
        f'</div>',
        unsafe_allow_html=True
    )
    st.progress(bar_pct)

    # ── Answer input ───────────────────────────────────────────────────

    if timed_out:
        st.warning("⏰ Time's up! Your answer has been saved and we're moving on.")
        placeholder = st.session_state.answers.get(idx, {}).get("text", "")
        answer = st.text_area("Your answer (read-only)", value=placeholder, disabled=True, height=150)
    else:
        answer = st.text_area(
            "Your answer",
            height=160,
            placeholder="Type your answer here… be specific and use examples.",
            key=f"ans_{idx}"
        )

    # ── Navigation buttons ─────────────────────────────────────────────

    col_skip, col_next = st.columns([1, 2])

    def advance(skipped=False):
        st.session_state.answers[idx] = {
            "text":    answer if not skipped else "",
            "elapsed": elapsed,
            "skipped": skipped,
        }
        if idx + 1 < total:
            st.session_state.current_q = idx + 1
            st.session_state.q_start   = datetime.datetime.now()
        else:
            st.session_state.phase = "evaluating"
        st.rerun()

    with col_skip:
        if st.button("⏭ Skip question", use_container_width=True):
            advance(skipped=True)

    with col_next:
        label = "Submit & Finish →" if idx == total - 1 else "Submit & Next →"
        if st.button(label, type="primary", use_container_width=True) or timed_out:
            if timed_out and idx not in st.session_state.answers:
                advance(skipped=not bool(answer.strip()) if not timed_out else True)
            elif not timed_out:
                if answer.strip():
                    advance()
                else:
                    st.warning("Please write an answer or skip the question.")

    # Auto-rerun to keep timer live (every 1 second)
    if not timed_out:
        import time as _time
        _time.sleep(1)
        st.rerun()

# ══════════════════════════════════════════════════════════════════════
#  PHASE: EVALUATING  (runs AI, then flips to report)
# ══════════════════════════════════════════════════════════════════════

elif st.session_state.phase == "evaluating":

    questions = st.session_state.questions
    answers   = st.session_state.answers

    with st.spinner("🧠 Evaluating all your answers… this may take a moment."):
        for i, q in enumerate(questions):
            ans_data = answers.get(i, {})
            if ans_data.get("skipped") or not ans_data.get("text", "").strip():
                st.session_state.evaluations[i] = {
                    "feedback": "*(Question skipped — no answer provided.)*",
                    "score":    None,
                }
            else:
                feedback = evaluate_answer(q, ans_data["text"])
                score    = extract_score(feedback)
                st.session_state.evaluations[i] = {
                    "feedback": feedback,
                    "score":    score,
                }

    # Save to history
    scores = [e["score"] for e in st.session_state.evaluations.values() if e["score"] is not None]
    st.session_state.history.append({
        "date":      datetime.date.today().strftime("%d %b %Y"),
        "avg_score": avg(scores),
        "questions": questions,
        "answers":   answers,
        "evals":     st.session_state.evaluations,
    })

    st.session_state.phase = "report"
    st.rerun()

# ══════════════════════════════════════════════════════════════════════
#  PHASE: REPORT
# ══════════════════════════════════════════════════════════════════════

elif st.session_state.phase == "report":

    questions = st.session_state.questions
    answers   = st.session_state.answers
    evals     = st.session_state.evaluations

    scores    = [e["score"] for e in evals.values() if e["score"] is not None]
    avg_score = avg(scores)
    answered  = sum(1 for a in answers.values() if not a.get("skipped") and a.get("text","").strip())
    skipped   = len(questions) - answered

    # ── Report header ──────────────────────────────────────────────────

    grade = "🏆 Excellent" if avg_score >= 8 else ("👍 Good" if avg_score >= 6 else ("📈 Keep Practising" if avg_score >= 4 else "🧱 Needs Work"))

    st.markdown(
        f'<div class="report-header">'
        f'<div style="font-size:0.8rem;letter-spacing:0.1em;color:rgba(255,255,255,0.5);text-transform:uppercase;margin-bottom:8px">Interview Report</div>'
        f'<div class="big-score">{avg_score}/10</div>'
        f'<div class="sub">{grade} · {answered} answered · {skipped} skipped</div>'
        f'</div>',
        unsafe_allow_html=True
    )

    # ── Summary metrics ────────────────────────────────────────────────

    best  = max(scores, default=0)
    worst = min(scores, default=0)
    total_time = sum(a.get("elapsed", 0) for a in answers.values())
    mins, secs = divmod(total_time, 60)

    st.markdown(
        f'<div class="metric-row">'
        f'<div class="metric-card"><div class="val">{avg_score}</div><div class="lbl">Avg score</div></div>'
        f'<div class="metric-card"><div class="val">{best}/10</div><div class="lbl">Best</div></div>'
        f'<div class="metric-card"><div class="val">{worst}/10</div><div class="lbl">Lowest</div></div>'
        f'<div class="metric-card"><div class="val">{answered}/{len(questions)}</div><div class="lbl">Answered</div></div>'
        f'<div class="metric-card"><div class="val">{mins}m {secs:02d}s</div><div class="lbl">Total time</div></div>'
        f'</div>',
        unsafe_allow_html=True
    )

    # ── Score chart ────────────────────────────────────────────────────

    try:
        import plotly.graph_objects as go
        chart_scores = [evals[i]["score"] if evals[i]["score"] is not None else 0 for i in range(len(questions))]
        colors = ["#ef4444" if s <= 4 else ("#f59e0b" if s <= 7 else "#22c55e") for s in chart_scores]

        fig = go.Figure(go.Bar(
            x=[f"Q{i+1}" for i in range(len(questions))],
            y=chart_scores,
            marker_color=colors,
            text=chart_scores,
            textposition="outside",
        ))
        fig.add_hline(y=7, line_dash="dot", line_color="#6c63ff", annotation_text=" Target (7)")
        fig.update_layout(
            height=240, margin=dict(l=0, r=0, t=10, b=0),
            plot_bgcolor="white", paper_bgcolor="white",
            yaxis=dict(range=[0, 11], gridcolor="#f1f5f9", title="Score"),
            xaxis=dict(gridcolor="#f1f5f9"),
            font=dict(family="Inter", size=12), showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)
    except ImportError:
        pass  # plotly optional

    st.divider()

    # ── Per-question evaluation ────────────────────────────────────────

    st.subheader("Detailed Feedback")

    for i, q in enumerate(questions):
        ev      = evals.get(i, {})
        ans_d   = answers.get(i, {})
        score   = ev.get("score")
        sc      = score_class(score)
        s_lbl   = f"{score}/10" if score is not None else "Skipped"
        ans_txt = ans_d.get("text", "") if not ans_d.get("skipped") else "*(Skipped)*"
        el      = ans_d.get("elapsed", 0)
        em, es  = divmod(el, 60)

        with st.expander(f"Q{i+1} · {q[:80]}{'…' if len(q)>80 else ''}", expanded=(i==0)):
            st.markdown(
                f'<span class="score-pill {sc}">{s_lbl}</span>'
                f'<span style="font-size:0.8rem;color:#94a3b8;margin-left:10px">⏱ {em:02d}:{es:02d}</span>',
                unsafe_allow_html=True
            )
            st.markdown(f"**Question:** {q}")
            st.markdown(f"**Your answer:** {ans_txt}")
            st.markdown("---")
            st.markdown(ev.get("feedback", ""))

    # ── AI coaching insights ───────────────────────────────────────────

    st.divider()
    st.subheader("Coaching Insights")

    tips = []
    if avg_score >= 8:
        tips = [("🏆","Excellent overall! Work on maintaining this under real pressure."),
                ("🎙️","Practice verbal delivery — pacing, pauses, and confidence matter live."),
                ("🔬","Dive deeper: add system-design or architecture angles to strong answers.")]
    elif avg_score >= 6:
        tips = [("📊","Good foundation. Add one concrete metric to each answer."),
                ("🔄","Revisit your lowest-scoring answer and rewrite it before your next session."),
                ("⏱️","Notice your timing — long pauses or rushing both signal nervousness.")]
    else:
        tips = [("🧱","Focus on structure first: Situation → Task → Action → Result (STAR)."),
                ("📝","After reviewing feedback, rewrite each answer before moving on."),
                ("🤔","Quality over quantity — a focused 90-second answer beats a rambling 3-minute one.")]

    if skipped > 0:
        tips.append(("⚠️", f"You skipped {skipped} question(s). Practice those topics before your next session."))

    for icon, text in tips:
        st.markdown(
            f'<div class="tip-card"><span style="font-size:1.1rem">{icon}</span><span>{text}</span></div>',
            unsafe_allow_html=True
        )

    # ── Actions ────────────────────────────────────────────────────────

    st.divider()
    col_retry, col_new = st.columns(2)

    with col_retry:
        if st.button("🔁 Retry Same Questions", use_container_width=True):
            qs = st.session_state.questions
            st.session_state.current_q   = 0
            st.session_state.answers     = {}
            st.session_state.evaluations = {}
            st.session_state.q_start     = datetime.datetime.now()
            st.session_state.phase       = "interview"
            st.rerun()

    with col_new:
        if st.button("🆕 New Interview", type="primary", use_container_width=True):
            st.session_state.phase       = "setup"
            st.session_state.questions   = []
            st.session_state.answers     = {}
            st.session_state.evaluations = {}
            st.session_state.current_q   = 0
            st.rerun()

    # Export
    export_data = {
        "date": datetime.date.today().isoformat(),
        "avg_score": avg_score,
        "questions": questions,
        "answers": {str(k): v for k, v in answers.items()},
        "evaluations": {str(k): v for k, v in evals.items()},
    }
    st.download_button(
        "⬇ Export Report (JSON)",
        data=json.dumps(export_data, indent=2),
        file_name=f"interview_report_{datetime.date.today()}.json",
        mime="application/json",
        use_container_width=True
    )