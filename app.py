# app.py — Streamlit front-end for YOUR health_coach_main.py

from pathlib import Path
import sqlite3

import streamlit as st

# 🔗 Import your backend module (the file you pasted)
import health_coach_main as hc

from agents.consulting_agent_with_memory import PatientProfile
from langchain_core.messages import HumanMessage

#Session state
# Initialize session state variables if they don't exist
if "graph_state" not in st.session_state:
    st.session_state["graph_state"] = None
if "step" not in st.session_state:
    st.session_state["step"] = {"type": "", "output": ""}
if "patient_message" not in st.session_state:
    st.session_state["patient_message"] = ""


# ============================================================
# 1) BACKEND INITIALIZATION — USES DB_PATH, GLOBAL_STORE
# ============================================================
@st.cache_resource
def init_backend():
    """
    Init SQLite DB, memory store, and compiled graph ONCE,
    using a thread-safe SQLite connection for Streamlit.
    """
    db_path = Path(hc.DB_PATH)

    # 1) Let backend ensure DB file + tables exist
    tmp_conn = hc.init_db(db_path)
    tmp_conn.close()

    # 2) Re-open DB with check_same_thread=False for Streamlit threads
    if hc.GLOBAL_DB is None:
        hc.GLOBAL_DB = sqlite3.connect(db_path, check_same_thread=False)

    # 3) Load memory store using this connection
    if hc.GLOBAL_STORE is None:
        hc.GLOBAL_STORE = hc.load_memory_store_from_db(hc.GLOBAL_DB)

    #4) Load embedding model for semantic search
    hc.load_embedding_model()
    # 4) LangGraph workflow
    graph = hc.build_planner_graph()
    return graph


graph = init_backend()


# ============================================================
# 2) HELPERS THAT MIRROR CONSOLE LOGIC
# ============================================================
def analyze_state(state: dict):
    """
    Same logic as PlannerAgent._analyze in health_coach_main.py.
    """
    if state.get("final_output"):
        return {"type": "final", "output": state["final_output"]}

    questions = state.get("questions") or []
    answers = state.get("answers") or []

    if questions:
        explanation = answers[0] if answers else ""
        return {
            "type": "followup",
            "question": questions[0],
            "explanation": explanation,
        }

    return {"type": "final", "output": "[No further questions or output.]"}


def load_or_create_patient(
    typed_patient_id: str,
    name: str,
    sex: str,
    age_str: str,
):
    """
    Follows the same idea as patient_intake_node.
    """
    store = hc.GLOBAL_STORE
    typed_patient_id = (typed_patient_id or "").strip()

    # 1) Try existing ID
    if typed_patient_id:
        history = hc.lookup_patient_in_store(typed_patient_id, store)
        if history is not None:
            profile = history.profile
            return (
                profile.model_dump(),
                history.model_dump(),
                profile.id,
                True,   # is_returning
                f"Loaded existing patient: {profile.name} (ID: {profile.id})",
            )
        else:
            info = (
                f"Patient ID '{typed_patient_id}' not found. "
                f"A new profile will be created."
            )
    else:
        info = "New patient profile will be created."

    # 2) Create new profile (mirrors console questions)
    if not name.strip():
        name = "Unknown Patient"

    if not sex:
        sex = "Other"

    try:
        age = int(age_str) if age_str else 0
    except ValueError:
        age = 0

    # Use YOUR ID generator
    new_id = hc.generate_patient_id(name)

    profile_obj = PatientProfile(
        id=new_id,
        name=name,
        sex=sex,
        age=age,
        symptoms=[],
    )

    # Use YOUR saver (writes via GLOBAL_STORE + GLOBAL_DB)
    hc.save_new_patient_to_store(profile_obj, store)

    return (
        profile_obj.model_dump(),
        None,
        new_id,
        False,  # is_returning
        f"Created new patient profile: {name} (ID: {new_id})",
    )


# ============================================================
# 3) STREAMLIT PAGE LAYOUT & STYLES
# ============================================================
st.set_page_config(
    page_title="Personal Health Coach",
    page_icon="💙",
    layout="wide",
)

# ---- Custom CSS: layout + colors + cards ----
st.markdown(
    """
    <style>
    /* Make main area wider and centered */
    .block-container {
        max-width: 1200px;
        padding-top: 0.1rem;
        padding-bottom: 2rem;
        margin: auto;
    }

    .main {
        background: radial-gradient(circle at top left, #fee2e2 0, #fff1f2 35%, #fdf2f8 70%, #ffffff 100%);
        color:#111827;
    }

    /* Buttons */
    .stButton>button {
        border-radius: 999px;
        padding: 0.55rem 1.3rem;
        font-weight: 600;
        border: none;
        box-shadow: 0 6px 18px rgba(248, 113, 167, 0.4);
        background: linear-gradient(135deg, #fb7185, #ec4899);
        color: white;
    }
    .stButton>button:hover {
        filter: brightness(1.03);
        transform: translateY(-1px);
    }

    /* Hero card */
    .hero-card {
        border-radius: 24px;
        padding: 0.5rem 1.6rem 1.6rem 1.6rem;
        background: linear-gradient(135deg, #0ea5e9 0%, #6366f1 40%, #ec4899 100%);
        color: #f9fafb;
        box-shadow: 0 18px 40px rgba(15, 23, 42, 0.45);
        margin-bottom: 1.4rem;
        position: relative;
        overflow: hidden;
    }
    .hero-pill {
        display:inline-block;
        padding:0.18rem 0.8rem;
        border-radius:999px;
        font-size:0.75rem;
        border:1px solid rgba(248,250,252,0.65);
        background:rgba(15,23,42,0.2);
        color:#e0f2fe;
        margin-bottom:0.5rem;
    }
    .hero-title {
        font-size:1.7rem;
        font-weight:700;
        margin:0.1rem 0 0.35rem 0;
        display:flex;
        align-items:center;
        gap:0.4rem;
    }
    .hero-desc {
        font-size:0.9rem;
        max-width: 680px;
        opacity:0.96;
    }
    .hero-heart {
        font-size:1.9rem;
    }

    /* Decorative bubble in hero */
    .hero-card::after {
        content:"";
        position:absolute;
        right:-60px;
        top:-40px;
        width:220px;
        height:220px;
        border-radius:999px;
        background:radial-gradient(circle at 30% 30%, rgba(248,250,252,0.85), transparent 60%);
        opacity:0.35;
    }

    /* Generic pink card */
    .card {
        border-radius: 16px;
        padding: 1.2rem 1.4rem;
        background: #fdf2f8;
        border: 1px solid #f9a8d4;
        box-shadow: 0 10px 30px rgba(244, 114, 182, 0.35);
    }

    .card h2, .card p {
        color:#111827;
    }

    /* Result card with header */
    .result-card {
        margin-top:0.5rem;
        border-radius: 18px;
        padding: 0;
        overflow:hidden;
        border:1px solid #fecaca;
        box-shadow: 0 18px 45px rgba(248, 113, 167, 0.45);
        background: #fff7fb;
    }
    .result-header {
        padding:0.75rem 1.2rem;
        background: linear-gradient(135deg, #f97316, #fb7185, #ec4899);
        color:white;
        display:flex;
        align-items:center;
        justify-content:space-between;
        gap:0.8rem;
    }
    .result-header-left {
        display:flex;
        align-items:center;
        gap:0.6rem;
    }
    .result-badge {
        font-size:0.75rem;
        padding:0.08rem 0.6rem;
        border-radius:999px;
        background:rgba(15,23,42,0.2);
        border:1px solid rgba(248,250,252,0.6);
    }
    .result-body {
        padding:1.1rem 1.3rem 1.15rem 1.3rem;
        max-height: 1024px;
        overflow-y:auto;
        background: linear-gradient(to bottom right, #fff7fb, #fefce8);
    }
    .result-body pre {
        white-space:pre-wrap;
        font-family:system-ui,-apple-system,BlinkMacSystemFont,"SF Pro Text",sans-serif;
        font-size:0.9rem;
        color:#111827;
    }

    /* make Streamlit alerts softer */
    div[data-testid="stAlert"] {
        border-radius: 14px;
        border: 1px solid #bfdbfe;
        background: #eff6ff;
    }

    </style>
    """,
    unsafe_allow_html=True,
)



# ============================================================
# 4) HERO CARD + PATIENT INFO (SIDEBAR)
# ============================================================
# ---- Hero header card ----
st.markdown(
    """
    <div class="hero-card">
      <div class="hero-pill">Multi-Agent • MedlinePlus • Memory</div>
      <div class="hero-title">
        <span class="hero-heart">💙</span>
        <span>Personal Health Coach</span>
      </div>
      <p class="hero-desc">
        This tool uses your multi-agent pipeline (retriever, consulting agent, personal care agent, planner)
        to turn trusted MedlinePlus content into tailored health education. It is <b>not</b> a substitute for
        professional medical advice. In emergencies, call 911 or your local emergency number.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Patient identification")

    patient_id_input = st.text_input(
        "Existing Patient ID (optional)",
        help="If you used the system before, paste your ID. "
             "Otherwise leave this blank.",
    )

    st.markdown("---")
    st.subheader("New / updated profile")

    name_input = st.text_input("Full name")
    sex_input = st.selectbox("Sex", ["", "Male", "Female", "Other"], index=0)
    age_input = st.text_input("Age", placeholder="e.g., 35")

    st.caption(
        "If an existing ID is found, that profile is used. "
        "Otherwise a new profile is created with the info above."
    )


# ============================================================
# 5) MAIN HEALTH CONCERN INPUT
# ============================================================
tab1, tab2 = st.tabs(["Consulting Converstation", "Personal Care Plan"])

with tab1:
    st.subheader("Describe your health concern")

    user_query = st.text_area(
        "What would you like to discuss today?",
        height=140,
        placeholder="Example: I've had a cough and mild fever for 3 days…",
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        start_btn = st.button("🩺 Start / Restart consultation", type="primary")
    with col2:
        clear_btn = st.button("🧹 Clear conversation")



    if clear_btn:
        st.session_state["graph_state"] = None
        st.session_state["step"] = {"type": "", "output": ""}
        st.session_state["patient_message"] = ""
        st.rerun()


    # ============================================================
    # 6) FIRST TURN: RUN FULL PIPELINE (INCLUDING YOUR NODES)
    # ============================================================
    if start_btn:
        if not user_query.strip():
            st.warning("Please describe your health concern first.")
        else:
            (
                profile_dict,
                history_dict,
                final_patient_id,
                is_returning,
                msg,
            ) = load_or_create_patient(
                patient_id_input,
                name_input,
                sex_input,
                age_input,
            )

            st.session_state["patient_message"] = msg

            # We simulate that patient_intake_node has already finished,
            # by giving profile + setting profile_complete=True.
            initial_state = {
                "user_query": user_query.strip(),
                "history": [],
                "final_output": "",
                "patient_profile": profile_dict,
                "patient_history": history_dict,
                "patient_id": final_patient_id,
                "messages": [],
                "turn_count": 0,
                "last_question": "",
                "profile_complete": True,
                "is_returning_patient": is_returning,
                "patient_lookup_attempted": True,
            }

            config = {"configurable": {"thread_id": f"web-{final_patient_id}"}}
            out_state = graph.invoke(initial_state, config)

            st.session_state["graph_state"] = out_state
            st.session_state["step"] = analyze_state(out_state)
            st.rerun()


    # ============================================================
    # 7) RENDER CURRENT STEP (FINAL OR FOLLOW-UP)
    # ============================================================
    step = st.session_state.get("step")
    state = st.session_state.get("graph_state")

    # Show message from load_or_create_patient (loaded or created)
    if st.session_state.get("patient_message"):
        st.info(st.session_state["patient_message"])

    if step is not None and state is not None:
        # st.session_state["agent_step"] = step["type"]

        # ---------- FOLLOW-UP QUESTION ----------
        if step["type"] == "followup":
            st.markdown("### 🧑‍⚕️ Follow-up question from consulting agent")

            if step.get("explanation"):
                with st.expander("Why I'm asking this"):
                    st.write(step["explanation"])

            st.write(f"**Question:** {step['question']}")

            followup_answer = st.text_area(
                "Your answer",
                key="followup_answer",
                placeholder="Type your reply here…",
            )
            submit_followup = st.button("➡️ Send answer")

            if submit_followup:
                if not followup_answer.strip():
                    st.warning("Please type an answer before submitting.")
                else:
                    # Mirror PlannerAgent.continue_with_answer()
                    msgs = state.get("messages", [])
                    msgs.append(HumanMessage(content=followup_answer.strip()))
                    state["messages"] = msgs
                    state["user_query"] = followup_answer.strip()

                    config = {
                        "configurable": {
                            "thread_id": f"web-{state.get('patient_id', 'noid')}"
                        }
                    }
                    new_state = graph.invoke(state, config)
                    st.session_state["graph_state"] = new_state
                    st.session_state["step"] = analyze_state(new_state)
                    st.rerun()
        elif st.session_state.get("step")["type"]  == "final":
            st.markdown("### ✅ Consultation complete, personal health plan ready!")
            
# st.caption(
#     "Front-end strictly uses your health_coach_main.py pipeline "
#     "(semantic_search, consulting_agent, personal_care_agent, DB + memory_store)."
# )
    # ---------- FINAL PLAN ----------
with tab2:
    st.markdown("### 💊🌡️🩺 Personalized health guidance")
    if st.session_state.get("graph_state") is not None and st.session_state.get("step")["type"]  == "final":
        
        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-header">
                <div class="result-header-left">
                    <span>🩺 Personalized Plan</span>
                    <span class="result-badge">Auto-generated from your multi-agent pipeline</span>
                </div>
                <span>MedlinePlus • Evidence-informed</span>
                </div>
                <div class="result-body">
                <pre>{step['output']}</pre>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.info(
            "Educational use only — not a medical diagnosis. "
            "If symptoms are severe or worsening, please seek urgent care."
        )

        # Save memory store to your DB (same as console main)
        if hc.GLOBAL_STORE is not None and hc.GLOBAL_DB is not None:
            hc.save_memory_store_to_db(hc.GLOBAL_STORE, hc.GLOBAL_DB)