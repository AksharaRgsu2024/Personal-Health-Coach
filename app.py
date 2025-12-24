# app.py — Updated Streamlit front-end for health_coach_main.py with MemoryManager

from pathlib import Path
import sqlite3

import streamlit as st

import health_coach_main as hc

from agents.consulting_agent_with_memory import PatientProfile
from langchain_core.messages import HumanMessage
import html
import markdown
import re

# Session state
# Initialize session state variables if they don't exist
if "graph_state" not in st.session_state:
    st.session_state["graph_state"] = None
if "step" not in st.session_state:
    st.session_state["step"] = {"type": "", "output": ""}
if "patient_message" not in st.session_state:
    st.session_state["patient_message"] = ""
if "agent_status" not in st.session_state:
    st.session_state["agent_status"] = []


# ============================================================
# 1) BACKEND INITIALIZATION — USES MemoryManager
# ============================================================
@st.cache_resource
def init_backend():
    """
    Initialize MemoryManager and compiled graph ONCE,
    using a thread-safe approach for Streamlit.
    """
    hc.init_memory_backend()
    # Build LangGraph workflow
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
    Follows the same idea as patient_intake_node but uses MemoryManager.
    """
    memory_mgr = hc.MEMORY_MANAGER
    typed_patient_id = (typed_patient_id or "").strip()

    # 1) Try existing ID
    if typed_patient_id:
        history = memory_mgr.lookup_patient(typed_patient_id)
        if history is not None:
            profile = history.profile
            # Track this returning patient
            memory_mgr.track_patient_id(profile.id)
            
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

    # 2) Create new profile
    if not name.strip():
        name = "Unknown Patient"

    if not sex:
        sex = "Other"

    try:
        age = int(age_str) if age_str else 0
    except ValueError:
        age = 0

    # Use MemoryManager's ID generator
    new_id = hc.MemoryManager.generate_patient_id(name)

    profile_obj = PatientProfile(
        id=new_id,
        name=name,
        sex=sex,
        age=age,
        symptoms=[],
    )

    # Use MemoryManager's save method
    success = memory_mgr.save_new_patient_profile(profile_obj, persist_immediately=True)
    
    if not success:
        st.error("Failed to save patient profile. Please try again.")
        return None, None, None, False, "Error creating profile"

    return (
        profile_obj.model_dump(),
        None,
        new_id,
        False,  # is_returning
        f"Created new patient profile: {name} (ID: {new_id})",
    )


def clean_llm_output(text: str) -> str:
    """Clean LLM output for better display."""
    text = text.strip("\n")
    text = "\n".join(line.lstrip() for line in text.splitlines())
    text = re.sub(r'[\u00A0\u200B\u202F]', ' ', text)
    # Normalize multiple spaces to single space (but preserve intentional line breaks)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(
        r'<strong>PATIENT CONTEXT</strong>.*?(?=\n|$)', 
        r'<strong>PATIENT CONTEXT</strong>\n\n', 
        text, 
        flags=re.IGNORECASE | re.DOTALL
    )
    # remove stray closing divs and body/html tags
    text = re.sub(r"</?(div|body|html)[^>]*>", "", text, flags=re.IGNORECASE)
    return text


def fix_markdown_tables(text):
    """Fix malformed markdown tables by splitting on || and rebuilding with proper line breaks."""
    lines = text.split('\n')
    fixed_lines = []
    
    for line in lines:
        # Check if this line contains a table (has multiple |)
        if line.count('|') >= 2:
            # Check if it has || which indicates concatenated rows
            if '||' in line:
                # Split by || to separate rows
                parts = line.split('||')
                for part in parts:
                    if part.strip():
                        # Ensure it starts and ends with |
                        part = part.strip()
                        if not part.startswith('|'):
                            part = '|' + part
                        if not part.endswith('|'):
                            part = part + '|'
                        fixed_lines.append(part)
            else:
                # Already properly formatted
                fixed_lines.append(line)
        else:
            # Not a table line
            fixed_lines.append(line)
    
    return '\n'.join(fixed_lines)


def render_markdown_with_tables(text):
    """Convert markdown text to HTML, handling tables properly."""
    # Clean the text first
    cleaned_text = clean_llm_output(text)
    
    # Fix table formatting issues
    fixed_text = fix_markdown_tables(cleaned_text)
    
    # Convert markdown to HTML
    html_output = markdown.markdown(
        fixed_text,
        extensions=['tables', 'fenced_code', 'nl2br']
    )
    
    return html_output


# ============================================================
# 3) STREAMLIT PAGE LAYOUT & STYLES
# ============================================================
st.set_page_config(
    page_title="Personal Health Coach",
    page_icon="💙",
    layout="wide",
)

# ---- Custom CSS: layout + colors + cards + markdown styling ----
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
        padding:1.5rem 1.8rem;
        max-height: 1024px;
        overflow-y:auto;
        background: linear-gradient(to bottom right, #fff7fb, #fefce8);
    }

    /* Markdown content styling */
    .result-body h1, .result-body h2, .result-body h3 {
        color: #1f2937;
        margin-top: 1.5rem;
        margin-bottom: 0.75rem;
        font-weight: 600;
    }
    
    .result-body h1 { font-size: 1.5rem; }
    .result-body h2 { font-size: 1.3rem; }
    .result-body h3 { font-size: 1.1rem; }
    
    .result-body p {
        color: #374151;
        line-height: 1.7;
        margin-bottom: 1rem;
    }
    
    .result-body ul, .result-body ol {
        margin-left: 1.5rem;
        margin-bottom: 1rem;
        color: #374151;
    }
    
    .result-body li {
        margin-bottom: 0.5rem;
        line-height: 1.6;
    }
    
    .result-body strong {
        color: #1f2937;
        font-weight: 600;
    }
    
    .result-body code {
        background: #f3f4f6;
        padding: 0.2rem 0.4rem;
        border-radius: 4px;
        font-size: 0.9em;
        color: #ef4444;
    }
    
    /* Table styling */
    .result-body table {
        width: 100%;
        border-collapse: collapse;
        margin: 1.5rem 0;
        background: white;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        border-radius: 8px;
        overflow: hidden;
    }
    
    .result-body table thead {
        background: linear-gradient(135deg, #ec4899, #f472b6);
        color: white;
    }
    
    .result-body table th {
        padding: 0.75rem 1rem;
        text-align: left;
        font-weight: 600;
        font-size: 0.9rem;
        border-bottom: 2px solid #f9a8d4;
    }
    
    .result-body table td {
        padding: 0.75rem 1rem;
        border-bottom: 1px solid #fce7f3;
        color: #374151;
        line-height: 1.5;
    }
    
    .result-body table tbody tr:hover {
        background: #fdf2f8;
    }
    
    .result-body table tbody tr:last-child td {
        border-bottom: none;
    }
    
    /* Links */
    .result-body a {
        color: #ec4899;
        text-decoration: none;
        font-weight: 500;
    }
    
    .result-body a:hover {
        color: #db2777;
        text-decoration: underline;
    }
    
    /* Blockquotes */
    .result-body blockquote {
        border-left: 4px solid #ec4899;
        padding-left: 1rem;
        margin: 1rem 0;
        color: #6b7280;
        font-style: italic;
    }
    
    /* Horizontal rules */
    .result-body hr {
        border: none;
        border-top: 2px solid #f9a8d4;
        margin: 2rem 0;
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
        This tool uses a multi-agent pipeline (retriever, consulting agent, personal care agent, planner)
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
tab1, tab2 = st.tabs(["Consulting Conversation", "Personal Care Plan"])

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
        st.session_state["agent_status"] = []
        st.rerun()

    # ============================================================
    # 6) FIRST TURN: RUN FULL PIPELINE WITH AGENT STATUS TRACKING
    # ============================================================
    if start_btn:
        if not user_query.strip():
            st.warning("Please describe your health concern first.")
        else:
            # Create a status container
            status_container = st.status("Processing your request...", expanded=True)
            
            with status_container:
                st.write("🔍 Loading patient information...")
                
                result = load_or_create_patient(
                    patient_id_input,
                    name_input,
                    sex_input,
                    age_input,
                )
                
                # Check if profile creation was successful
                if result[0] is None:
                    st.error("Failed to load or create patient profile. Please try again.")
                    st.stop()
                
                (
                    profile_dict,
                    history_dict,
                    final_patient_id,
                    is_returning,
                    msg,
                ) = result

                st.session_state["patient_message"] = msg
                st.write(f"✅ {msg}")

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
                
                # Track agent execution
                agent_statuses = []
                final_state = None
                
                # Stream through the graph to show agent transitions
                for event in graph.stream(initial_state, config):
                    for node_name, node_state in event.items():
                        final_state = node_state  # Keep updating to get final state
                        
                        if node_name == "patient_intake":
                            st.write("👤 **Patient Intake**: Profile verified")
                            agent_statuses.append("Patient Intake completed")
                            
                        elif node_name == "retriever_agent":
                            st.write("🔍 **Retriever Agent**: Searching MedlinePlus knowledge base...")
                            agent_statuses.append("Retriever Agent completed")
                            
                        elif node_name == "consulting_agent":
                            st.write("🧑‍⚕️ **Consulting Agent**: Analyzing your health concern...")
                            agent_statuses.append("Consulting Agent analyzing")
                            
                        elif node_name == "followup":
                            st.write("🤔 **Decision Point**: Evaluating if more information is needed...")
                            agent_statuses.append("Evaluating follow-up needs")
                            
                        
 
                        elif node_name == "personal_care_agent":
                            agent_statuses.append("Personal Care Agent working")
                            
                        elif node_name == "planner_summary":
                            st.write("📋 **Planner**: Finalizing recommendations...")
                            agent_statuses.append("Planner completing")

                    
                
                # Use the final state captured during streaming
                out_state = final_state if final_state else initial_state
                
                st.session_state["graph_state"] = out_state
                st.session_state["step"] = analyze_state(out_state)
                st.session_state["agent_status"] = agent_statuses
                
                # Determine final status
                step = analyze_state(out_state)
                if step["type"] == "followup":
                    st.write("❓ Follow-up questions needed to provide better guidance")
                    status_container.update(label="Consultation in progress - awaiting your response", state="running")
                else:
                    st.write("✅ **All agents completed** - Your personalized plan is ready!")
                    status_container.update(label="Processing complete!", state="complete")
            
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
                    # Create status container for follow-up processing
                    status_container = st.status("Processing your answer...", expanded=True)
                    
                    with status_container:
                        st.write("📝 Recording your response...")
                        
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
                        
                        # Stream through the graph to show agent transitions
                        agent_statuses = []
                        final_state = None
                        
                        for event in graph.stream(state, config):
                            for node_name, node_state in event.items():
                                final_state = node_state  # Keep updating to get final state
                                    
                                if node_name == "consulting_agent":
                                    st.write("🧑‍⚕️ **Consulting Agent**: Re-analyzing with new information...")
                                    agent_statuses.append("Consulting Agent re-analyzing")
                                    
                                elif node_name == "followup":
                                    st.write("🤔 **Decision Point**: Evaluating if more information is needed...")
                                    agent_statuses.append("Evaluating follow-up needs")
                                    # Check if we're proceeding to personal care (no more follow-ups)
                                    if not node_state.get("needs_more"):
                                        st.write("✅ Sufficient information gathered")

                                elif node_name == "personal_care_agent":
                                    st.write("💊 **Personal Care Agent**: Generating personalized health plan...")
                                    agent_statuses.append("Personal Care Agent working")
                                    
                                elif node_name == "planner_summary":
                                    st.write("📋 **Planner**: Finalizing recommendations...")
                                    agent_statuses.append("Planner completing")

                            
                        # Use the final state captured during streaming
                        new_state = final_state if final_state else state
                        
                        st.session_state["graph_state"] = new_state
                        st.session_state["step"] = analyze_state(new_state)
                        st.session_state["agent_status"] = agent_statuses
                        
                        # Determine final status
                        step = analyze_state(new_state)
                        if step["type"] == "followup":
                            st.write("❓ Additional follow-up questions needed")
                            status_container.update(label="Processing complete - awaiting your response", state="running")
                        else:
                            st.write("✅ **All agents completed** - Your personalized plan is ready!")
                            status_container.update(label="Processing complete!", state="complete")
                    
                    st.rerun()
        
        # ---------- FINAL PLAN ----------
        elif step["type"] == "final" and step.get("output"):
            # Check if output is not just a placeholder message
            if step["output"] and step["output"] != "[No further questions or output.]":
                st.markdown("### ✅ Consultation complete!")
                st.success("Your personalized health plan is ready. Switch to the 'Personal Care Plan' tab to view it.")

# ============================================================
# 8) PERSONAL CARE PLAN TAB WITH PROPER MARKDOWN RENDERING
# ============================================================
with tab2:
    step = st.session_state.get("step")
    
    if step and step["type"] == "final" and step.get("output"):
        if step["output"] and step["output"] != "[No further questions or output.]":
            st.markdown("### 💊 Personalized Health Guidance")

            # 1. Clean the LLM output, but keep it as Markdown
            cleaned_output = clean_llm_output(step["output"])

            full_card_markdown = f"""
            <div class="result-card" style="max-width: 100%; overflow-x: auto;">
            <div class="result-header">
                <div class="result-header-left">
                <span>🩺 Personalized Plan</span>
                <span class="result-badge">Generated by Multi-agent Health Coach</span>
                </div>
                <span>MedlinePlus • Evidence-informed</span>
            </div>
            <div class="result-body" style="padding: 20px;">
                <div class="output-content" style="line-height: 1.6; font-size: 14px;">
            {cleaned_output}
                </div>
            </div>
            </div>
            """

            # 3. Let Streamlit render the Markdown (including tables) inside the card
            st.markdown(full_card_markdown, unsafe_allow_html=True)

            st.info(
                "Educational use only — not a medical diagnosis. "
                "If symptoms are severe or worsening, please seek urgent care."
            )
            
            # Save memory on completion
            if hc.MEMORY_MANAGER:
                try:
                    hc.MEMORY_MANAGER.save_to_db()
                except Exception as e:
                    st.error(f"Error saving data: {e}")