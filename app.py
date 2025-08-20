import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_DIR = r"E:\CSI_CB\final_merged_model"

st.set_page_config(
    page_title="AI Loan Chatbot",
    page_icon="💳",
    layout="wide",
)

# Cyberpunk + Banking theme
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Share+Tech+Mono&display=swap');

    html, body, [class^="block-container"], .main {
        background: radial-gradient(1200px 600px at 20% 0%, rgba(0,255,255,0.06), transparent 40%),
                    radial-gradient(1000px 600px at 100% 20%, rgba(255,0,200,0.06), transparent 40%),
                    linear-gradient(135deg, #0b0f17 0%, #0d0f1a 100%);
        color: #E2F1FF;
        font-family: 'Share Tech Mono', monospace;
    }

    /* Hide default header */
    header, .stDeployButton {visibility: hidden; height: 0;}

    /* Neon title */
    .neon-title {
        font-family: 'Orbitron', sans-serif;
        font-size: 44px;
        letter-spacing: 1px;
        color: #00E5FF;
        text-shadow: 0 0 8px rgba(0,229,255,0.9), 0 0 22px rgba(255,0,200,0.6);
        margin: 10px 0 4px 0;
    }
    .neon-subtitle {
        color: #9AD1FF;
        font-size: 14px;
        opacity: 0.8;
        margin-bottom: 20px;
    }

    /* Glass panels */
    .glass {
        background: linear-gradient(180deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02));
        border: 1px solid rgba(0,229,255,0.25);
        border-radius: 12px;
        box-shadow: 0 0 30px rgba(0,229,255,0.06);
        backdrop-filter: blur(8px);
    }

    /* Chat bubbles */
    .chat-container {padding: 12px 12px 2px 12px;}
    .msg {
        padding: 12px 14px;
        border-radius: 10px;
        margin: 10px 0;
        line-height: 1.4;
        font-size: 15px;
    }
    .user {background: rgba(0,229,255,0.08); border: 1px solid rgba(0,229,255,0.35);} 
    .bot  {background: rgba(255,0,200,0.08); border: 1px solid rgba(255,0,200,0.35);} 
    .msg small {display: block; opacity: 0.6; margin-top: 6px; font-size: 12px;}

    /* Inputs */
    .neon-input textarea, .neon-input input {
        background: rgba(8,14,22,0.8) !important;
        color: #E2F1FF !important;
        border: 1px solid rgba(0,229,255,0.35) !important;
        border-radius: 10px !important;
    }
    .neon-btn button {
        background: linear-gradient(90deg, #00E5FF 0%, #FF00D4 100%) !important;
        border: 0 !important;
        color: #0b0f17 !important;
        font-weight: 700 !important;
        letter-spacing: 0.5px;
        border-radius: 10px !important;
        box-shadow: 0 0 16px rgba(0,229,255,0.3), 0 0 26px rgba(255,0,212,0.2);
    }

    /* Sidebar */
    section[data-testid="stSidebar"] > div {background: transparent;}
    .sidebar-card {
        padding: 14px;
        margin: 6px 10px 16px 10px;
    }

    /* Footer */
    .footer {
        text-align: center; color: #7FB9FF; opacity: 0.7; font-size: 12px; margin-top: 14px;
    }
    .pill {
        display: inline-block; padding: 2px 10px; font-size: 11px; border-radius: 999px; border: 1px solid rgba(0,229,255,0.35); color: #9AD1FF;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_resource(show_spinner=False)
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        torch_dtype="auto",
        device_map="auto",
    )
    # Precompute END token id once
    try:
        end_id = tokenizer.encode('<|END|>')[-1]
    except Exception:
        end_id = tokenizer.eos_token_id
    return tokenizer, model, end_id

def format_prompt(question: str) -> str:
    return f"Question: {question.strip()}\nAnswer:"

def generate_answer(question: str, temperature: float, top_p: float, top_k: int, max_new_tokens: int, do_sample: bool) -> str:
    tokenizer, model, end_id = load_model()
    prompt = format_prompt(question)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")  # Explicitly move to CUDA
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=end_id,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract model answer after "Answer:"
    start = text.find("Answer:")
    answer = text[start + len("Answer:") :].strip() if start != -1 else text.strip()
    # Stop at custom END token text if present in decoded output
    cut = answer.find("<|END|>")
    if cut != -1:
        answer = answer[:cut].strip()
    return answer

# Sidebar controls
with st.sidebar:
    st.markdown("<div class='neon-title'>NEONBANK</div>", unsafe_allow_html=True)
    st.markdown("<div class='neon-subtitle'>AI Loan Officer</div>", unsafe_allow_html=True)
    st.markdown("<div class='glass sidebar-card'>\n<span class='pill'>Model</span><br/>Llama 3.2 1B (Fine-tuned)\n</div>", unsafe_allow_html=True)

    st.markdown("<div class='glass sidebar-card'>Generation Settings</div>", unsafe_allow_html=True)
    temperature = st.slider("Temperature", 0.0, 1.5, 0.7, 0.05)
    top_p = st.slider("Top-p", 0.1, 1.0, 0.9, 0.05)
    top_k = st.slider("Top-k", 1, 100, 50, 1)
    max_new_tokens = st.slider("Max new tokens", 32, 512, 192, 16)
    do_sample = st.checkbox("Enable sampling", True)

    st.markdown("<div class='glass sidebar-card'>Utilities</div>", unsafe_allow_html=True)
    if st.button("Clear chat history"):
        st.session_state.pop("chat", None)

# Header
st.markdown("<div class='neon-title'>💳 AI Loan Chatbot</div>", unsafe_allow_html=True)
st.markdown("<div class='neon-subtitle'>Ask anything about loan eligibility, documentation, interest, and more.</div>", unsafe_allow_html=True)

# Init chat history
if "chat" not in st.session_state:
    st.session_state.chat = []  # list of {role, content}

# Chat display
chat_panel = st.container()
with chat_panel:
    st.markdown("<div class='glass chat-container'>", unsafe_allow_html=True)
    if not st.session_state.chat:
        st.markdown(
            "<div class='msg bot'>🟣 Hello! I'm your AI loan Chatbot. How can I help you with loans today?</div>",
            unsafe_allow_html=True,
        )
    else:
        for msg in st.session_state.chat:
            role_class = "user" if msg["role"] == "user" else "bot"
            icon = "🧑‍💼" if msg["role"] == "user" else "🤖"
            st.markdown(
                f"<div class='msg {role_class}'>{icon} {msg['content']}</div>",
                unsafe_allow_html=True,
            )
    st.markdown("</div>", unsafe_allow_html=True)

# Input area
with st.form("ask_form", clear_on_submit=True):
    st.markdown("<div class='neon-input'>", unsafe_allow_html=True)
    user_q = st.text_area("", placeholder="Type your loan question here...", height=90)
    st.markdown("</div>", unsafe_allow_html=True)
    submitted = st.form_submit_button("Ask AI", use_container_width=True)

if submitted and user_q and user_q.strip():
    st.session_state.chat.append({"role": "user", "content": user_q.strip()})
    with st.spinner("Analyzing your profile and policies..."):
        try:
            answer = generate_answer(
                question=user_q,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
            )
            st.session_state.chat.append({"role": "assistant", "content": answer})
        except Exception as e:
            st.session_state.chat.append({"role": "assistant", "content": f"Sorry, I hit an error generating the response: {e}"})
    st.rerun()

# Footer
st.markdown(
    "<div class='footer'>ⓘ Responses are AI-generated and for informational purposes only. Consult official bank policies for final decisions.</div>",
    unsafe_allow_html=True,
)