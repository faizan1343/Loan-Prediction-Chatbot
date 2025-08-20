import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import pandas as pd
import numpy as np
import requests
import pickle
import os
import re
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, filename="logs/chatbot_log.txt", 
                    format="%(asctime)s - %(levelname)s - %(message)s")

# File paths
BASE_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
LORA_DIR = os.path.join("output", "fine_tuned_model")
LORA_HF = "faizan1343/loan-assistant-lora"  # Replace with your Hugging Face LoRA ID
MODEL_PATH = "rag_catboost/best_model.pkl"
TRAIN_PATH = "rag_catboost/processed/train_processed.csv"
OUTPUT_PATH = "rag_catboost/chatbot_output.txt"

# Load CatBoost model
try:
    with open(MODEL_PATH, "rb") as f:
        catboost_model = pickle.load(f)
    logging.info("CatBoost model loaded.")
except Exception as e:
    st.error(f"Error loading CatBoost model: {e}")
    logging.error(f"Error loading CatBoost model: {e}")
    st.stop()

# Load training data (optional, for context or validation)
try:
    train_df = pd.read_csv(TRAIN_PATH)
    logging.info("Training data loaded.")
except Exception as e:
    st.error(f"Error loading training data: {e}")
    logging.error(f"Error loading training data: {e}")
    st.stop()

# Load LLaMA model and tokenizer
if not os.path.exists(LORA_DIR):
    os.makedirs(LORA_DIR, exist_ok=True)
    AutoTokenizer.from_pretrained(BASE_MODEL).save_pretrained(LORA_DIR)
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype="auto", device_map="auto")
    PeftModel.from_pretrained(base_model, LORA_HF).save_pretrained(LORA_DIR)

@st.cache_resource(show_spinner=False)
def load_llama_model():
    tokenizer = AutoTokenizer.from_pretrained(LORA_DIR)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    quantization_config = BitsAndBytesConfig(load_in_4bit=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype="auto",
        device_map="auto",
        quantization_config=quantization_config,
    )
    model = PeftModel.from_pretrained(base_model, LORA_DIR)
    try:
        end_id = tokenizer.encode('<|END|>')[-1]
    except Exception:
        end_id = tokenizer.eos_token_id
    return tokenizer, model, end_id

# Function to generate LLaMA response
def generate_llama_answer(question: str, temperature: float, top_p: float, top_k: int, max_new_tokens: int, do_sample: bool) -> str:
    tokenizer, model, end_id = load_llama_model()
    prompt = f"Question: {question.strip()}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
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
    start = text.find("Answer:")
    answer = text[start + len("Answer:") :].strip() if start != -1 else text.strip()
    cut = answer.find("<|END|>")
    if cut != -1:
        answer = answer[:cut].strip()
    return answer

# Function to predict loan approval with CatBoost
def predict_loan_approval(prompt: str) -> str:
    prediction_patterns = {
        'Gender': r"Gender:\s*(\w+)",
        'Married': r"Married:\s*(\w+)",
        'Dependents': r"Dependents:\s*(\w+)",
        'Education': r"Education:\s*(\w+\s*\w*)",
        'Self_Employed': r"Self_Employed:\s*(\w+)",
        'ApplicantIncome': r"ApplicantIncome:\s*(\d+\.?\d*)",
        'CoapplicantIncome': r"CoapplicantIncome:\s*(\d+\.?\d*)",
        'LoanAmount': r"LoanAmount:\s*(\d+\.?\d*)",
        'Loan_Amount_Term': r"Loan_Amount_Term:\s*(\d+\.?\d*)",
        'Credit_History': r"Credit_History:\s*(\w+)",
        'Property_Area': r"Property_Area:\s*(\w+)"
    }
    try:
        if not all(re.search(pattern, prompt, re.IGNORECASE) for pattern in prediction_patterns.values()):
            return None  # Not a prediction request
        input_dict = {}
        for key, pattern in prediction_patterns.items():
            match = re.search(pattern, prompt, re.IGNORECASE)
            if not match:
                raise ValueError(f"Invalid or missing {key}")
            input_dict[key] = match.group(1)

        input_data = {
            'Gender': "1" if input_dict['Gender'].lower() == "male" else "0",
            'Married': "1" if input_dict['Married'].lower() == "yes" else "0",
            'Dependents': "3" if input_dict['Dependents'] == "3+" else input_dict['Dependents'],
            'Education': "1" if input_dict['Education'].lower() == "graduate" else "0",
            'Self_Employed': "1" if input_dict['Self_Employed'].lower() == "yes" else "0",
            'ApplicantIncome': float(input_dict['ApplicantIncome']),
            'CoapplicantIncome': float(input_dict['CoapplicantIncome']),
            'LoanAmount': float(input_dict['LoanAmount']),
            'Loan_Amount_Term': float(input_dict['Loan_Amount_Term']),
            'Credit_History': "1" if input_dict['Credit_History'].lower() == "good" else "0",
            'Property_Rural': "1" if input_dict['Property_Area'].lower() == "rural" else "0",
            'Property_Semiurban': "1" if input_dict['Property_Area'].lower() == "semiurban" else "0",
            'Property_Urban': "1" if input_dict['Property_Area'].lower() == "urban" else "0",
            'TotalIncome': float(input_dict['ApplicantIncome']) + float(input_dict['CoapplicantIncome']),
            'LoanAmount_to_Income': float(input_dict['LoanAmount']) / (float(input_dict['ApplicantIncome']) + float(input_dict['CoapplicantIncome']) + 1e-6),
            'Credit_History_LoanAmount': (1 if input_dict['Credit_History'].lower() == "good" else 0) * float(input_dict['LoanAmount'])
        }

        input_df = pd.DataFrame([input_data])
        expected_features = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'ApplicantIncome', 
                            'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 
                            'Property_Rural', 'Property_Semiurban', 'Property_Urban', 'TotalIncome', 
                            'LoanAmount_to_Income', 'Credit_History_LoanAmount']
        input_df = input_df[expected_features]
        prediction = catboost_model.predict(input_df)[0]
        return f"Loan Prediction: {'Approved (Y)' if prediction == 1 else 'Not Approved (N)'}\nModel accuracy: 82.71% (cross-validation)."
    except Exception as e:
        logging.error(f"Error in CatBoost prediction: {e}")
        return f"Error: Could not generate prediction ({e})"

# Streamlit app configuration
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
    header, .stDeployButton {visibility: hidden; height: 0;}
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
    .glass {
        background: linear-gradient(180deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02));
        border: 1px solid rgba(0,229,255,0.25);
        border-radius: 12px;
        box-shadow: 0 0 30px rgba(0,229,255,0.06);
        backdrop-filter: blur(8px);
    }
    .chat-container {padding: 12px 12px 2px 12px;}
    .msg {
        padding: 12px 14px;
        border-radius: 10px;
        margin: 10px 0;
        line-height: 1.4;
        font-size: 15px;
    }
    .user {background: rgba(0,229,255,0.08); border: 1px solid rgba(0,229,255,0.35);} 
    .bot {background: rgba(255,0,200,0.08); border: 1px solid rgba(255,0,200,0.35);} 
    .msg small {display: block; opacity: 0.6; margin-top: 6px; font-size: 12px;}
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
    section[data-testid="stSidebar"] > div {background: transparent;}
    .sidebar-card {
        padding: 14px;
        margin: 6px 10px 16px 10px;
    }
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

# Sidebar controls
with st.sidebar:
    st.markdown("<div class='neon-title'>NEONBANK</div>", unsafe_allow_html=True)
    st.markdown("<div class='neon-subtitle'>AI Loan Officer</div>", unsafe_allow_html=True)
    st.markdown("<div class='glass sidebar-card'>\n<span class='pill'>Model</span><br/>Llama 3.2 1B (Fine-tuned with LoRA) + CatBoost\n</div>", unsafe_allow_html=True)
    st.markdown("<div class='glass sidebar-card'>Generation Settings</div>", unsafe_allow_html=True)
    temperature = st.slider("Temperature", 0.0, 1.5, 0.7, 0.05)
    top_p = st.slider("Top-p", 0.1, 1.0, 0.9, 0.05)
    top_k = st.slider("Top-k", 1, 100, 50, 1)
    max_new_tokens = st.slider("Max new tokens", 32, 512, 192, 16)
    do_sample = st.checkbox("Enable sampling", True)
    st.markdown("<div class='glass sidebar-card'>Utilities</div>", unsafe_allow_html=True)
    if st.button("Clear chat history"):
        st.session_state.pop("messages", None)

# Header
st.markdown("<div class='neon-title'>💳 AI Loan Chatbot</div>", unsafe_allow_html=True)
st.markdown("<div class='neon-subtitle'>Ask about loan eligibility, documentation, or enter details for a prediction (e.g., Gender: Male, Married: Yes, Dependents: 2, Education: Graduate, Self_Employed: No, ApplicantIncome: 5000, CoapplicantIncome: 2000, LoanAmount: 150, Loan_Amount_Term: 360, Credit_History: Good, Property_Area: Urban).</div>", unsafe_allow_html=True)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Chat display
chat_panel = st.container()
with chat_panel:
    st.markdown("<div class='glass chat-container'>", unsafe_allow_html=True)
    if not st.session_state.messages:
        st.markdown(
            "<div class='msg bot'>🟣 Hello! I'm your AI Loan Chatbot. Ask about loans or provide details for a prediction!</div>",
            unsafe_allow_html=True,
        )
    else:
        for message in st.session_state.messages:
            role_class = "user" if message["role"] == "user" else "bot"
            icon = "🧑‍💼" if message["role"] == "user" else "🤖"
            st.markdown(
                f"<div class='msg {role_class}'>{icon} {message['content']}</div>",
                unsafe_allow_html=True,
            )
    st.markdown("</div>", unsafe_allow_html=True)

# Input area
with st.form("ask_form", clear_on_submit=True):
    st.markdown("<div class='neon-input'>", unsafe_allow_html=True)
    user_q = st.text_area("", placeholder="Ask a question (e.g., 'Does good credit history help?') or enter details for prediction (e.g., Gender: Male, Married: Yes, ...)", height=90)
    st.markdown("</div>", unsafe_allow_html=True)
    submitted = st.form_submit_button("Ask AI", use_container_width=True)

if submitted and user_q and user_q.strip():
    st.session_state.messages.append({"role": "user", "content": user_q.strip()})
    with st.chat_message("user"):
        st.markdown(user_q.strip())
    with st.spinner("Analyzing your profile and policies..."):
        try:
            # Try prediction first
            prediction_response = predict_loan_approval(user_q)
            if prediction_response:
                response = prediction_response
                logging.info(f"Prediction response for '{user_q}': {response}")
            else:
                # Fallback to LLaMA conversational response
                response = generate_llama_answer(
                    question=user_q,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                )
                logging.info(f"LLaMA response for '{user_q}': {response}")
            st.session_state.messages.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.markdown(response)
            with open(OUTPUT_PATH, "a", encoding="utf-8") as f:
                f.write(f"Question: {user_q}\nAnswer: {response}\n\n")
        except Exception as e:
            error_response = f"Error processing input: {e}"
            st.session_state.messages.append({"role": "assistant", "content": error_response})
            with st.chat_message("assistant"):
                st.markdown(error_response)
            logging.error(f"Error processing input '{user_q}': {e}")

# Footer
st.markdown(
    "<div class='footer'>ⓘ Responses are AI-generated and for informational purposes only. Consult official bank policies for final decisions.</div>",
    unsafe_allow_html=True,
)