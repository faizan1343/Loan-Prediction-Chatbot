
import pandas as pd
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import os
import torch
import re
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, filename="chatbot_log.txt", 
                    format="%(asctime)s - %(levelname)s - %(message)s")

# File paths (relative to repository root)
train_path = "processed/train_processed.csv"
index_path = "faiss_index/dataset_index.faiss"
output_path = "chatbot_output.txt"
model_path = "best_model.pkl"
embeddings_path = "embeddings.npy"
feature_image_path = "feature_importance.png"

# Set device and suppress warnings
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"
device = "cuda" if torch.cuda.is_available() else "cpu"
st.write(f"Using device: {device}")
logging.info(f"Using device: {device}")

# Load CatBoost model
try:
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    st.write("CatBoost model loaded successfully.")
    logging.info("CatBoost model loaded.")
except Exception as e:
    st.error(f"Error loading CatBoost model: {e}")
    logging.error(f"Error loading CatBoost model: {e}")
    st.stop()

# Load documents
try:
    train_df = pd.read_csv(train_path)
    st.write("Training data loaded successfully.")
    logging.info("Training data loaded.")
    # Log dataset statistics
    approval_rate = train_df[train_df['Credit_History'] == 1]['Loan_Status'].mean()
    logging.info(f"Dataset approval rate for Credit_History=1: {approval_rate:.2%}")
except Exception as e:
    st.error(f"Error loading training data: {e}")
    logging.error(f"Error loading training data: {e}")
    st.stop()

# Convert row to descriptive text for embedding, emphasizing Credit_History
def row_to_text(row):
    credit_history = "Good Credit History" if row['Credit_History'] == 1 else "No Credit History"
    loan_status = "Loan Approved" if row['Loan_Status'] == 1 else "Loan Not Approved"
    gender = "Male" if row['Gender'] == 1 else "Female"
    married = "Married" if row['Married'] == 1 else "Not Married"
    self_employed = "Self-Employed" if row['Self_Employed'] == 1 else "Not Self-Employed"
    income_level = "High Income" if row['ApplicantIncome'] > train_df['ApplicantIncome'].quantile(0.75) else "Low Income"
    # Repeat credit_history to boost its weight in embeddings
    return f"{credit_history}, {credit_history}, {loan_status}, {gender}, {married}, {self_employed}, {income_level}"

# Initialize sentence transformer
try:
    embedder = SentenceTransformer('multi-qa-mpnet-base-dot-v1', device=device)
    st.write("Sentence transformer loaded.")
    logging.info("Sentence transformer loaded.")
except Exception as e:
    st.error(f"Error loading sentence transformer: {e}")
    logging.error(f"Error loading sentence transformer: {e}")
    st.stop()

# Load or create FAISS index and embeddings
try:
    if os.path.exists(index_path) and os.path.exists(embeddings_path):
        index = faiss.read_index(index_path)
        embeddings = np.load(embeddings_path)
        st.write("Loaded existing FAISS index and embeddings.")
        logging.info("Loaded existing FAISS index and embeddings.")
    else:
        texts = train_df.apply(row_to_text, axis=1).tolist()
        embeddings = embedder.encode(texts, convert_to_numpy=True, batch_size=32)
        np.save(embeddings_path, embeddings)
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        faiss.write_index(index, index_path)
        st.write("FAISS index and embeddings created and saved.")
        logging.info("FAISS index and embeddings created and saved.")
except Exception as e:
    st.error(f"Error with FAISS index: {e}")
    logging.error(f"Error with FAISS index: {e}")
    st.stop()

# Rule-based response for Q&A
def generate_qa_response(query, retrieved):
    credit_history_query = "Good Credit History" if "good credit history" in query.lower() else "No Credit History" if "no credit history" in query.lower() else None
    if credit_history_query:
        filtered = retrieved[retrieved['Credit_History'] == (1 if credit_history_query == "Good Credit History" else 0)]
    else:
        filtered = retrieved
    if len(filtered) == 0:
        filtered = retrieved

    approvals = sum(filtered['Loan_Status'] == 1)
    total = len(filtered)
    approval_rate = approvals / total if total > 0 else 0

    feature_importance = "Credit_History is the most important factor (35.64 importance), followed by LoanAmount_to_Income (9.50) and Dependents (8.25)."

    if "credit history" in query.lower():
        if approval_rate > 0.5:
            response = f"Yes, good credit history significantly increases loan approval chances (approval rate: {approval_rate:.2%} in similar profiles). {feature_importance} Model accuracy: 82.71%."
        else:
            response = f"Good credit history helps, but approval is not guaranteed (approval rate: {approval_rate:.2%} in similar profiles). {feature_importance} Model accuracy: 82.71%."
    else:
        response = f"Loan approval depends on several factors. {feature_importance} Similar profiles have an approval rate of {approval_rate:.2%}. Model accuracy: 82.71%."
    
    logging.info(f"Query: {query}, Approval rate: {approval_rate:.2%}, Retrieved profiles: {total}")
    return response

# Streamlit app
st.title("Loan Approval Prediction & Q&A Chatbot")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
prompt = st.chat_input("Ask a question (e.g., 'Does good credit history help?') or enter details for prediction (e.g., Gender: Male, Married: Yes, ...)")

if prompt:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    try:
        # Check if input is a prediction request
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
        is_prediction = all(re.search(pattern, prompt, re.IGNORECASE) for pattern in prediction_patterns.values())

        if is_prediction:
            # Parse prediction input
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

            prediction = model.predict(input_df)[0]
            prediction_label = "Approved (Y)" if prediction == 1 else "Not Approved (N)"
            response = f"Loan Prediction: {prediction_label}\nModel accuracy: 82.71% (cross-validation)."
            logging.info(f"Prediction made: {response}")
        else:
            # RAG Q&A
            query_embedding = embedder.encode([prompt], convert_to_numpy=True)
            distances, indices = index.search(query_embedding, k=20)
            retrieved = train_df.iloc[indices[0]]
            response = generate_qa_response(prompt, retrieved)
            logging.info(f"Q&A response for '{prompt}': {response}")

        # Add assistant response
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)

        # Save interaction
        with open(output_path, "a", encoding="utf-8") as f:
            f.write(f"Question: {prompt}\nAnswer: {response}\n\n")
    except Exception as e:
        error_response = f"Error processing input: {e}"
        st.session_state.messages.append({"role": "assistant", "content": error_response})
        with st.chat_message("assistant"):
            st.markdown(error_response)
        logging.error(f"Error processing input '{prompt}': {e}")

# Display feature importance plot
try:
    st.image(feature_image_path, caption="Feature Importance")
except Exception as e:
    st.warning(f"Could not load feature importance plot: {e}")
    logging.error(f"Error loading feature importance plot: {e}")
