# Loan Prediction Chatbot

This project implements a RAG-based Q&A chatbot and loan approval predictor using CatBoost (82.71% accuracy).

## Setup
1. Create a virtual environment: python -m venv venv
2. Activate: .\venv\Scripts\Activate.ps1 (Windows) or source venv/bin/activate (Linux/Mac)
3. Install dependencies: pip install -r requirements.txt
4. Run: streamlit run rag_chatbot_with_prediction.py

## Files
- 
ag_chatbot_with_prediction.py: Main Streamlit chatbot script (RAG Q&A and predictions).
- predict_loan_status.py: Script for loan status predictions.
- processed/train_processed.csv: Processed dataset.
- faiss_index/dataset_index.faiss: FAISS index for retrieval.
- embeddings.npy: Cached embeddings.
- best_model.pkl: CatBoost model.
- feature_importance.png: Feature importance plot.
- chatbot_log.txt, chatbot_output.txt: Logs and outputs.

## Usage
- Q&A: Ask questions like "Does good credit history help?"
- Prediction: Input like "Gender: Male, Married: Yes, Dependents: 0, Education: Graduate, Self_Employed: No, ApplicantIncome: 5000, CoapplicantIncome: 0, LoanAmount: 150, Loan_Amount_Term: 360, Credit_History: Good, Property_Area: Urban"

## Deployed App
- Access the chatbot at [https://loan-prediction-chatbot-ktftmkhgdiwaauiwehkqxv.streamlit.app/](https://loan-prediction-chatbot-ktftmkhgdiwaauiwehkqxv.streamlit.app/)
