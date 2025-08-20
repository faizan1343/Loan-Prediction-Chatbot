Legacy RAG/CatBoost Loan Prediction Chatbot
This subfolder contains a RAG-based Q&A chatbot and a CatBoost model (82.71% accuracy) for loan approval prediction.
Files

rag_chatbot_with_prediction.py: Main Streamlit script for RAG Q&A and predictions.
predict_loan_status.py: Loan status prediction script.
processed/train_processed.csv: Processed dataset.
faiss_index/dataset_index.faiss: FAISS index for retrieval.
embeddings.npy: Cached embeddings.
best_model.pkl: CatBoost model.
feature_importance.png: Feature importance plot.
chatbot_log.txt, chatbot_output.txt: Logs and outputs.

Setup

Install dependencies: pip install -r requirements.txt
Run: streamlit run rag_chatbot_with_prediction.py

Usage

Q&A: Ask questions like “Does good credit history help?”
Prediction: Input features like “Gender: Male, Married: Yes, Dependents: 0, Education: Graduate, Self_Employed: No, ApplicantIncome: 5000, CoapplicantIncome: 0, LoanAmount: 150, Loan_Amount_Term: 360, Credit_History: Good, Property_Area: Urban”.
