Loan Approval Assistant Chatbot
A fine-tuned LLaMA 3.2 (1B) model for answering loan-related queries, built with Hugging Face Transformers, LoRA, and 4-bit quantization on Windows (NVIDIA GTX 1650 Ti). Trained on loan_qa_dataset_llama.csv (136 samples) with ~3 epochs (loss: 2.0057). Uses Ollama for local inference and Streamlit for an interactive UI with a cyberpunk theme.
Setup

Install dependencies: pip install -r requirements.txt
Fine-tune: python fine_tune_llama3_2.py
Validate dataset: python validate_dataset.py
Run UI: streamlit run app.py
Run with Ollama: ollama run loan_assistant

Files

app.py: Streamlit chatbot UI with LLaMA 3.2 inference.
fine_tune_llama3_2.py: Fine-tuning script for LLaMA 3.2.
loan_qa_dataset_llama.csv: Training dataset (136 Q&A pairs).
validate_dataset.py: Dataset validation script.
Modelfile: Ollama configuration for local inference.
rag_catboost/: Subfolder containing legacy RAG-based Q&A and CatBoost loan prediction model (82.71% accuracy).

Requirements

Python 3.10
CUDA 11.8, cuDNN 8.6
torch==2.4.0+cu118, transformers==4.45.2, datasets==2.21.0, peft==0.12.0, bitsandbytes==0.43.3, streamlit, pandas

Usage

Q&A: Ask questions like “What is the minimum credit score for a loan?” or “Can I get a loan with bad credit?” via the Streamlit UI or Ollama.
Ollama: Run ollama run loan_assistant "Question: Your question here\nAnswer:".

Legacy RAG/CatBoost Model
The rag_catboost/ subfolder contains a RAG-based Q&A chatbot and CatBoost loan prediction model. See rag_catboost/README.md for details.