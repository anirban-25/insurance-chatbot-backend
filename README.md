# Insurance Chatbot Backend (RAG with Langchain + Pinecone)

This is the backend for an AI-powered insurance policy chatbot that utilizes Retrieval-Augmented Generation (RAG) with Langchain and Pinecone. The system is built using Flask and processes PDF documents to provide intelligent answers based on the content.

---

## 📁 Project Structure

├── .ipynb_checkpoints/ # Jupyter notebook auto-checkpoints (ignored) ├── documents/ │ └── t100-policy.pdf # Demo PDF used for implementing RAG ├── venv/ # Virtual environment (ignored) ├── .env # Environment variables (ignored) ├── app.py # Flask app entry point ├── requirements.txt # Python dependencies └── test.ipynb # Notebook with full RAG implementation using Langchain and Pinecone


---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/anirban-25/insurance-chatbot-backend.git
cd insurance-chatbot-backend
2. Install Dependencies

pip install -r requirements.txt
Make sure you are using Python 3.8 or higher.

3. Run the Flask App
python app.py
The Flask server will start on http://localhost:5000.

📄 Documents
documents/t100-policy.pdf:
This is the sample insurance policy document used for testing the RAG-based chatbot.

📓 Notebook Overview
test.ipynb:
This Jupyter Notebook contains the full implementation of the Retrieval-Augmented Generation pipeline using:

Langchain for document loading, chunking, embedding, and QA

Pinecone for vector database storage and retrieval

Use this notebook to explore, modify, or test the RAG pipeline with other documents.

🛡️ .gitignore
Make sure your .gitignore includes the following to avoid pushing sensitive or unnecessary files:

.env
venv/
.ipynb_checkpoints/
📬 Contact
For questions or suggestions, feel free to open an issue or reach out directly.
