
# Study Abroad RAG System

This project is a document-augmented question answering system combining a React frontend with a Flask backend. It supports real-time conversational interaction with a local large language model (LLM), document upload and semantic retrieval, and integrates a study abroad encyclopedia along with a curated QA dataset to enhance answering capabilities.

---

## ✨ Core Features

- **Real-time Chat Interface**  
  ChatGPT-style multi-turn conversation with instant feedback.

- **Document Upload and Parsing**  
  Upload PDF and Word documents; the system automatically parses and integrates their content into the retrieval process.

- **Dense Semantic Retrieval (RAG)**  
  Supports retrieval from three sources: user-uploaded documents, a built-in study abroad encyclopedia, and a curated study abroad QA dataset.

- **Local Large Language Model Inference**  
  Efficient local inference with 4-bit quantization (BitsAndBytes), supporting large models such as Qwen2.5-8B-Chat.

- **Multi-Source Knowledge Fusion**  
  Dynamically retrieves and fuses multiple knowledge sources to build a context-aware prompt, delivering highly relevant answers.

---

## 🛠️ Technology Stack

**Frontend**:
- React
- TypeScript
- Axios
- Vite (or Create React App)
- LocalStorage (optional)

**Backend**:
- Flask
- Flask-CORS
- Huggingface Transformers
- PyTorch
- BitsAndBytes (4-bit quantization)
- Sentence-Transformers
- FAISS
- pdfplumber (for PDF parsing)
- python-docx (for Word parsing)

---

## 🗂️ Project Structure

```plaintext
study-abroad-rag-system/
├── README.md
├── LICENSE
├── .gitignore
├── backend/
│   ├── app.py
│   ├── requirements.txt
│   ├── uploads/
│   ├── data/
│   │   ├── uk_study_guide.txt
│   │   └── reconverted_prompt_completion.jsonl
├── frontend/
│   ├── package.json
│   ├── tsconfig.json
│   ├── src/
│   │   ├── App.tsx
│   │   ├── components/
│   │   ├── types/
│   └── public/
└── docs/ (optional)
```

---

## 🚀 Quick Start Guide

This guide provides step-by-step instructions to set up and run the Study Abroad RAG System locally.

### 1. Clone the Repository

```bash
git clone https://github.com/yourname/study-abroad-rag-system.git
cd study-abroad-rag-system
```

### 2. Set Up the Backend (Flask)

```bash
cd backend
python -m venv env
```

Activate the virtual environment:

- On **Windows**:
  ```bash
  env\Scriptsctivate
  ```
- On **Linux/MacOS**:
  ```bash
  source env/bin/activate
  ```

Install the required Python packages:

```bash
pip install -r requirements.txt
```

Start the backend server:

```bash
python app.py
```

> 📌 Note:  
> Ensure that your merged and quantized LLM model (e.g., Qwen2.5-8B-Chat) is placed correctly, and the model path is properly configured in `app.py`.

By default, the Flask server will run at `http://localhost:5000/`.

---

### 3. Set Up the Frontend (React)

In a new terminal window, navigate to the frontend directory:

```bash
cd frontend
npm install
```

Start the frontend development server:

```bash
npm run dev
```

The React frontend will be available at `http://localhost:5173/`.

---

### 4. Prepare Knowledge Sources

Make sure the following knowledge files are placed under `backend/data/`:

- `uk_study_guide.txt` — Study abroad encyclopedia
- `reconverted_prompt_completion.jsonl` — Study abroad QA dataset

These files are preloaded automatically when the backend starts.

---

### ✅ 5. System is Ready!

- Access the system at `http://localhost:5173/`
- Start chatting or upload documents to enhance your questions with document-based knowledge.
- Enjoy the intelligent document-augmented conversation!

---

## 📚 Knowledge Sources

- **User-Uploaded Documents** (PDF/Word)
- **Study Abroad Encyclopedia** (`uk_study_guide.txt`)
- **Study Abroad QA Dataset** (`study_abroad_qa.jsonl`)
