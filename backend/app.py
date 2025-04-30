from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import os
import pdfplumber
from docx import Document
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
PROFILE_PATH = os.path.join(UPLOAD_FOLDER, "user_profile.txt")
QA_PATH = './data/reconverted_prompt_completion.jsonl'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
model_path = "./model/qwen2.5_7B_Instruct_finetune"

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.float16,
    quantization_config=bnb_config,
    trust_remote_code=True
)
model.eval()

embedder = SentenceTransformer("all-MiniLM-L6-v2")
chat_history = []

qa_chunks = []
qa_answers = []
qa_index = None

def extract_text_from_pdf(filepath):
    with pdfplumber.open(filepath) as pdf:
        return '\n'.join(page.extract_text() or '' for page in pdf.pages)

def extract_text_from_docx(filepath):
    doc = Document(filepath)
    return '\n'.join(paragraph.text for paragraph in doc.paragraphs)

def build_compare_prompt(profile_text, requirement_text, user_query):
    return f"""You are an admissions assistant. Compare the user’s background with the program requirements and determine eligibility.

User Background:
{profile_text}

Program Requirements:
{requirement_text}

Question:
{user_query}

Answer:
"""


def load_qa_dataset_and_index(path):
    global qa_chunks, qa_answers, qa_index
    prompts, completions = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            prompt, completion = item.get("prompt", "").strip(), item.get("completion", "").strip()
            if prompt and completion:
                prompts.append(prompt)
                completions.append(completion)
    qa_chunks = prompts
    qa_answers = completions
    embeddings = embedder.encode(qa_chunks, normalize_embeddings=True)
    qa_index = faiss.IndexFlatIP(embeddings.shape[1])
    qa_index.add(embeddings)


def retrieve_requirement_answer(query, top_k=1, threshold=0.3):
    query_embedding = embedder.encode([query], normalize_embeddings=True)
    D, I = qa_index.search(query_embedding, top_k)
    score = D[0][0]
    idx = I[0][0]
    if score >= threshold and idx >= 0:
        return qa_answers[idx], qa_chunks[idx], score
    return None, None, score


@app.route('/api/upload', methods=['POST'])
def upload_profile():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded."}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file."}), 400

    save_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(save_path)

    if file.filename.endswith('.pdf'):
        text = extract_text_from_pdf(save_path)
    elif file.filename.endswith('.docx'):
        text = extract_text_from_docx(save_path)
    else:
        return jsonify({"error": "Unsupported file format."}), 400

    with open(PROFILE_PATH, "w", encoding="utf-8") as f:
        f.write(text.strip())
    return jsonify({"message": "Profile uploaded and saved."})


@app.route('/api/chat', methods=['POST'])
def chat():
    global chat_history
    data = request.json
    user_prompt = data.get('prompt', '')
    system_prompt = {"role": "system", "content": "You are a helpful assistant."}

    if not os.path.exists(PROFILE_PATH):
        return jsonify({"error": "No user profile uploaded yet."}), 400
    profile_text = open(PROFILE_PATH, encoding="utf-8").read()

    requirement_text, matched_prompt, score = retrieve_requirement_answer(user_prompt)
    if requirement_text is None:
        return jsonify({"response": "Sorry, I couldn't find matching program requirements to compare."})

    rag_prompt = build_compare_prompt(profile_text, requirement_text, user_prompt)
    messages = [system_prompt, {"role": "user", "content": rag_prompt}]

    input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            repetition_penalty=1.2,
            eos_token_id=tokenizer.eos_token_id
        )

    answer = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True).strip()
    return jsonify({
        "response": answer,
        "matched_prompt": matched_prompt,
        "retrieval_score": float(score)
    })


@app.route('/api/reset', methods=['POST'])
def reset_all():
    global chat_history
    chat_history = []
    if os.path.exists(PROFILE_PATH):
        os.remove(PROFILE_PATH)
    return jsonify({"message": "Profile and chat reset."})


if __name__ == "__main__":
    load_qa_dataset_and_index(QA_PATH)
    app.run(host="0.0.0.0", port=5000)
