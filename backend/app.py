
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
GUIDE_PATH = './data/uk_study_guide.txt'
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

guide_chunks, guide_index = [], None
qa_chunks, qa_answers, qa_index = [], [], None
doc_chunks, doc_index = [], None

def extract_text_from_pdf(filepath):
    with pdfplumber.open(filepath) as pdf:
        return '\n'.join(page.extract_text() or '' for page in pdf.pages)

def extract_text_from_docx(filepath):
    doc = Document(filepath)
    return '\n'.join(p.text for p in doc.paragraphs)

def split_text(text, chunk_size=300):
    sentences = text.split('\n')
    chunks, current = [], ''
    for s in sentences:
        if len(current) + len(s) < chunk_size:
            current += s + ' '
        else:
            chunks.append(current.strip())
            current = s + ' '
    if current:
        chunks.append(current.strip())
    return chunks

def build_faiss_index(chunks):
    vecs = embedder.encode(chunks, normalize_embeddings=True)
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    return index

def retrieve_with_score(index, chunks, query, top_k=2):
    vec = embedder.encode([query], normalize_embeddings=True)
    D, I = index.search(vec, top_k)
    score = float(D[0][0]) if D[0][0] > 0 else 0.0
    results = [chunks[i] for i in I[0] if i >= 0]
    return results, score

def build_prompt(chunks, query):
    return f"""You are a helpful assistant.

Context:
{chr(10).join(chunks)}

Question:
{query}

Answer:
"""

def load_guide():
    global guide_chunks, guide_index
    with open(GUIDE_PATH, encoding='utf-8') as f:
        guide_chunks = split_text(f.read())
        guide_index = build_faiss_index(guide_chunks)

def load_qa():
    global qa_chunks, qa_answers, qa_index
    prompts, completions = [], []
    with open(QA_PATH, encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            prompts.append(item['prompt'].strip())
            completions.append(item['completion'].strip())
    qa_chunks = [f"Q: {q}\nA: {a}" for q, a in zip(prompts, completions)]
    qa_answers = completions
    qa_index = build_faiss_index(prompts)

@app.route('/api/upload', methods=['POST'])
def upload_file():
    global doc_chunks, doc_index
    file = request.files.get('file')
    if not file:
        return jsonify({"error": "no file"}), 400
    save_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(save_path)
    if file.filename.endswith('.pdf'):
        content = extract_text_from_pdf(save_path)
    elif file.filename.endswith('.docx'):
        content = extract_text_from_docx(save_path)
    else:
        return jsonify({"error": "unsupported format"}), 400
    doc_chunks = split_text(content)
    doc_index = build_faiss_index(doc_chunks)
    return jsonify({"message": "file uploaded"})

@app.route('/api/chat', methods=['POST'])

def chat():
    import re
    
    data = request.json
    query = data.get('prompt', '')
    threshold = 0.3
    used_chunks = []
    used_sources = []

    
    # 优先级判断关键词
    decision_keywords = ["requirement", "qualified", "eligibility", "can i apply", "suitable", "am i eligible", "fit the program"]
    is_eligibility_question = any(kw in query.lower() for kw in decision_keywords)

    # Always include uploaded user document content
    if doc_chunks:
        used_chunks += doc_chunks[:2]  # 拼接前两个块以节省上下文空间
        used_sources.append('document')

    if qa_index and (is_eligibility_question or not used_chunks):
        qa_retrieved, qa_score = retrieve_with_score(qa_index, qa_chunks, query)
        if qa_score >= 0.25:
            used_chunks += qa_retrieved
            used_sources.append('qa')

    if guide_index and not is_eligibility_question:
        guide_retrieved, guide_score = retrieve_with_score(guide_index, guide_chunks, query)
        if guide_score >= 0.25:
            used_chunks += guide_retrieved
            used_sources.append('guide')

        qa_retrieved, qa_score = retrieve_with_score(qa_index, qa_chunks, query)
        if qa_score >= threshold:
            used_chunks += qa_retrieved
            used_sources.append('qa')

    if guide_index:
        guide_retrieved, guide_score = retrieve_with_score(guide_index, guide_chunks, query)
        if guide_score >= threshold:
            used_chunks += guide_retrieved
            used_sources.append('guide')

    if doc_index:
        doc_retrieved, doc_score = retrieve_with_score(doc_index, doc_chunks, query)
        if doc_score >= threshold:
            used_chunks += doc_retrieved
            used_sources.append('document')

    if used_chunks:
        prompt = build_prompt(used_chunks, query)
    else:
        prompt = query

    messages = [{"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}]

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

    result = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True).strip()
    return jsonify({"response": result, "sources": used_sources})

@app.route('/api/reset', methods=['POST'])
def reset():
    global doc_chunks, doc_index
    doc_chunks, doc_index = [], None
    return jsonify({"message": "reset"})

if __name__ == "__main__":
    load_guide()
    load_qa()
    app.run(host="0.0.0.0", port=5000)
