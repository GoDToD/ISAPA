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

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ==== 加载大模型（4bit量化） ====
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)

model_path = "C:/Qwen2.5_8B_chat_dpo_finetune"  # 🚨 修改为你的实际合并后模型路径！

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.float16,
    quantization_config=bnb_config,
    trust_remote_code=True
)
model.eval()

# ==== 加载 SentenceTransformer 用于向量化 ====
embedder = SentenceTransformer("all-MiniLM-L6-v2")  # 很轻量，快且准确

# ==== 文档和知识库变量 ====
doc_chunks = []
doc_index = None

study_abroad_chunks = []
study_abroad_index = None

# ==== 文档处理工具 ====
def extract_text_from_pdf(filepath):
    with pdfplumber.open(filepath) as pdf:
        text = '\n'.join(page.extract_text() or '' for page in pdf.pages)
    return text

def extract_text_from_docx(filepath):
    doc = Document(filepath)
    text = '\n'.join(paragraph.text for paragraph in doc.paragraphs)
    return text

def split_text(text, chunk_size=300):
    sentences = text.split('\n')
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= chunk_size:
            current_chunk += sentence + " "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

# ==== FAISS索引工具 ====
def build_faiss_index(chunks):
    embeddings = embedder.encode(chunks, normalize_embeddings=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # 余弦相似度检索
    index.add(embeddings)
    return index

def dense_retrieve(index, chunks, query, top_k=3):
    query_embedding = embedder.encode([query], normalize_embeddings=True)
    D, I = index.search(query_embedding, top_k)
    retrieved_chunks = [chunks[i] for i in I[0] if i >= 0]
    return retrieved_chunks

def build_rag_prompt(context_chunks, user_query):
    context_text = "\n".join(context_chunks)
    rag_prompt = f"""Based on the following context, answer the user's question.

Context:
{context_text}

Question:
{user_query}

Answer:"""
    return rag_prompt

# ==== 加载内置留学知识库 ====
def load_study_abroad_knowledge(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    return split_text(content)

# ==== 加载留学知识库并建索引 ====
study_abroad_chunks = load_study_abroad_knowledge("./data/uk_study_guide.txt")  # 📄留学文档路径
study_abroad_index = build_faiss_index(study_abroad_chunks)

# ==== 聊天历史缓存（可选）
chat_history = []

# ==== 聊天接口 ====
@app.route('/api/chat', methods=['POST'])
def chat():
    global chat_history, doc_chunks, doc_index, study_abroad_chunks, study_abroad_index
    data = request.json
    user_prompt = data.get('prompt', '')

    system_prompt = {"role": "system", "content": "You are a helpful assistant."}

    relevant_chunks = []

    # 优先文档检索
    if doc_chunks and doc_index:
        relevant_chunks += dense_retrieve(doc_index, doc_chunks, user_prompt, top_k=2)

    # 留学知识库检索
    if study_abroad_chunks and study_abroad_index:
        relevant_chunks += dense_retrieve(study_abroad_index, study_abroad_chunks, user_prompt, top_k=2)

    if relevant_chunks:
        # 有相关内容，走RAG模式
        rag_prompt = build_rag_prompt(relevant_chunks, user_prompt)
        messages = [system_prompt, {"role": "user", "content": rag_prompt}]
    else:
        # 没有文档，只走普通对话
        chat_history.append({"role": "user", "content": user_prompt})
        messages = [system_prompt] + chat_history

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

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

    generated_text = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
    generated_text = generated_text.strip()

    if not relevant_chunks:
        chat_history.append({"role": "assistant", "content": generated_text})

    return jsonify({"response": generated_text})

# ==== 上传文件接口 ====
@app.route('/api/upload', methods=['POST'])
def upload_file():
    global doc_chunks, doc_index
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded."}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file."}), 400

    save_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(save_path)

    # 解析
    if file.filename.endswith('.pdf'):
        text = extract_text_from_pdf(save_path)
    elif file.filename.endswith('.docx'):
        text = extract_text_from_docx(save_path)
    else:
        return jsonify({"error": "Unsupported file format."}), 400

    # 切分+向量化
    doc_chunks = split_text(text)
    doc_index = build_faiss_index(doc_chunks)

    return jsonify({"message": f"File {file.filename} uploaded and processed successfully!"})

# ==== 清空聊天和上传文档接口 ====
@app.route('/api/reset', methods=['POST'])
def reset_chat():
    global chat_history, doc_chunks, doc_index
    chat_history = []
    doc_chunks = []
    doc_index = None
    return jsonify({"message": "Chat history and uploaded documents cleared."})

# ==== 启动Flask应用 ====
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
