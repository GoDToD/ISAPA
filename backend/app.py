from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import os

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ==== 加载模型（使用4bit量化） ====
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)

model_path = "C:/Qwen2.5_8B_chat_dpo_finetune"  # 🚨 修改成你的合并后的模型路径！

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.float16,
    quantization_config=bnb_config,
    trust_remote_code=True
)
model.eval()

# ==== 聊天历史缓存（简单版）
chat_history = []

# ==== 聊天接口 ====
@app.route('/api/chat', methods=['POST'])
def chat():
    global chat_history
    data = request.json
    user_prompt = data.get('prompt', '')

    # 添加到历史（role分配）
    chat_history.append({"role": "user", "content": user_prompt})

    # system prompt只在最开始加一次
    system_prompt = {"role": "system", "content": "You are a helpful assistant."}

    # 构建最终消息列表
    messages = [system_prompt] + chat_history

    # 使用正确的chat模板
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    # 推理
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

    # 解码，只拿新生成的部分
    generated_text = tokenizer.decode(outputs[0][input_ids.shape[-1]:], skip_special_tokens=True)
    generated_text = generated_text.strip()

    # 把assistant回答也加入历史
    chat_history.append({"role": "assistant", "content": generated_text})

    return jsonify({"response": generated_text})

# ==== 上传文件接口 ====
@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded."}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file."}), 400

    save_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(save_path)

    return jsonify({"message": f"File {file.filename} uploaded successfully."})

# ==== 清空历史接口（可选）
@app.route('/api/reset', methods=['POST'])
def reset_chat():
    global chat_history
    chat_history = []
    return jsonify({"message": "Chat history cleared."})

# ==== 启动Flask应用 ====
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
