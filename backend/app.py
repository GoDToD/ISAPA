from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allow frontend to access backend

@app.route('/upload', methods=['POST'])
def upload():
    text = request.form.get("text", "")
    file = request.files.get("file", None)
    
    # Process text and file
    response_text = f"Received text: {text}"
    
    if file:
        response_text += f" | File received: {file.filename}"
    
    return jsonify({"message": response_text})

if __name__ == '__main__':
    app.run(debug=True)