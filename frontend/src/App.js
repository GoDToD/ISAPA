import React, { useState } from "react";
import axios from "axios";

function App() {
  const [inputText, setInputText] = useState("");
  const [file, setFile] = useState(null);
  const [messages, setMessages] = useState([]);

  const handleSend = async () => {
    if (!inputText && !file) return;

    const newUserMsg = { sender: "user", text: inputText, file };
    setMessages((prev) => [...prev, newUserMsg]);

    const formData = new FormData();
    formData.append("text", inputText);
    if (file) {
      formData.append("file", file);
    }

    try {
      const res = await axios.post("http://localhost:5000/upload", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      const botMsg = { sender: "bot", text: res.data.message };
      setMessages((prev) => [...prev, botMsg]);
    } catch (err) {
      console.error(err);
    }

    // Reset
    setInputText("");
    setFile(null);
  };

  return (
    <div className="flex flex-col h-screen bg-gray-100 p-4">
      <h1 className="text-2xl font-bold mb-4 text-center">ChatBot</h1>
      <div className="flex-1 overflow-y-auto bg-white p-4 rounded shadow space-y-4">
        {messages.map((msg, idx) => (
          <div
            key={idx}
            className={`flex ${
              msg.sender === "user" ? "justify-end" : "justify-start"
            }`}
          >
            <div
              className={`max-w-xs p-3 rounded-lg ${
                msg.sender === "user"
                  ? "bg-blue-500 text-white"
                  : "bg-gray-200 text-black"
              }`}
            >
              {msg.text && <p>{msg.text}</p>}
              {msg.file && (
                <p className="text-xs mt-2 italic">
                  📎 {msg.file.name || "Uploaded file"}
                </p>
              )}
            </div>
          </div>
        ))}
      </div>

      <div className="mt-4 flex gap-2">
        <input
          type="text"
          placeholder="Type your message"
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          className="flex-1 border rounded px-3 py-2"
        />
        <input
          type="file"
          onChange={(e) => setFile(e.target.files[0])}
          className="border rounded px-2 py-2"
        />
        <button
          onClick={handleSend}
          className="bg-blue-600 text-white px-4 py-2 rounded"
        >
          Send
        </button>
      </div>
    </div>
  );
}

export default App;
