import { useState, useRef, useEffect } from "react";
import axios from "axios";
import { ChatMessage } from "./types";
import ChatMessageItem from "./components/ChatMessage";
import FileUploader from "./components/FileUploader";
import { v4 as uuidv4 } from "uuid";

function App() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [prompt, setPrompt] = useState("");
  const [loading, setLoading] = useState(false);
  const [docMode, setDocMode] = useState(false); // ✅ 是否开启文档问答模式
  const chatContainerRef = useRef<HTMLDivElement>(null);

  const handleSend = async () => {
    if (!prompt.trim() || loading) return;

    const userMessage: ChatMessage = {
      id: uuidv4(),
      role: "user",
      content: prompt,
    };

    setMessages((prev) => [...prev, userMessage]);
    setPrompt("");
    setLoading(true);

    try {
      const res = await axios.post("http://localhost:5000/api/chat", { prompt });
      const botMessage: ChatMessage = {
        id: uuidv4(),
        role: "assistant",
        content: res.data.response,
      };
      setMessages((prev) => [...prev, botMessage]);
    } catch (error) {
      console.error(error);
    } finally {
      setLoading(false);
    }
  };

  const handleUpload = async (file: File) => {
    const formData = new FormData();
    formData.append('file', file);

    try {
      await axios.post("http://localhost:5000/api/upload", formData, {
        headers: { "Content-Type": "multipart/form-data" }
      });
      alert("File uploaded and processed successfully!");
      setDocMode(true); // ✅ 文件上传成功后，启用文档模式
    } catch (error) {
      console.error(error);
      alert("Upload failed.");
    }
  };

  const handleReset = async () => {
    try {
      await axios.post("http://localhost:5000/api/reset");
      setMessages([]);
      setDocMode(false);
      alert("Chat history and uploaded documents cleared!");
    } catch (error) {
      console.error(error);
      alert("Reset failed.");
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  useEffect(() => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
    }
  }, [messages]);

  return (
    <div style={{
      display: "flex",
      flexDirection: "column",
      alignItems: "center",
      minHeight: "100vh",
      padding: "20px",
      backgroundColor: "#f9f9f9",
      fontFamily: "Arial, sans-serif"
    }}>
      <h2 style={{ marginBottom: "20px" }}> <span style={{ fontWeight: "bold" }}>Chat with Assistant</span></h2>

      {docMode && (
        <div style={{
          backgroundColor: "#e0f7fa",
          color: "#006064",
          padding: "10px 20px",
          borderRadius: "8px",
          marginBottom: "10px",
          fontWeight: "bold"
        }}>
          📄 Document Q&A Mode Active
        </div>
      )}

      <div
        ref={chatContainerRef}
        style={{
          width: "100%",
          maxWidth: "800px",
          height: "500px",
          overflowY: "auto",
          background: "white",
          border: "1px solid #ccc",
          borderRadius: "8px",
          padding: "10px",
          marginBottom: "20px",
          boxShadow: "0px 4px 10px rgba(0, 0, 0, 0.1)"
        }}
      >
        {messages.map(msg => (
          <ChatMessageItem key={msg.id} message={msg} />
        ))}
      </div>

      <textarea
        rows={3}
        value={prompt}
        onChange={(e) => setPrompt(e.target.value)}
        onKeyDown={handleKeyDown}
        placeholder="Enter your message..."
        style={{
          width: "100%",
          maxWidth: "800px",
          marginBottom: "10px",
          padding: "10px",
          borderRadius: "8px",
          border: "1px solid #ccc"
        }}
      />

      <div style={{ width: "100%", maxWidth: "800px", display: "flex", gap: "10px", marginBottom: "10px" }}>
        <button
          onClick={handleSend}
          disabled={loading}
          style={{
            flex: "1",
            padding: "10px",
            borderRadius: "8px",
            border: "none",
            backgroundColor: loading ? "#cccccc" : "#4CAF50",
            color: "white",
            cursor: loading ? "not-allowed" : "pointer"
          }}
        >
          {loading ? "Thinking..." : "Send"}
        </button>

        <FileUploader onUpload={handleUpload} />
      </div>

      <div style={{ width: "100%", maxWidth: "800px" }}>
        <button
          onClick={handleReset}
          style={{
            width: "100%",
            padding: "10px",
            borderRadius: "8px",
            border: "1px solid #ccc",
            backgroundColor: "#f44336",
            color: "white",
            cursor: "pointer"
          }}
        >
          🔄 Clear Chat & Documents
        </button>
      </div>
    </div>
  );
}

export default App;
