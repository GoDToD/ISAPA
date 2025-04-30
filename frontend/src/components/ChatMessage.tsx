// React component (e.g., inside ChatMessage.tsx or MessageBubble.tsx)

import React from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

interface ChatMessageProps {
  content: string;
  role: 'user' | 'assistant';
}

const ChatMessage: React.FC<ChatMessageProps> = ({ content, role }) => {
  const isUser = role === 'user';
  return (
    <div
      style={{
        textAlign: isUser ? 'right' : 'left',
        background: isUser ? '#d4f8cb' : '#f2f2f2',
        borderRadius: '8px',
        padding: '12px',
        marginBottom: '8px',
        maxWidth: '90%',
        alignSelf: isUser ? 'flex-end' : 'flex-start'
      }}
    >
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          strong: ({ children }) => <strong style={{ fontWeight: 600 }}>{children}</strong>,
          li: ({ children }) => <li style={{ marginLeft: 12 }}>{children}</li>
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
};

export default ChatMessage;
