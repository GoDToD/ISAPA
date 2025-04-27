import { ChatMessage } from "../types";

interface Props {
  message: ChatMessage;
}

export default function ChatMessageItem({ message }: Props) {
  return (
    <div style={{
      textAlign: message.role === 'user' ? 'right' : 'left',
      margin: '10px 0'
    }}>
      <div style={{
        display: 'inline-block',
        background: message.role === 'user' ? '#DCF8C6' : '#F1F0F0',
        borderRadius: '8px',
        padding: '10px',
        maxWidth: '70%',
      }}>
        {message.content}
      </div>
    </div>
  );
}
