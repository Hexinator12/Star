import { useEffect, useRef } from 'react';
import MessageBubble from './MessageBubble';
import TypingIndicator from './TypingIndicator';
import { Sparkles } from 'lucide-react';

export default function ChatWindow({ messages, isLoading }) {
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isLoading]);

  return (
    <div className="flex-1 overflow-y-auto custom-scrollbar p-4 space-y-4">
      <div className="max-w-4xl mx-auto">
        {/* Welcome Message */}
        {messages.length === 0 && !isLoading && (
          <div className="text-center py-12 animate-fade-in">
            <div className="inline-flex items-center justify-center w-16 h-16 bg-primary-100 dark:bg-primary-900 rounded-full mb-4">
              <Sparkles className="w-8 h-8 text-primary-500" />
            </div>
            <h2 className="text-2xl font-bold text-gray-800 dark:text-white mb-2">
              Welcome to RAG AI Assistant!
            </h2>
            <p className="text-gray-600 dark:text-gray-400 mb-6">
              Ask me anything about university programs, courses, admissions, fees, and more.
            </p>
            
            {/* Quick Action Buttons */}
            <div className="flex flex-wrap justify-center gap-2 mt-6">
              <button className="btn-secondary text-sm">
                📚 What programs do you offer?
              </button>
              <button className="btn-secondary text-sm">
                💰 Tell me about fees
              </button>
              <button className="btn-secondary text-sm">
                🎓 Admission requirements
              </button>
              <button className="btn-secondary text-sm">
                👨‍🏫 Faculty information
              </button>
            </div>
          </div>
        )}

        {/* Messages */}
        {messages.map((msg, index) => (
          <MessageBubble
            key={index}
            message={msg.text}
            isUser={msg.isUser}
          />
        ))}

        {/* Typing Indicator */}
        {isLoading && <TypingIndicator />}

        {/* Scroll Anchor */}
        <div ref={messagesEndRef} />
      </div>
    </div>
  );
}
