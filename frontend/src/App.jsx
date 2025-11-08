import { useState } from 'react';
import axios from 'axios';
import Header from './components/Header';
import ChatWindow from './components/ChatWindow';
import InputBox from './components/InputBox';
import { AlertCircle, Trash2 } from 'lucide-react';

function App() {
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const sendMessage = async (text) => {
    // Add user message
    const userMessage = { text, isUser: true };
    setMessages(prev => [...prev, userMessage]);
    setIsLoading(true);
    setError(null);

    try {
      // Call API
      const response = await axios.post('http://localhost:8000/chat', {
        message: text
      });

      // Add assistant response
      const assistantMessage = {
        text: response.data.response,
        isUser: false
      };
      setMessages(prev => [...prev, assistantMessage]);

    } catch (err) {
      console.error('Error sending message:', err);
      setError('Failed to get response. Please make sure the API server is running.');
      
      // Add error message
      const errorMessage = {
        text: 'Sorry, I encountered an error. Please try again or check if the server is running.',
        isUser: false
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const clearChat = () => {
    setMessages([]);
    setError(null);
  };

  return (
    <div className="flex flex-col h-screen">
      {/* Header */}
      <Header />

      {/* Error Banner */}
      {error && (
        <div className="bg-red-50 dark:bg-red-900/20 border-b border-red-200 dark:border-red-800 px-4 py-3">
          <div className="max-w-7xl mx-auto flex items-center space-x-2 text-red-800 dark:text-red-200">
            <AlertCircle className="w-5 h-5 flex-shrink-0" />
            <p className="text-sm">{error}</p>
          </div>
        </div>
      )}

      {/* Chat Window */}
      <ChatWindow messages={messages} isLoading={isLoading} />

      {/* Clear Chat Button */}
      {messages.length > 0 && (
        <div className="px-4 pb-2">
          <div className="max-w-4xl mx-auto flex justify-end">
            <button
              onClick={clearChat}
              className="text-sm text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200 flex items-center space-x-1 transition-colors"
            >
              <Trash2 className="w-4 h-4" />
              <span>Clear chat</span>
            </button>
          </div>
        </div>
      )}

      {/* Input Box */}
      <InputBox onSendMessage={sendMessage} isLoading={isLoading} />
    </div>
  );
}

export default App;
