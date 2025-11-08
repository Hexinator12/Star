import os
import sys
import time
import os
print(f"Current directory: {os.getcwd()}")
print("Files in directory:", os.listdir('.'))
# Add parent directory to Python path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from AIVoiceAssistant_new import AIVoiceAssistant

class ChatApp:
    def __init__(self):
        print("Initializing University Chat Assistant...")
        try:
            # Get the absolute path to the knowledge base file in the current directory
            current_dir = os.path.dirname(os.path.abspath(__file__))
            kb_path = os.path.join(current_dir, "voice_rag_kb.json")
            print(f"Loading knowledge base from: {kb_path}")
            self.assistant = AIVoiceAssistant(kb_path)
            print("\n" + "="*50)
            print("University Chat Assistant")
            print("Type 'exit' to quit")
            print("="*50 + "\n")
        except Exception as e:
            print(f"Error initializing assistant: {e}")
            print("Starting in simple mode with limited functionality...")
            self.assistant = None
    
    def get_response(self, user_input):
        """Get response from assistant or use fallback."""
        if self.assistant:
            try:
                return self.assistant.interact_with_llm(user_input)
            except Exception as e:
                print(f"\nError getting AI response: {e}")
                return "I'm having trouble connecting to the AI service. Please try again later."
        else:
            # Fallback response if assistant initialization failed
            return f"I'm currently in simple mode. You said: {user_input}"
    
    def start(self):
        """Start the chat interface."""
        print("\nStarting chat...")
        print("Type your message and press Enter. Type 'exit' to quit.\n")
        
        while True:
            try:
                # Get user input
                user_input = input("You: ").strip()
                
                # Check for exit command
                if user_input.lower() in ['exit', 'quit', 'bye']:
                    print("\nGoodbye!")
                    break
                
                if not user_input:
                    continue
                
                # Get and display response
                print("\nAssistant is thinking...")
                start_time = time.time()
                response = self.get_response(user_input)
                response_time = time.time() - start_time
                
                print(f"\nAssistant (took {response_time:.1f}s): {response}\n")
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"\nAn error occurred: {e}")
                print("Please try again or type 'exit' to quit.\n")

if __name__ == "__main__":
    try:
        app = ChatApp()
        app.start()
    except Exception as e:
        print(f"Fatal error: {e}")
        print("The application will now exit.")
        exit(1)
