from qdrant_client import QdrantClient, models
from llama_index.llms.ollama import Ollama
from llama_index.core import Document, Settings
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from typing import Optional, List, Dict, Any
import json
import os
import warnings
from hashlib import md5
from functools import lru_cache

# Suppress warnings
warnings.filterwarnings("ignore")

class AIVoiceAssistant:
    def __init__(self, knowledge_base_path="voice_rag_kb.json"):
        self.response_cache = {}
        self.cache_hits = 0
        self.total_queries = 0
        self.knowledge_base_path = knowledge_base_path
        
        # Initialize Qdrant client
        self._init_qdrant()
        
        # Initialize models and embeddings
        self._init_models()
        
        # Create knowledge base if it doesn't exist
        self.create_kb()
        
        # Create chat engine
        self._create_chat_engine()

    def _init_qdrant(self):
        """Initialize Qdrant client with enhanced logging."""
        try:
            print("\n=== Initializing Qdrant Client ===")
            self._client = QdrantClient(
                url="http://localhost:6333",
                prefer_grpc=False,
                timeout=10.0
            )
            # Test the connection and list collections
            collections = self._client.get_collections()
            print("\n=== Qdrant Collections ===")
            for collection in collections.collections:
                # Get detailed collection info to get the vector count
                try:
                    info = self._client.get_collection(collection.name)
                    print(f"- {collection.name} (Vectors: {info.vectors_count if hasattr(info, 'vectors_count') else 'N/A'})")
                except Exception as e:
                    print(f"- {collection.name} (Error getting vector count: {str(e)})")
            print("\n✓ Successfully connected to Qdrant!")
        except Exception as e:
            print(f"❌ Failed to connect to Qdrant: {e}")
            raise
            raise

    def _is_ollama_running(self) -> bool:
        """Check if Ollama server is running and accessible."""
        import requests
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False

    def _init_models(self):
        """Initialize LLM and embedding models with better error handling and timeouts."""
        try:
            # Check if Ollama server is running
            if not self._is_ollama_running():
                print("Ollama server is not running. Please start it with: ollama serve")
                print("Trying to start Ollama server...")
                import subprocess
                try:
                    # Start Ollama in the background
                    subprocess.Popen(
                        ["ollama", "serve"], 
                        stdout=subprocess.PIPE, 
                        stderr=subprocess.PIPE
                    )
                    # Wait for server to start
                    import time
                    time.sleep(5)
                except Exception as e:
                    print(f"Failed to start Ollama server: {e}")
                    print("Please start Ollama manually in a separate terminal with: ollama serve")
                    raise

            # Initialize LLM with smaller model and optimized settings
            print("Initializing LLM with optimized settings...")
            # Using gemma:2b - much faster and lighter than llama2:7b
            self.llm = Ollama(
                model="gemma:2b",
                base_url="http://localhost:11434",
                request_timeout=30.0,
                temperature=0.1,  # Lower temperature for more focused responses
                num_ctx=2048,     # Smaller context for faster processing
            )
            
            # Initialize embedding model
            print("Initializing Embedding Model (BAAI/bge-small-en-v1.5)...")
            self.embeddings = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
            
            # Update settings
            Settings.llm = self.llm
            Settings.embed_model = self.embeddings
            
            # Test embedding generation
            test_text = "Test embedding generation"
            embedding = self.embeddings.get_text_embedding(test_text)
            print(f"\n✓ Models initialized successfully!")
            print(f"  - Test embedding dimension: {len(embedding)}")
            
        except Exception as e:
            print(f"❌ Error initializing models: {e}")
            import traceback
            traceback.print_exc()
            raise

    def _create_kb(self):
        """Create or update the knowledge base in Qdrant with detailed logging."""
        try:
            print("\n=== Creating Knowledge Base ===")
            
            # Delete existing collection if it exists
            try:
                print("Checking for existing collection...")
                self._client.get_collection("university_kb")
                print("Deleting existing collection...")
                self._client.delete_collection("university_kb")
                print("✓ Deleted existing collection.")
            except Exception as e:
                if "not found" not in str(e):
                    raise
                print("No existing collection found, creating new one...")
            
            # Create new collection with simple vector configuration
            # Using unnamed vector (default) to match LlamaIndex's default behavior
            print("\nCreating new collection with vector configuration...")
            self._client.create_collection(
                collection_name="university_kb",
                vectors_config=models.VectorParams(
                    size=384,  # Match the embedding size of BAAI/bge-small-en-v1.5
                    distance=models.Distance.COSINE
                )
            )
            
            # Create vector store without specifying vector_name (use default)
            vector_store = QdrantVectorStore(
                client=self._client,
                collection_name="university_kb",
                batch_size=32  # Process in batches for better performance
            )
            
            # Load and log documents
            print("\nLoading and parsing documents...")
            documents = self._load_university_data()
            print(f"✓ Loaded {len(documents)} documents from knowledge base")
            
            # Log document types
            doc_types = {}
            for doc in documents:
                doc_type = doc.metadata.get('type', 'unknown')
                doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
            
            print("\nDocument Type Distribution:")
            for doc_type, count in doc_types.items():
                print(f"- {doc_type}: {count} documents")
            
            # Create vector store and index
            print("\nCreating vector store and generating embeddings...")
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            print("Building vector index...")
            
            # Create index from vector store (not from documents directly)
            # This ensures the vector store configuration is properly used
            self._index = VectorStoreIndex.from_documents(
                documents,
                storage_context=storage_context,
                show_progress=True,
                embed_model=self.embeddings  # Explicitly pass the embedding model
            )
            
            # Verify the data was stored
            collection_info = self._client.get_collection("university_kb")
            
            # Get vector count safely
            vectors_count = 0
            if hasattr(collection_info, 'vectors_count'):
                vectors_count = collection_info.vectors_count or 0
            elif hasattr(collection_info, 'points_count'):
                vectors_count = collection_info.points_count or 0
            
            # Alternative: Count directly from Qdrant
            if vectors_count == 0:
                try:
                    count_result = self._client.count(collection_name="university_kb")
                    vectors_count = count_result.count if hasattr(count_result, 'count') else count_result
                except:
                    vectors_count = len(documents)  # Fallback to document count
            
            print(f"\n✓ Knowledge base created successfully!")
            print(f"  - Documents processed: {len(documents)}")
            print(f"  - Vectors stored in Qdrant: {vectors_count}")
            
        except Exception as e:
            print(f"❌ Error creating knowledge base: {e}")
            import traceback
            traceback.print_exc()
            raise

    def create_kb(self, force_recreate: bool = False) -> None:
        """
        Create or recreate the knowledge base.
        
        Args:
            force_recreate: If True, will delete and recreate the knowledge base even if it exists.
        """
        try:
            if force_recreate:
                print("\n=== Force Recreating Knowledge Base ===")
                self._create_kb()
            else:
                # Check if collection exists and has data
                try:
                    collection_info = self._client.get_collection("university_kb")
                    vectors_count = getattr(collection_info, 'vectors_count', 0) or 0
                    if vectors_count > 0:
                        print("\n✓ Using existing knowledge base")
                        print(f"  - Vectors stored: {vectors_count}")
                        return
                    else:
                        print("\nExisting knowledge base is empty, creating a new one...")
                        self._create_kb()
                except Exception as e:
                    if "not found" in str(e):
                        print("\nNo existing knowledge base found, creating a new one...")
                        self._create_kb()
                    else:
                        raise
        except Exception as e:
            print(f"❌ Error in create_kb: {e}")
            import traceback
            traceback.print_exc()
            raise

    def _load_university_data(self) -> List[Document]:
        """Load and process all sections of the university knowledge base with detailed logging."""
        print("\n=== Loading and Parsing Knowledge Base ===")
        try:
            print(f"Loading file: {os.path.abspath(self.knowledge_base_path)}")
            with open(self.knowledge_base_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            documents = []
            
            def log_section(section_name, items):
                count = len(items) if items else 0
                print(f"- Found {count} {section_name}")
                return count
            
            # Process Programs
            programs = data.get('programs', [])
            log_section("programs", programs)
            
            # Create a summary document listing ALL programs (for "how many" queries)
            if programs:
                program_list = []
                for i, program in enumerate(programs, 1):
                    program_list.append(f"{i}. {program.get('name', 'N/A')} ({program.get('degree', 'N/A')}, {program.get('duration', 'N/A')})")
                
                summary_text = f"""Question: How many programs are there?
Answer: There are {len(programs)} academic programs in total.

Question: What are all the programs?
Answer: Here is the complete list of all {len(programs)} programs:
{chr(10).join(program_list)}

Question: List all programs
Answer: The university offers {len(programs)} academic programs:
{chr(10).join(program_list)}

SUMMARY: This university has exactly {len(programs)} academic programs."""
                
                summary_doc = Document(
                    text=summary_text,
                    metadata={
                        "type": "programs_summary",
                        "id": "programs_summary",
                        "name": "All Programs Summary",
                        "count": str(len(programs))
                    }
                )
                documents.append(summary_doc)
            
            # Create individual program documents
            for program in programs:
                try:
                    text = f"""Program: {program.get('name', 'N/A')}
Duration: {program.get('duration', 'N/A')}
Degree: {program.get('degree', 'N/A')}
Description: {program.get('description', 'N/A')}
Eligibility: {program.get('eligibility_summary', 'N/A')}"""
                    doc = Document(
                        text=text,
                        metadata={
                            "type": "program",
                            "id": program.get('id', ''),
                            "name": program.get('name', ''),
                            "degree": program.get('degree', ''),
                            "duration": program.get('duration', '')
                        }
                    )
                    documents.append(doc)
                except Exception as e:
                    print(f"Error processing program {program.get('id', 'unknown')}: {e}")
            
            # Process other sections (courses, faculty, fees, etc.)
            sections = [
                ('courses', 'course'),
                ('faculty', 'faculty'),
                ('admissions', 'admission'),
                ('fees', 'fee'),
                ('scholarships', 'scholarship'),
                ('faqs', 'faq'),
                ('placements', 'placement'),
                ('hostel', 'hostel'),
                ('contacts', 'contact'),
                ('events', 'event'),
                ('policies', 'policy'),
                ('announcements', 'announcement'),
                ('qa_pairs', 'qa_pair')
            ]
            
            # Process each section
            for section_name, doc_type in sections:
                items = data.get(section_name, [])
                if not items:
                    continue
                    
                log_section(section_name, items)
                
                # Create summary document for this section
                item_list = []
                for i, item in enumerate(items, 1):
                    name = item.get('name', item.get('title', 'N/A'))
                    item_list.append(f"{i}. {name}")
                
                # Create query-friendly summary with multiple phrasings
                section_summary = f"""Question: How many {section_name} are there?
Answer: There are {len(items)} {section_name} in total.

Question: What are all the {section_name}?
Answer: Here is the complete list of all {len(items)} {section_name}:
{chr(10).join(item_list)}

Question: List all {section_name}
Answer: The university offers {len(items)} {section_name}:
{chr(10).join(item_list)}

SUMMARY: This university has exactly {len(items)} {section_name}."""
                
                summary_doc = Document(
                    text=section_summary,
                    metadata={
                        "type": f"{doc_type}_summary",
                        "id": f"{section_name}_summary",
                        "name": f"All {section_name.title()} Summary",
                        "count": str(len(items))
                    }
                )
                documents.append(summary_doc)
                
                # Create individual documents
                for item in items:
                    try:
                        # Create a clean text representation
                        text_lines = [f"{k.replace('_', ' ').title()}: {v}" 
                                    for k, v in item.items() 
                                    if v and not k.startswith('_')]
                        
                        doc = Document(
                            text='\n'.join(text_lines),
                            metadata={
                                "type": doc_type,
                                "id": str(item.get('id', '')),
                                "name": str(item.get('name', item.get('title', '')))[:100]
                            }
                        )
                        documents.append(doc)
                    except Exception as e:
                        print(f"Error processing {section_name} item: {e}")
            
            print(f"\n✓ Successfully parsed {len(documents)} documents from knowledge base")
            return documents
            
        except Exception as e:
            print(f"❌ Error loading university data: {e}")
            import traceback
            traceback.print_exc()
            return []
            for faculty in data.get('faculty', []):
                text = f"""Faculty: {faculty.get('name', 'N/A')}
                Department: {faculty.get('department', 'N/A')}
                Designation: {faculty.get('designation', 'N/A')}
                Specialization: {', '.join(faculty.get('specialization', []))}
                Bio: {faculty.get('bio', 'N/A')}"""
                doc = Document(
                    text=text,
                    metadata={
                        "type": "faculty",
                        "id": faculty.get('id', ''),
                        "name": faculty.get('name', ''),
                        "department": faculty.get('department', '')
                    }
                )
                documents.append(doc)
            
            # Process Fees
            for fee in data.get('fees', []):
                text = f"""Fee Type: {fee.get('fee_type', 'N/A')}
                Amount: {fee.get('amount', 'N/A')}
                Program ID: {fee.get('program_id', 'N/A')}
                Payment Deadline: {fee.get('payment_deadline', 'N/A')}"""
                doc = Document(
                    text=text,
                    metadata={
                        "type": "fee",
                        "id": fee.get('id', ''),
                        "program_id": fee.get('program_id', ''),
                        "fee_type": fee.get('fee_type', '')
                    }
                )
                documents.append(doc)
            
            # Process Placements
            if data.get('placements'):
                placement = data['placements'][0]
                text = f"""Placement Statistics:
                Top Recruiters: {', '.join(placement.get('top_recruiters', []))}
                Average Package: {placement.get('average_package', 'N/A')}
                Highest Package: {placement.get('highest_package', 'N/A')}"""
                doc = Document(
                    text=text,
                    metadata={
                        "type": "placement",
                        "id": "placement_stats"
                    }
                )
                documents.append(doc)
            
            print(f"Loaded {len(documents)} documents from knowledge base")
            return documents
            
        except Exception as e:
            print(f"Error loading university data: {e}")
            import traceback
            traceback.print_exc()
            return []

    def _create_chat_engine(self):
        """Create the chat engine with memory."""
        try:
            print("\n=== Creating Chat Engine ===")
            
            # Create vector store (use default vector name to match collection config)
            vector_store = QdrantVectorStore(
                client=self._client,
                collection_name="university_kb"
            )
            
            # Create storage context
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            
            # Create index from existing vector store
            self._index = VectorStoreIndex.from_vector_store(
                vector_store=vector_store,
                storage_context=storage_context
            )
            
            # Create chat engine with memory
            memory = ChatMemoryBuffer.from_defaults(token_limit=1500)
            self._chat_engine = self._index.as_chat_engine(
                chat_mode="context",
                memory=memory,
                system_prompt=self._get_system_prompt(),
                verbose=True,
                similarity_top_k=10,  # Retrieve more documents for better context
                response_mode="compact"  # Faster response mode
            )
            
            # Also create a query engine for direct queries
            self._query_engine = self._index.as_query_engine(
                similarity_top_k=10,  # Retrieve more documents
                verbose=True,
                response_mode="compact"
            )
            
            print("✓ Chat engine created successfully!")
            
        except Exception as e:
            print(f"❌ Error creating chat engine: {e}")
            import traceback
            traceback.print_exc()
            raise

    def _get_system_prompt(self) -> str:
        """Get the system prompt for the chat engine."""
        return """You are a helpful university assistant. Answer questions based ONLY on the provided context.

IMPORTANT RULES:
1. Use ONLY the information from the context provided
2. If asked about counts (how many programs, courses, etc.), count ALL items in the context
3. Be concise and direct - avoid unnecessary elaboration
4. If the context doesn't contain the answer, say "I don't have that information"
5. When listing items, include ALL of them from the context, not just one example

For counting questions: Count every single item mentioned in the context before answering.
        """

    def _get_cache_key(self, text: str) -> str:
        """Generate a cache key for the given text."""
        return md5(text.strip().lower().encode('utf-8')).hexdigest()

    def _handle_fee_query(self, query: str) -> Optional[str]:
        """Handle fee-related queries with direct database lookup."""
        try:
            with open(self.knowledge_base_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract program name from query
            program_name = None
            program_id = None
            
            # Check programs for a match
            for program in data.get('programs', []):
                if program.get('name', '').lower() in query.lower():
                    program_name = program.get('name')
                    program_id = program.get('id')
                    break
            
            if not program_name:
                return None
                
            # Query fees for this program
            fees = []
            for fee in data.get('fees', []):
                if fee.get('program_id') == program_id:
                    fees.append(
                        f"{fee.get('fee_type', 'Fee')}: {fee.get('amount', 'N/A')} "
                        f"(Due: {fee.get('payment_deadline', 'N/A')})"
                    )
            
            if not fees:
                return None
                
            return f"""Here are the fee details for {program_name}:

{chr(10).join(fees)}

For more information, please contact the admissions office."""
            
        except Exception as e:
            print(f"Error in _handle_fee_query: {e}")
            return None

    def _classify_query_intent(self, user_input: str) -> str:
        """Classify the user's query intent to route it appropriately."""
        user_lower = user_input.lower().strip()
        
        # Check for greetings
        greetings = ['hi', 'hello', 'hey', 'greetings', 'good morning', 'good afternoon', 'good evening', 'howdy']
        if any(user_lower == g or user_lower.startswith(g + ' ') for g in greetings):
            return "greeting"
        
        # Check for farewells
        farewells = ['bye', 'goodbye', 'see you', 'take care', 'farewell']
        if any(user_lower == f or user_lower.startswith(f + ' ') for f in farewells):
            return "farewell"
        
        # Check for thanks
        thanks = ['thank you', 'thanks', 'appreciate', 'grateful']
        if any(t in user_lower for t in thanks):
            return "thanks"
        
        # Check for university-specific keywords
        university_keywords = [
            'program', 'course', 'faculty', 'professor', 'admission', 'fee', 'tuition',
            'scholarship', 'event', 'announcement', 'degree', 'b.tech', 'mba', 'phd',
            'department', 'semester', 'exam', 'hostel', 'placement', 'campus'
        ]
        if any(keyword in user_lower for keyword in university_keywords):
            return "university"
        
        # Check for "how many", "list all", "what are" - likely university queries
        counting_phrases = ['how many', 'list all', 'what are all', 'show me all', 'tell me about']
        if any(phrase in user_lower for phrase in counting_phrases):
            return "university"
        
        # Default to general conversation
        return "general"
    
    def _handle_general_conversation(self, user_input: str) -> str:
        """Handle general conversation using LLM without RAG."""
        try:
            # Use LLM directly for natural conversation
            prompt = f"""You are a friendly university assistant. Respond naturally to this message.
Keep your response conversational and helpful. If appropriate, mention that you can help with university information.

User: {user_input}

Assistant:"""
            
            response = self.llm.complete(prompt)
            return response.text.strip()
            
        except Exception as e:
            print(f"Error in general conversation: {e}")
            return "I'm here to help! Feel free to ask me about university programs, courses, faculty, or admissions."
    
    def interact_with_llm(self, user_input: str) -> str:
        """Interact with the LLM using smart routing (RAG for university queries, LLM for general chat)."""
        self.total_queries += 1
        
        if not user_input.strip():
            return "Please provide a valid question or query."
        
        # Classify the query intent
        intent = self._classify_query_intent(user_input)
        
        # Route based on intent
        if intent in ["greeting", "farewell", "thanks", "general"]:
            # Use LLM directly for general conversation
            return self._handle_general_conversation(user_input)
        
        # For university queries, use RAG
        cache_key = self._get_cache_key(user_input)
        if cache_key in self.response_cache:
            self.cache_hits += 1
            return self.response_cache[cache_key]
        
        try:
            # Use RAG for university-specific questions
            response = self._chat_engine.chat(user_input).response
            self.response_cache[cache_key] = response
            return response
            
        except Exception as e:
            print(f"Error in RAG query: {e}")
            import traceback
            traceback.print_exc()
            return "I apologize, but I encountered an error. Please try rephrasing your question."

if __name__ == "__main__":
    try:
        print("Initializing AI Voice Assistant...")
        assistant = AIVoiceAssistant("voice_rag_kb.json")
        print("\n" + "="*50)
        print("AI Voice Assistant is ready!")
        print("Type your questions or type 'exit' to quit.")
        print("Type 'reload' to recreate the knowledge base.")
        print("="*50 + "\n")
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                # Handle exit command
                if user_input.lower() in ['exit', 'quit', 'bye']:
                    print("\nGoodbye!")
                    break
                    
                # Handle reload command
                if user_input.lower() == 'reload':
                    print("\nRecreating knowledge base...")
                    assistant.create_kb(force_recreate=True)
                    print("Knowledge base has been recreated. You can continue asking questions.")
                    continue
                
                # Process the user input
                if user_input:  # Only process non-empty input
                    response = assistant.interact_with_llm(user_input)
                    print(f"\nAssistant: {response}")
                    
            except KeyboardInterrupt:
                print("\n\nOperation cancelled by user.")
                break
                
            except Exception as e:
                print(f"\nAn error occurred: {str(e)}")
                print("Please try again or type 'exit' to quit.")
    
    except Exception as e:
        print(f"\nFailed to initialize the AI Voice Assistant: {str(e)}")
        print("Please check if all required services (Qdrant, Ollama) are running.")
        if "ConnectionError" in str(type(e).__name__):
            print("Make sure Qdrant is running at http://localhost:6333")