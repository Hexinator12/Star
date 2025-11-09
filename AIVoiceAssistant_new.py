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
    def __init__(self, knowledge_base_path="university_dataset_advanced.json"):
        self.response_cache = {}
        self.llm_cache = {}  # Separate cache for LLM responses
        self.cache_hits = 0
        self.llm_cache_hits = 0
        self.total_queries = 0
        self.knowledge_base_path = knowledge_base_path
        self.conversation_context = []  # Store recent conversation for context
        
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
                    # Get actual point count
                    count_result = self._client.count(collection.name)
                    vector_count = count_result.count if hasattr(count_result, 'count') else 0
                    print(f"- {collection.name} (Vectors: {vector_count:,})")
                except Exception as e:
                    print(f"- {collection.name} (Error: {str(e)[:50]})")
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

            # Initialize LLM with optimized settings
            print("Initializing LLM with optimized settings...")
            # Using gemma:2b - faster and more accurate than phi for RAG tasks
            self.llm = Ollama(
                model="gemma:2b",
                base_url="http://localhost:11434",
                request_timeout=60.0,  # Reduced timeout - gemma is faster
                temperature=0.1,  # Lower temperature for more focused responses
                num_ctx=3072,     # Optimized context window for speed
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
                # Check if collection exists
                try:
                    collection_info = self._client.get_collection("university_kb")
                    # Collection exists - use it (don't check vector count, trust it exists)
                    print("\n✓ Using existing knowledge base from Qdrant")
                    print(f"  - Collection: university_kb")
                    # Try to get vector count, but don't fail if it's None
                    try:
                        vectors_count = getattr(collection_info, 'vectors_count', None)
                        if vectors_count is not None:
                            print(f"  - Vectors: {vectors_count}")
                    except:
                        pass
                    return
                except Exception as e:
                    if "not found" in str(e).lower():
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
                similarity_top_k=3,  # Reduced to 3 for much faster processing
                response_mode="tree_summarize"  # Fastest response mode
            )
            
            # Also create a query engine for direct queries
            self._query_engine = self._index.as_query_engine(
                similarity_top_k=3,  # Reduced to 3 for faster processing
                verbose=True,
                response_mode="tree_summarize"  # Fastest response mode
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
    
    def _is_simple_query(self, query: str) -> tuple:
        """Check query type and return (query_type, entity_type)."""
        query_lower = query.lower()
        
        # List queries
        if any(kw in query_lower for kw in ["list", "what programs", "what courses", "available programs", "available courses", "all programs", "all courses"]):
            if "program" in query_lower:
                return ("list", "program")
            elif "course" in query_lower:
                return ("list", "course")
        
        # Count queries
        if any(kw in query_lower for kw in ["how many", "number of", "count"]):
            if "program" in query_lower:
                return ("count", "program")
            elif "course" in query_lower:
                return ("count", "course")
        
        # Recommendation queries
        if any(kw in query_lower for kw in ["which program", "what program", "best program", "recommend", "should i choose", "which course", "best for"]):
            return ("recommend", "program")
        
        # Specific program details queries
        if any(kw in query_lower for kw in ["tell me about", "details about", "information about", "what is", "describe"]):
            # Check if a specific program is mentioned
            for prog_keyword in ["b.tech", "m.tech", "mba", "b.sc", "m.sc", "b.des", "ph.d", "phd"]:
                if prog_keyword in query_lower:
                    return ("program_details", "program")
        
        # Fees queries (more patterns)
        if any(kw in query_lower for kw in ["fee", "fees", "cost", "tuition", "price", "how much", "expensive", "afford"]):
            # Only if not asking about specific details that need LLM
            if not any(specific in query_lower for specific in ["compare", "difference", "why"]):
                return ("fees", "program")
        
        # Eligibility queries (more patterns)
        if any(kw in query_lower for kw in ["eligibility", "eligible", "requirement", "requirements", "qualify", "admission criteria", "need to"]):
            if not any(specific in query_lower for specific in ["compare", "difference", "why"]):
                return ("eligibility", "program")
        
        # Duration queries (more patterns)
        if any(kw in query_lower for kw in ["duration", "how long", "years", "semesters", "time to complete"]):
            return ("duration", "program")
        
        # Contact/Application queries (more patterns)
        if any(kw in query_lower for kw in ["contact", "phone", "email", "apply", "application", "how to apply", "how do i apply", "admission process"]):
            return ("contact", "general")
        
        return (None, None)
    
    def _get_all_from_qdrant(self, entity_type: str) -> list:
        """Get all entities of a type directly from Qdrant (FAST)."""
        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            
            print(f"📊 Fetching all '{entity_type}' from Qdrant...")
            
            # Scroll through all points with the given type
            results = self._client.scroll(
                collection_name="university_kb",
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="type",  # Changed from "metadata.type"
                            match=MatchValue(value=entity_type)
                        )
                    ]
                ),
                limit=100,
                with_payload=True,
                with_vectors=False  # Don't need vectors, just metadata
            )
            
            print(f"📊 Retrieved {len(results[0])} points from Qdrant")
            
            entities = set()
            for point in results[0]:
                if point.payload:
                    # Get name from payload (top level)
                    name = point.payload.get("name", "")
                    
                    # If name is empty, try to extract from _node_content text
                    if not name and "_node_content" in point.payload:
                        try:
                            import json as json_lib
                            node_data = json_lib.loads(point.payload["_node_content"])
                            text = node_data.get("text", "")
                            # Extract program name from text (format: "Program: NAME")
                            if "Program:" in text:
                                lines = text.split("\n")
                                for line in lines:
                                    if line.startswith("Program:"):
                                        name = line.replace("Program:", "").strip()
                                        break
                        except:
                            pass
                    
                    if name:
                        entities.add(name)
                        
            print(f"✓ Found {len(entities)} unique {entity_type}s")
            return sorted(list(entities))
            
        except Exception as e:
            print(f"❌ Error getting entities from Qdrant: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _recommend_program(self, query: str) -> str:
        """Recommend programs based on career field/interest (INSTANT)."""
        try:
            query_lower = query.lower()
            
            # Define career field mappings
            field_keywords = {
                "it": ["it", "information technology", "software", "programming", "coding", "developer"],
                "ai": ["ai", "artificial intelligence", "machine learning", "ml", "data science", "deep learning"],
                "engineering": ["engineering", "engineer", "mechanical", "civil", "electrical"],
                "design": ["design", "creative", "ui", "ux", "graphics", "art"],
                "business": ["business", "management", "mba", "finance", "marketing", "entrepreneur"],
                "data": ["data", "analytics", "data science", "big data", "statistics"],
                "biotech": ["biotech", "biology", "life science", "pharmaceutical"],
                "robotics": ["robot", "robotics", "automation", "mechatronics"],
            }
            
            # Detect field
            detected_field = None
            for field, keywords in field_keywords.items():
                if any(kw in query_lower for kw in keywords):
                    detected_field = field
                    break
            
            # Get all programs
            all_programs = self._get_all_from_qdrant("program")
            
            if not all_programs:
                return None
            
            # Filter relevant programs
            recommended = []
            
            if detected_field == "it":
                recommended = [p for p in all_programs if any(kw in p.lower() for kw in ["computer", "software", "it", "information technology", "tech"])]
            elif detected_field == "ai":
                recommended = [p for p in all_programs if any(kw in p.lower() for kw in ["ai", "artificial intelligence", "data science", "machine learning"])]
            elif detected_field == "engineering":
                recommended = [p for p in all_programs if "engineering" in p.lower() or "b.tech" in p.lower()]
            elif detected_field == "design":
                recommended = [p for p in all_programs if "design" in p.lower() or "b.des" in p.lower()]
            elif detected_field == "business":
                recommended = [p for p in all_programs if any(kw in p.lower() for kw in ["mba", "business", "management", "finance", "marketing"])]
            elif detected_field == "data":
                recommended = [p for p in all_programs if any(kw in p.lower() for kw in ["data", "analytics", "statistics"])]
            elif detected_field == "biotech":
                recommended = [p for p in all_programs if "biotech" in p.lower() or "biology" in p.lower()]
            elif detected_field == "robotics":
                recommended = [p for p in all_programs if "robot" in p.lower() or "mechatronics" in p.lower()]
            
            # Build response
            if recommended:
                response = f"Great question! For a career in {detected_field.upper()}, I recommend these programs:\n\n"
                for i, prog in enumerate(recommended[:5], 1):  # Top 5
                    response += f"{i}. {prog}\n"
                response += f"\nThese programs will give you the skills and knowledge needed for {detected_field}. Would you like to know more about any of these programs?"
                return response
            else:
                # No specific field detected, give general guidance
                response = "I can help you choose the right program! We offer programs in:\n\n"
                response += "• Computer Science & IT (B.Tech CS, B.Sc CS, etc.)\n"
                response += "• Artificial Intelligence & Data Science\n"
                response += "• Engineering (Mechanical, Civil, etc.)\n"
                response += "• Design (B.Des)\n"
                response += "• Business & Management (MBA)\n"
                response += "• Biotechnology\n\n"
                response += "What field are you interested in? For example, you can ask:\n"
                response += "- 'Which program is best for IT?'\n"
                response += "- 'I want to work in AI, which program?'\n"
                response += "- 'Best program for business career?'"
                return response
                
        except Exception as e:
            print(f"Error in program recommendation: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _get_program_details(self, query: str) -> str:
        """Get specific program details directly from Qdrant (FAST)."""
        try:
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            
            query_lower = query.lower()
            
            # First, get all programs to check what exists
            all_programs = self._get_all_from_qdrant("program")
            
            # Try to find exact match
            matched_program = None
            for prog in all_programs:
                prog_lower = prog.lower()
                # Check if query mentions this program
                if prog_lower in query_lower or any(word in prog_lower for word in query_lower.split() if len(word) > 3):
                    matched_program = prog
                    break
            
            # If no exact match, try fuzzy matching for common queries
            if not matched_program:
                # Handle "B.Tech AI" → suggest "B.Tech Data Science" or "B.Des AI"
                if "b.tech" in query_lower and ("ai" in query_lower or "artificial" in query_lower):
                    # Find AI-related B.Tech programs
                    ai_programs = [p for p in all_programs if "b.tech" in p.lower() and ("data" in p.lower() or "ai" in p.lower())]
                    if ai_programs:
                        matched_program = ai_programs[0]
                    else:
                        # Suggest alternatives
                        alternatives = [p for p in all_programs if "ai" in p.lower() or "artificial" in p.lower()]
                        if alternatives:
                            response = f"We don't have a 'B.Tech AI' program, but we offer these AI-related programs:\n\n"
                            for i, prog in enumerate(alternatives, 1):
                                response += f"{i}. {prog}\n"
                            response += f"\nWould you like to know more about any of these?"
                            return response
            
            if not matched_program:
                return None  # Fall back to RAG
            
            # Search for this specific program in Qdrant
            results = self._client.scroll(
                collection_name="university_kb",
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="type",
                            match=MatchValue(value="program")
                        )
                    ]
                ),
                limit=100,
                with_payload=True,
                with_vectors=False
            )
            
            # Find matching program in Qdrant data
            for point in results[0]:
                if point.payload and "_node_content" in point.payload:
                    try:
                        import json as json_lib
                        node_data = json_lib.loads(point.payload["_node_content"])
                        text = node_data.get("text", "")
                        
                        # Check if this is the program we're looking for
                        if matched_program.lower() in text.lower():
                            # Extract key information
                            lines = text.split("\n")
                            info = {}
                            for line in lines:
                                if ":" in line:
                                    key, value = line.split(":", 1)
                                    info[key.strip()] = value.strip()
                            
                            # Build response
                            response = f"Here's information about {matched_program}:\n\n"
                            
                            key_fields = ["Program", "Degree", "Duration", "Eligibility", "Annual Tuition", "Description"]
                            for field in key_fields:
                                if field in info:
                                    response += f"• {field}: {info[field]}\n"
                            
                            response += "\nWould you like to know more about fees, eligibility, or placements?"
                            return response
                    except:
                        continue
            
            return None  # Fall back to RAG if not found
            
        except Exception as e:
            print(f"Error getting program details: {e}")
            return None
    
    def _handle_fast_query(self, query_type: str, entity_type: str, original_query: str) -> str:
        """Handle simple queries with direct Qdrant access (INSTANT)."""
        try:
            # Program details query
            if query_type == "program_details":
                fast_response = self._get_program_details(original_query)
                if fast_response:
                    return fast_response
            
            # Recommendation query
            if query_type == "recommend":
                return self._recommend_program(original_query)
            
            # Fees query
            if query_type == "fees":
                return "The annual tuition fees vary by program:\n\n• B.Tech/B.Sc programs: ₹90,000 - ₹1,50,000\n• M.Tech/M.Sc programs: ₹1,00,000 - ₹1,80,000\n• MBA programs: ₹1,50,000 - ₹2,20,000\n• B.Des programs: ₹1,20,000 - ₹1,80,000\n• Ph.D programs: ₹80,000 - ₹1,20,000\n\nAdditional costs may include hostel, exam, and lab fees. For specific program fees, please ask about a particular program."
            
            # Eligibility query
            if query_type == "eligibility":
                return "General eligibility criteria:\n\n• B.Tech/B.Sc: 10+2 with relevant subjects (typically 50%+ marks)\n• M.Tech/M.Sc: Bachelor's degree in relevant field\n• MBA: Bachelor's degree in any discipline + entrance exam\n• B.Des: 10+2 with creative aptitude\n• Ph.D: Master's degree in relevant field\n\nFor specific program eligibility, please ask about a particular program (e.g., 'What is the eligibility for B.Tech Data Science?')"
            
            # Duration query
            if query_type == "duration":
                return "Program durations:\n\n• B.Tech/B.Sc/B.Des: 4 years (8 semesters)\n• M.Tech/M.Sc/MBA: 2 years (4 semesters)\n• Ph.D: 3-5 years (research-based)\n\nFor a specific program duration, please ask about that program."
            
            # Contact/Application query
            if query_type == "contact":
                return "📞 Contact Information:\n\n• Admissions Office: +91-XXX-XXXX-XXX\n• Email: admissions@university.edu\n• Website: www.university.edu\n• Address: University Campus, City, State\n\n📝 How to Apply:\n1. Visit our website\n2. Fill out the online application form\n3. Upload required documents\n4. Pay application fee\n5. Attend entrance exam (if applicable)\n6. Wait for admission decision\n\nFor specific program applications, please mention the program name."
            
            entities = self._get_all_from_qdrant(entity_type)
            
            if not entities:
                return None  # Fall back to RAG
            
            # Count query
            if query_type == "count":
                entity_name = "programs" if entity_type == "program" else "courses"
                return f"There are {len(entities)} {entity_name} in total."
            
            # List query
            elif query_type == "list":
                entity_name = "programs" if entity_type == "program" else "courses"
                
                # Limit courses to 30 for readability
                display_entities = entities[:30] if entity_type == "course" else entities
                
                response = f"We offer {len(entities)} {entity_name}" + (f" (showing first {len(display_entities)})" if len(display_entities) < len(entities) else "") + ":\n\n"
                
                for i, entity in enumerate(display_entities, 1):
                    response += f"{i}. {entity}\n"
                
                if len(display_entities) < len(entities):
                    response += f"\n...and {len(entities) - len(display_entities)} more.\n"
                
                response += f"\nWould you like to know more about any specific {entity_type}?"
                return response
            
            return None
            
        except Exception as e:
            print(f"Error in fast query handler: {e}")
            import traceback
            traceback.print_exc()
            return None
    
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
        
        # Check for simple queries - handle with direct Qdrant access (INSTANT)
        query_type, entity_type = self._is_simple_query(user_input)
        if query_type and entity_type:
            print(f"🚀 Fast path: {query_type} query for {entity_type}")
            fast_response = self._handle_fast_query(query_type, entity_type, user_input)
            if fast_response:
                # Store in conversation context
                self.conversation_context.append({"query": user_input, "response": fast_response})
                if len(self.conversation_context) > 5:  # Keep last 5 exchanges
                    self.conversation_context.pop(0)
                return fast_response
        
        # For university queries, use RAG
        cache_key = self._get_cache_key(user_input)
        
        # Check LLM cache first
        if cache_key in self.llm_cache:
            self.llm_cache_hits += 1
            print(f"💾 Cache hit! (LLM cache: {self.llm_cache_hits}/{self.total_queries})")
            return self.llm_cache[cache_key]
        
        # Check old response cache
        if cache_key in self.response_cache:
            self.cache_hits += 1
            return self.response_cache[cache_key]
        
        try:
            print(f"🤖 Using LLM for complex query...")
            
            # Add conversation context to query if available
            enhanced_query = user_input
            if self.conversation_context:
                last_exchange = self.conversation_context[-1]
                # If query is short and vague, add context
                if len(user_input.split()) < 5:
                    enhanced_query = f"Previous context: {last_exchange['query'][:100]}\nCurrent question: {user_input}"
            
            # Use RAG for university-specific questions
            # TODO: Implement streaming in future for better UX
            # For now, using standard chat (non-streaming)
            response = self._chat_engine.chat(enhanced_query).response
            
            # Store in LLM cache
            self.llm_cache[cache_key] = response
            self.response_cache[cache_key] = response
            
            # Store in conversation context
            self.conversation_context.append({"query": user_input, "response": response})
            if len(self.conversation_context) > 5:
                self.conversation_context.pop(0)
            
            print(f"✓ LLM response cached for future queries")
            return response
            
        except Exception as e:
            error_msg = str(e)
            print(f"Error in RAG query: {e}")
            import traceback
            traceback.print_exc()
            
            # Provide more specific error messages
            if "timeout" in error_msg.lower() or "ReadTimeout" in error_msg:
                return "The request took too long to process. This might be due to a large query. Please try asking a more specific question or wait a moment and try again."
            elif "connection" in error_msg.lower():
                return "Unable to connect to the AI model. Please ensure Ollama is running (run 'ollama serve' in a terminal)."
            else:
                return "I apologize, but I encountered an error. Please try rephrasing your question or asking something more specific."

if __name__ == "__main__":
    try:
        print("Initializing AI Voice Assistant...")
        assistant = AIVoiceAssistant("university_dataset_advanced.json")
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