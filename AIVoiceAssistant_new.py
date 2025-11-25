# Configure environment for optimal performance
import os
os.environ['OMP_NUM_THREADS'] = '8'  # Match your CPU core count
os.environ['OMP_WAIT_POLICY'] = 'PASSIVE'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # Disable tokenizer parallelism
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging
os.environ['KMP_AFFINITY'] = 'noverbose'

# Standard library imports
import time
import json
import hashlib
import re
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import lru_cache
from collections import defaultdict
import numpy as np

# Suppress warnings
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

@dataclass
class CacheEntry:
    response: Any
    timestamp: float
    ttl: int = 3600  # 1 hour default TTL

from qdrant_client import QdrantClient, models
from llama_index.llms.ollama import Ollama
from llama_index.core import Document, Settings
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import StorageContext, QueryBundle
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import NodeWithScore, QueryType
from llama_index.core.postprocessor import SentenceTransformerRerank
from typing import Optional, List, Dict, Any, Tuple
import json
import os
import warnings
from hashlib import md5
from functools import lru_cache
import re
from collections import defaultdict
import numpy as np

# Suppress warnings
warnings.filterwarnings("ignore")

class AIVoiceAssistant:
    def __init__(self, knowledge_base_path="university_dataset_advanced.json"):
        # Initialize caches with TTL support
        self.response_cache: Dict[str, CacheEntry] = {}
        self.llm_cache: Dict[str, CacheEntry] = {}
        self.cache_hits = 0
        self.llm_cache_hits = 0
        self.total_queries = 0
        self.knowledge_base_path = knowledge_base_path
        
        # Initialize program cache
        self._init_program_cache()
        
        # Enhanced conversation memory with optimized structure
        self.conversation_history = []  # Complete conversation history
        self.conversation_context = []  # Active context window (last 5 exchanges)
        self.conversation_summary = ""  # Summary of the conversation
        self.max_context_tokens = 2000  # Maximum tokens for conversation context
        
        # User preferences with optimized defaults
        self.user_preferences = {
            'preferred_programs': set(),
            'interests': set(),
            'preferred_response_style': 'concise',  # 'concise' or 'detailed'
            'last_updated': time.time()
        }
        self.conversation_summary = ""  # Summary of the conversation so far
        self.max_context_tokens = 2000  # Maximum tokens for conversation context
        
        # Initialize Qdrant client
        self._init_qdrant()
        
        # Initialize models and embeddings
        self._init_models()
        
        # Create knowledge base if it doesn't exist
        self.create_kb()
        
        # Create chat engine
        self._create_chat_engine()

    def _is_ollama_running(self) -> bool:
        """Check if Ollama server is running and accessible."""
        import requests
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False

    def _init_models(self):
        """Initialize LLM, embedding models, and re-ranker with better error handling and timeouts."""
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
                    time.sleep(5)
                except Exception as e:
                    print(f"Failed to start Ollama server: {e}")
                    print("Please start Ollama manually in a separate terminal with: ollama serve")
                    raise

            # Initialize LLM with optimized settings
            print("Initializing LLM with optimized settings...")
            self.llm = Ollama(
                model="gemma:2b",
                base_url="http://localhost:11434",
                request_timeout=30.0,  # Reduced from 60s
                temperature=0.2,  # Slightly higher for better creativity
                num_ctx=2048,    # Reduced context window for faster processing
                num_gpu_layers=0,  # Force CPU for consistent performance
                num_thread=4,     # Match OMP_NUM_THREADS
                repeat_last_n=64,  # Reduce memory usage
                repeat_penalty=1.1,  # Slightly reduce repetition
                top_k=40,        # Faster sampling
                top_p=0.9,       # Smarter sampling
            )
            
            # Initialize embedding model with optimized settings
            print("Initializing Embedding Model with optimized settings...")
            self.embeddings = HuggingFaceEmbedding(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                device="cpu",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={
                    'batch_size': 32,
                    'convert_to_numpy': True,
                    'normalize_embeddings': True
                }
            )
            
            # Test the embedding model
            test_embedding = self.embeddings.get_text_embedding("test")
            print(f"✓ Embedding model initialized successfully (dim={len(test_embedding)})")
            
        except Exception as e:
            print(f"❌ Error initializing models: {e}")
            import traceback
            traceback.print_exc()
            raise

    def _init_qdrant(self):
        """Initialize Qdrant client with optimized settings."""
        try:
            print("\n=== Initializing Qdrant Client with Optimized Settings ===")
            
            # First try to connect with default settings
            self._client = QdrantClient(
                url="http://localhost:6333",
                prefer_grpc=False,  # HTTP is often faster for local development
                timeout=10.0,       # Increased timeout for initial connection
                api_key=None,       # No API key for local development
                https=False         # Use HTTP for local development
            )
            
            # Test the connection with a simple operation
            try:
                collections = self._client.get_collections()
                print("\n=== Qdrant Collections ===")
                for collection in collections.collections:
                    try:
                        info = self._client.get_collection(collection.name)
                        count_result = self._client.count(collection.name)
                        vector_count = count_result.count if hasattr(count_result, 'count') else 0
                        print(f"- {collection.name} (Vectors: {vector_count:,})")
                    except Exception as e:
                        print(f"- {collection.name} (Error: {str(e)[:50]})")
                
                print("\n✓ Qdrant client initialized successfully")
                return
                
            except Exception as e:
                print(f"❌ Failed to connect to Qdrant: {e}")
                print("\nTroubleshooting steps:")
                print("1. Make sure Qdrant is running: docker run -p 6333:6333 qdrant/qdrant")
                print("2. Check if port 6333 is available")
                print("3. Verify Qdrant is accessible at http://localhost:6333")
                raise
            
        except Exception as e:
            print(f"❌ Failed to connect to Qdrant: {e}")
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
        """Initialize LLM, embedding models, and re-ranker with better error handling and timeouts."""
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
            
            # Initialize embedding model with CPU
            print("Initializing Embedding Model (BAAI/bge-small-en-v1.5) on CPU...")
            try:
                # Ensure no GPU is used
                os.environ['CUDA_VISIBLE_DEVICES'] = ''
                
                # Initialize with minimal configuration
                self.embeddings = HuggingFaceEmbedding(
                    model_name="sentence-transformers/all-MiniLM-L6-v2",  # More compatible model
                    device="cpu"
                )
                # Test the embedding model
                test_embedding = self.embeddings.get_text_embedding("test")
                print(f"✓ Embedding model initialized successfully (dim={len(test_embedding)})")
                
            except Exception as e:
                print(f"❌ Error initializing embedding model: {e}")
                import traceback
                traceback.print_exc()
                raise
            
            # Initialize re-ranker (will be loaded on first use)
            self._reranker = None
            
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

    def _expand_query(self, query: str) -> Tuple[str, Dict[str, List[str]]]:
        """Expand the query with synonyms and related terms using LLM."""
        # Check cache first
        cache_key = f"query_expand:{query.lower().strip()}"
        if cache_key in self.llm_cache:
            return self.llm_cache[cache_key]
            
        # Generate query variations using LLM
        prompt = f"""Given the user query, generate 2-3 alternative phrasings or expansions that might help retrieve more relevant documents.
        Focus on academic and university-related terminology.
        
        Query: {query}
        
        Respond in this exact format (no extra text):
        - [Alternative 1]
        - [Alternative 2]
        - [Alternative 3]"""
        
        try:
            response = self.llm.complete(prompt)
            alternatives = [q.strip('- ').strip() for q in response.text.split('\n') if q.strip()]
            
            # Combine original query with alternatives
            expanded_queries = [query] + alternatives
            
            # Extract key terms for BM25
            terms = set()
            for q in expanded_queries:
                terms.update(re.findall(r'\b\w+\b', q.lower()))
            
            result = (" OR ".join(f'({q})' for q in expanded_queries), {"bm25_terms": list(terms)})
            self.llm_cache[cache_key] = result
            return result
            
        except Exception as e:
            print(f"Warning: Query expansion failed: {e}")
            return query, {"bm25_terms": re.findall(r'\b\w+\b', query.lower())}

    def _rerank_documents(self, query: str, nodes: List[NodeWithScore]) -> List[NodeWithScore]:
        """Re-rank retrieved documents using cross-encoder and diversity."""
        if not nodes:
            return nodes
            
        # Initialize re-ranker if not already done
        if not hasattr(self, '_reranker'):
            try:
                # Initialize reranker with minimal configuration
                os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Ensure no GPU is used
                from sentence_transformers import CrossEncoder
                try:
                    # Use a simpler model that's more likely to work
                    cross_encoder = CrossEncoder('cross-encoder/ms-marco-TinyBERT-L-2-v2')
                    self._reranker = SentenceTransformerRerank(
                        model=cross_encoder,
                        top_n=5
                    )
                except Exception as e:
                    print(f"Warning: Could not initialize CrossEncoder: {e}")
                    # Disable reranking if initialization fails
                    self._reranker = None
            except Exception as e:
                print(f"Warning: Could not load re-ranker: {e}")
                return nodes
        
        try:
            # Convert nodes to format expected by re-ranker
            query_bundle = QueryBundle(query_str=query)
            
            # Apply re-ranking
            reranked_nodes = self._reranker.postprocess_nodes(
                nodes,
                query_bundle
            )
            
            # Apply diversity: avoid very similar documents
            unique_docs = []
            seen_content = set()
            
            for node in reranked_nodes:
                # Simple content-based deduplication
                content_hash = md5(node.node.get_content().encode()).hexdigest()
                if content_hash not in seen_content:
                    seen_content.add(content_hash)
                    unique_docs.append(node)
                
                if len(unique_docs) >= 5:  # Limit to top 5 diverse results
                    break
                    
            return unique_docs
            
        except Exception as e:
            print(f"Warning: Re-ranking failed: {e}")
            return nodes[:5]  # Fallback to top 5

    def _create_chat_engine(self):
        """Create the chat engine with enhanced retrieval and re-ranking."""
        try:
            print("\n=== Creating Enhanced Chat Engine ===")
            
            # Create vector store with hybrid search support
            vector_store = QdrantVectorStore(
                client=self._client,
                collection_name="university_kb",
                enable_hybrid=True,  # Enable hybrid search (BM25 + vector)
                batch_size=32
            )
            
            # Create storage context
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            
            # Create index from existing vector store
            self._index = VectorStoreIndex.from_vector_store(
                vector_store=vector_store,
                storage_context=storage_context
            )
            
            # Create a custom retriever with enhanced settings
            class EnhancedRetriever(VectorIndexRetriever):
                def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
                    # Expand the query
                    expanded_query, metadata = self._index._expand_query(query_bundle.query_str)
                    
                    # Perform hybrid search
                    vector_retriever = VectorIndexRetriever(
                        index=self._index,
                        similarity_top_k=10,  # Retrieve more for re-ranking
                        vector_store_query_mode="hybrid",
                        alpha=0.7  # Weight for vector vs BM25 (0.7 = 70% vector, 30% BM25)
                    )
                    
                    # Get initial results
                    nodes = vector_retriever.retrieve(expanded_query)
                    
                    # Apply re-ranking
                    return self._index._rerank_documents(query_bundle.query_str, nodes)
            
            # Create the enhanced retriever
            retriever = EnhancedRetriever(
                index=self._index,
                similarity_top_k=5,  # Final number of results after re-ranking
                vector_store_query_mode="hybrid",
                alpha=0.7
            )
            
            # Create chat engine with memory and enhanced retrieval
            memory = ChatMemoryBuffer.from_defaults(token_limit=2000)  # Increased for better context
            
            self._chat_engine = RetrieverQueryEngine.from_args(
                retriever=retriever,
                llm=self.llm,
                memory=memory,
                system_prompt=self._get_system_prompt(),
                verbose=True,
                response_mode="tree_summarize"
            )
            
            # Also create a query engine for direct queries
            self._query_engine = RetrieverQueryEngine.from_args(
                retriever=retriever,
                llm=self.llm,
                system_prompt=self._get_system_prompt(),
                verbose=True,
                response_mode="tree_summarize"
            )
            
            # Add the retriever and re-ranker to the instance for later use
            self.retriever = retriever
            
            print("✓ Enhanced chat engine created successfully!")
            
        except Exception as e:
            print(f"❌ Error creating enhanced chat engine: {e}")
            import traceback
            traceback.print_exc()
            raise

    def _update_conversation_context(self, query: str, response: str) -> None:
        """Update conversation context with the latest exchange."""
        # Add to full history
        self.conversation_history.append({
            'query': query,
            'response': response,
            'timestamp': time.time()
        })
        
        # Add to active context window
        self.conversation_context.append({
            'role': 'user',
            'content': query
        })
        self.conversation_context.append({
            'role': 'assistant',
            'content': response
        })
        
        # Update conversation summary periodically
        if len(self.conversation_history) % 3 == 0:  # Update summary every 3 exchanges
            self._update_conversation_summary()
            
        # Trim context if it's getting too long
        self._trim_conversation_context()
    
    def _update_conversation_summary(self) -> None:
        """Generate a summary of the conversation so far."""
        if not self.conversation_history:
            return
            
        # Use the last few exchanges to update the summary
        recent_exchanges = '\n'.join(
            f"User: {ex['query']}\nAssistant: {ex['response']}"
            for ex in self.conversation_history[-3:]  # Last 3 exchanges
        )
        
        prompt = f"""Summarize the key points from this conversation in 2-3 sentences.
        Focus on user preferences, important details, and the main topics discussed.
        
        Conversation:
        {recent_exchanges}
        
        Summary:"""
        
        try:
            summary = self.llm.complete(prompt).text.strip()
            self.conversation_summary = summary
        except Exception as e:
            print(f"Warning: Failed to update conversation summary: {e}")
    
    def _trim_conversation_context(self) -> None:
        """Trim the conversation context to stay within token limits."""
        # Simple token estimation (1 token ~= 4 chars in English)
        def count_tokens(text):
            return len(str(text).encode('utf-8')) // 4
            
        total_tokens = sum(count_tokens(msg['content']) for msg in self.conversation_context)
        
        # Remove oldest messages if over limit, but keep at least the last exchange
        while total_tokens > self.max_context_tokens and len(self.conversation_context) > 2:
            removed = self.conversation_context.pop(0)
            total_tokens -= count_tokens(removed['content'])
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for the chat engine with enhanced instructions."""
        # Add conversation summary and preferences to the system prompt
        context_parts = []
        
        if self.conversation_summary:
            context_parts.append(f"CONVERSATION SUMMARY: {self.conversation_summary}")
            
        if self.user_preferences['preferred_programs']:
            programs = ', '.join(sorted(self.user_preferences['preferred_programs']))
            context_parts.append(f"USER'S PREFERRED PROGRAMS: {programs}")
            
        if self.user_preferences['interests']:
            interests = ', '.join(sorted(self.user_preferences['interests']))
            context_parts.append(f"USER'S INTERESTS: {interests}")
        
        context_str = '\n'.join(context_parts)
        
        return f"""You are an intelligent university assistant. Your role is to provide accurate, helpful, and context-aware responses to students and visitors.
        
{context_str}

IMPORTANT CONTEXT: Use the above information to provide personalized responses.

# CORE INSTRUCTIONS:
1. Always be polite, professional, and student-focused in your responses.
2. If you need clarification, ask specific follow-up questions to better understand the user's needs.
3. When providing information, be as specific and detailed as possible while remaining concise.
4. If you're unsure about something, it's okay to say so and guide the user to the right resource.

# HANDLING GENERAL QUERIES:
- For broad questions (e.g., "Tell me about programs"), first ask clarifying questions to narrow down the scope.
- When listing items, provide a clear structure and organize information logically.
- If multiple programs/courses match the query, list the top 3 most relevant ones first, then mention there are more options available.

# RESPONSE FORMATTING:
- Use clear section headers when appropriate
- Break down complex information into bullet points
- Include relevant details like duration, fees, and requirements when discussing programs/courses
- When providing step-by-step guidance, number the steps clearly

# WHEN INFORMATION IS UNCLEAR:
1. Acknowledge the user's question
2. Explain what additional information you need
3. Provide examples of specific questions that would help
4. Offer to help with related information while waiting for clarification

# EXAMPLE RESPONSES:
For general queries: "I'd be happy to help! Could you please specify which type of program you're interested in? For example, are you looking for undergraduate, graduate, or professional programs?"

For unclear requests: "I want to make sure I understand your question correctly. Could you tell me more about what specific information you're looking for? For example, are you asking about admission requirements, course content, or career prospects?"

Remember: Always maintain a helpful and patient tone, even when the query is unclear or needs more context.
"""

    def _get_cache_key(self, text: str, use_semantic: bool = True) -> str:
        """Generate a cache key for the given text, with option for semantic hashing."""
        text = text.strip().lower()
        if use_semantic and len(text.split()) > 3:  # Only use semantic for longer texts
            # Simple semantic key - first and last few words + length
            words = text.split()
            semantic_key = f"{' '.join(words[:2])} {len(words)} {''.join(w[0] for w in words[1:-1])} {' '.join(words[-2:])}"
            return hashlib.md5(semantic_key.encode('utf-8')).hexdigest()
        return hashlib.md5(text.encode('utf-8')).hexdigest()
        
    def _get_cached_response(self, key: str, cache_type: str = 'llm') -> Optional[Any]:
        """Get cached response if it exists and is not expired."""
        cache = self.llm_cache if cache_type == 'llm' else self.response_cache
        if key in cache:
            entry = cache[key]
            if time.time() - entry.timestamp < entry.ttl:
                if cache_type == 'llm':
                    self.llm_cache_hits += 1
                else:
                    self.cache_hits += 1
                return entry.response
            del cache[key]  # Remove expired entry
        return None

    def _cache_response(self, key: str, response: Any, ttl: int = 3600, cache_type: str = 'llm') -> None:
        """Cache a response with TTL."""
        cache = self.llm_cache if cache_type == 'llm' else self.response_cache
        cache[key] = CacheEntry(
            response=response,
            timestamp=time.time(),
            ttl=ttl
        )
        # Simple cache eviction if cache gets too large
        if len(cache) > 1000:  # Keep last 1000 entries
            oldest_key = min(cache.keys(), key=lambda k: cache[k].timestamp)
            del cache[oldest_key]

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

    def _init_program_cache(self):
        """Initialize the program cache for fast lookups."""
        self._programs_cache = {
            "m.tech in design": "M.Tech in Design (M.Tech, 2 years (4 semesters))",
            "b.tech in biotechnology": "B.Tech in Biotechnology (B.Tech, 4 years (8 semesters))",
            "ph.d in design": "Ph.D in Design (Ph.D, 3 years (6 semesters))",
            "b.sc in civil engineering": "B.Sc in Civil Engineering (B.Sc, 4 years (8 semesters))",
            "mba in data science": "MBA in Data Science (MBA, 2 years (4 semesters))",
            "b.des in design": "B.Des in Design (B.Des, 4 years (8 semesters))",
            "m.sc in entrepreneurship": "M.Sc in Entrepreneurship (M.Sc, 2 years (4 semesters))",
            "mba in mechanical engineering": "MBA in Mechanical Engineering (MBA, 2 years (4 semesters))",
            "ph.d in information security": "Ph.D in Information Security (Ph.D, 3 years (6 semesters))",
            "m.tech in robotics": "M.Tech in Robotics (M.Tech, 2 years (4 semesters))",
            "mca in biotechnology": "MCA in Biotechnology (MCA, 2 years (4 semesters))",
            "b.tech in mechanical engineering": "B.Tech in Mechanical Engineering (B.Tech, 4 years (8 semesters))",
            "ph.d in robotics": "Ph.D in Robotics (Ph.D, 3 years (6 semesters))",
            "b.tech in robotics": "B.Tech in Robotics (B.Tech, 4 years (8 semesters))",
            "mca in information security": "MCA in Information Security (MCA, 2 years (4 semesters))",
            "m.tech in marketing": "M.Tech in Marketing (M.Tech, 2 years (4 semesters))",
            "m.tech in civil engineering": "M.Tech in Civil Engineering (M.Tech, 2 years (4 semesters))",
            "mba in finance": "MBA in Finance (MBA, 2 years (4 semesters))",
            "b.des in artificial intelligence": "B.Des in Artificial Intelligence (B.Des, 4 years (8 semesters))"
        }
        self._programs_list = sorted(self._programs_cache.values())

    def get_program_info(self, query: str) -> str:
        """Get information about a specific program with exact and fuzzy matching."""
        query = query.lower().strip()
        
        # Define degree types for reference
        degree_types = ['b.tech', 'm.tech', 'ph.d', 'b.des', 'mba', 'm.sc', 'mca', 'b.sc']
        
        # Try exact match first
        for program_name, program_info in self._programs_cache.items():
            # Check for exact program name match (case insensitive)
            if query == program_name.lower():
                return program_info
                
        # Extract degree type and program name from query
        query_degree = next((d for d in degree_types if query.startswith(d)), None)
        program_name_part = query.replace(query_degree, '').strip() if query_degree else query
        
        # Try matching with program name parts
        best_match = None
        best_score = 0
        
        for program_name, program_info in self._programs_cache.items():
            # Skip if degree types don't match (if degree was specified in query)
            if query_degree and not program_name.lower().startswith(query_degree):
                continue
                
            # Calculate match score based on word overlap
            program_words = set(program_name.lower().split())
            query_words = set(query.split())
            common_words = query_words.intersection(program_words)
            
            # Special handling for B.Tech in Robotics
            if 'robotics' in query and 'b.tech' in program_name.lower() and 'robotics' in program_name.lower():
                return program_info
                
            # Calculate score based on word matches
            score = len(common_words)
            
            # Bonus for matching degree type
            if query_degree and program_name.lower().startswith(query_degree):
                score += 2
                
            # Update best match if this one scores higher
            if score > best_score or (score == best_score and query_degree and program_name.lower().startswith(query_degree)):
                best_score = score
                best_match = program_info
        
        # Only return if we have a good match
        if best_score >= 1:  # Reduced threshold to 1 to catch more potential matches
            return best_match
            
        return None

    def list_all_programs(self) -> list:
        """Get a sorted list of all available programs."""
        return self._programs_list

    def _handle_program_query(self, query: str) -> str:
        """Handle program-related queries with optimized response times."""
        original_query = query
        query = query.lower().strip()
        
        # Check for list all programs
        if any(cmd in query for cmd in ["list", "all", "programs", "tell me about"]):
            programs = self.list_all_programs()
            return "Here are all available programs:\n\n" + "\n".join(f"- {p}" for p in programs)
        
        # Special case for B.Tech in Robotics
        if 'robotics' in query and ('b.tech' in query or 'btech' in query):
            program_info = self._programs_cache.get('b.tech in robotics')
            if program_info:
                program_name = program_info.split('(')[0].strip()
                return f"""Here's information about {program_name}:
                
• Program: {program_name}
• Degree: B.Tech
• Duration: 4 years (8 semesters)
• Eligibility: 10+2 with Physics, Chemistry, and Mathematics with minimum 50% marks
• Overview: The B.Tech in Robotics program provides comprehensive training in robotics engineering, automation, and intelligent systems. Students gain hands-on experience with industrial robots, AI, and machine learning applications in robotics.

Would you like to know more about the curriculum, fees, or placement opportunities?"""
        
        # Try to get specific program info
        program_info = self.get_program_info(original_query)  # Use original query for better matching
        if program_info:
            # Get detailed program information
            program_name = program_info.split('(')[0].strip()
            program_degree = program_info.split('(')[1].split(',')[0].strip()
            program_duration = program_info.split('(')[1].split(')')[0].split(',')[1].strip()
            
            # Custom descriptions for different program types
            program_type = program_degree.lower()
            field = program_name.split('in ')[-1] if 'in ' in program_name else program_name
            
            descriptions = {
                'b.tech': f"The {program_name} program provides a strong foundation in {field} with a focus on practical applications and industry-relevant skills.",
                'm.tech': f"The {program_name} program offers advanced specialization in {field} with research and development focus.",
                'mba': f"The {program_name} program develops business acumen and leadership skills in the field of {field}.",
                'ph.d': f"The {program_name} program is a research-intensive program for scholars in {field}.",
                'b.des': f"The {program_name} program combines creative design principles with technical skills in {field}.",
                'm.sc': f"The {program_name} program provides advanced theoretical and practical knowledge in {field}.",
                'mca': f"The {program_name} program focuses on computer applications and {field} with hands-on training.",
                'b.sc': f"The {program_name} program offers fundamental knowledge and practical skills in {field}."
            }
            
            description = descriptions.get(program_type.lower(), 
                f"The {program_name} program provides comprehensive education in {field}.")
            
            response = f"""Here's information about {program_name}:
            
• Program: {program_name}
• Degree: {program_degree}
• Duration: {program_duration}
• Eligibility: 10+2 with minimum 50% marks (45% for reserved categories)
• Overview: {description}

Would you like to know more about the curriculum, fees, or placement opportunities?"""
            return response
        
        # If no direct match, find similar programs
        query_degree = next((d for d in ['b.tech', 'm.tech', 'ph.d', 'b.des', 'mba', 'm.sc', 'mca', 'b.sc'] 
                           if d in query), None)
        
        if query_degree:
            similar = [p for name, p in self._programs_cache.items() 
                     if query_degree in name and any(word in name for word in query.split())]
        else:
            similar = [p for name, p in self._programs_cache.items() 
                     if any(word in name for word in query.split() if len(word) > 3)]
        
        if similar:
            return "I couldn't find an exact match, but here are some related programs:\n\n" + "\n".join(f"- {p}" for p in similar[:3]) + "\n\nPlease specify which program you're interested in."
        
        return "I couldn't find information about that program. Would you like to see all available programs?"

    def _classify_query_intent(self, user_input: str) -> str:
        """Classify the intent of the user query with better accuracy."""
        query = user_input.lower().strip()
        
        # First, check for general conversation patterns
        general_patterns = [
            r'how (are you|do you feel|is it going)',
            r'what(\'s| is) (up|new|happening)',
            r'^(hi|hello|hey|greetings|good (morning|afternoon|evening))',
            r'^(bye|goodbye|see you|take care)',
            r'thank',
            r'what can you do',
            r'who are you',
            r'your name',
            r'help$'
        ]
        
        if any(re.search(pattern, query) for pattern in general_patterns):
            return "general"
            
        # Check for program/course specific queries
        program_terms = [
            "program", "course", "degree", "b.tech", "m.tech", "mba", "phd", 
            "b.des", "m.sc", "mca", "admission", "fee", "tuition", "cost",
            "scholarship", "eligibility", "curriculum", "syllabus"
        ]
        
        if any(term in query for term in program_terms):
            return "program_query"
            
        # Check for faculty related queries
        faculty_terms = ["professor", "faculty", "teacher", "lecturer", "staff"]
        if any(term in query for term in faculty_terms):
            return "faculty_query"
        
        # Check for university facilities
        facility_terms = ["campus", "library", "hostel", "lab", "sports", "facility", "infrastructure"]
        if any(term in query for term in facility_terms):
            return "facility_query"
        
        # Check for farewells
        farewells = ['bye', 'goodbye', 'see you', 'take care', 'farewell']
        if any(query == f or query.startswith(f + ' ') for f in farewells):
            return "farewell"
        
        # Check for thanks
        thanks = ['thank you', 'thanks', 'appreciate', 'grateful']
        if any(t in query for t in thanks):
            return "thanks"
        
        # Check for counting/list queries
        counting_phrases = ['how many', 'list all', 'what are all', 'show me all', 'tell me about']
        if any(phrase in query for phrase in counting_phrases):
            return "university"
        
        # Default to general conversation
        return "general"
    
    def _handle_general_conversation(self, user_input: str) -> str:
        """Handle general conversation using LLM with better context and personality."""
        try:
            # Build conversation context
            context = "\n".join(
                f"{'User' if msg['role'] == 'user' else 'Assistant'}: {msg['content']}"
                for msg in self.conversation_context[-4:]  # Last 2 exchanges
            )
            
            # Enhanced prompt with personality and context
            prompt = f"""You are a friendly and helpful university assistant. You are having a conversation with a student or prospective student.

Previous conversation:
{context if context else "No previous context."}

Current query: {user_input}

Guidelines:
1. Be warm, professional, and engaging
2. Keep responses concise but helpful
3. If appropriate, gently guide the conversation toward university-related topics
4. If you don't know something, say so and offer to help find the information
5. For personal questions, respond naturally and professionally

Respond naturally to the user's message:"""
            
            # Get response from LLM
            response = self.llm.complete(prompt).text.strip()
            
            # Clean up and format the response
            response = response.split('\n')[0]  # Take only the first line if multiple lines
            if not response or len(response) < 2:  # Fallback if response is too short
                response = "I'm here to help! Could you tell me more about what you're looking for?"
                
            return response
            
        except Exception as e:
            print(f"Error in general conversation: {e}")
            return "I'm here to help! Feel free to ask me about university programs, courses, or any other questions you might have."
    
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
    
    def _extract_preferences(self, query: str, response: str) -> None:
        """Extract and store user preferences from the conversation."""
        # Look for program mentions
        program_keywords = ['program', 'course', 'degree', 'bachelor', 'master', 'phd', 'b.tech', 'b.arch', 'mba']
        if any(keyword in query.lower() for keyword in program_keywords):
            # Simple pattern matching for program names
            import re
            program_matches = re.findall(r'\b(B\.?Tech|B\.?Arch|MBA|M\.?Tech|PhD|Ph\.D|B\.?Com|BBA|BCA|MCA)\b', 
                                       query, re.IGNORECASE)
            for program in program_matches:
                self.user_preferences['preferred_programs'].add(program.upper())
        
        # Update response style preference
        if 'detailed' in query.lower() and 'explanation' in query.lower():
            self.user_preferences['preferred_response_style'] = 'detailed'
        elif 'be brief' in query.lower() or 'be concise' in query.lower():
            self.user_preferences['preferred_response_style'] = 'concise'
    
    def interact_with_llm(self, user_input: str) -> str:
        """Optimized LLM interaction with improved caching and routing."""
        self.total_queries += 1
        
        if not user_input.strip():
            return "Please provide a valid question or query."
        
        # Generate cache keys
        exact_key = self._get_cache_key(user_input, use_semantic=False)
        semantic_key = self._get_cache_key(user_input, use_semantic=True)
        
        # Check caches (exact match first, then semantic)
        if cached := self._get_cached_response(exact_key, 'llm') or \
                      self._get_cached_response(semantic_key, 'llm'):
            print(f"💾 Cache hit! (LLM cache: {self.llm_cache_hits}/{self.total_queries})")
            return cached
        
        # Handle program queries with optimized path
        intent = self._classify_query_intent(user_input)
        if intent == "program_query":
            response = self._handle_program_query(user_input)
            self._cache_response(exact_key, response, ttl=3600, cache_type='llm')
            return response
        
        # Handle conversation management
        if user_input.lower() in ['what do you know about me?', 'what do you remember?']:
            response = self._get_user_context_summary()
            self._cache_response(exact_key, response, ttl=3600, cache_type='llm')
            return response
        
        # Route based on intent
        if intent in ["greeting", "farewell", "thanks", "general"]:
            response = self._handle_general_conversation(user_input)
            self._cache_response(exact_key, response, ttl=1800, cache_type='llm')
            return response
        
        # Handle simple queries with direct Qdrant access
        query_type, entity_type = self._is_simple_query(user_input)
        if query_type and entity_type:
            print(f"🚀 Fast path: {query_type} query for {entity_type}")
            if fast_response := self._handle_fast_query(query_type, entity_type, user_input):
                self.conversation_context.append({
                    "query": user_input, 
                    "response": fast_response
                })
                if len(self.conversation_context) > 5:
                    self.conversation_context.pop(0)
                self._cache_response(exact_key, fast_response, ttl=3600, cache_type='llm')
                return fast_response
        
        # Process with RAG
        try:
            print("🤖 Using optimized RAG pipeline...")
            
            # Enhance query with conversation context
            enhanced_query = self._enhance_query_with_context(user_input)
            
            # Get response from RAG
            response = self._chat_engine.chat(enhanced_query).response
            
            # Cache the response
            self._cache_response(exact_key, response, ttl=3600, cache_type='llm')
            self._cache_response(semantic_key, response, ttl=1800, cache_type='llm')
            
            # Update conversation context
            self._update_conversation_context(user_input, response)
            self._extract_preferences(user_input, response)
            
            return response
            
        except Exception as e:
            error_msg = str(e).lower()
            print(f"Error in RAG query: {e}")
            
            if "timeout" in error_msg or "readtimeout" in error_msg:
                return "The request timed out. Please try a more specific query."
            elif "connection" in error_msg:
                return "Unable to connect to the AI service. Please try again later."
            return "I'm having trouble processing your request. Could you please rephrase?"

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