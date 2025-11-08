"""
Data Ingestion Script for Qdrant Vector Database
This script reads the knowledge base JSON file, converts it to embeddings,
and uploads it to Qdrant for use by the AI Voice Assistant.

Usage:
    python ingest_data_to_qdrant.py
    
    Optional arguments:
    --kb-path: Path to knowledge base JSON file (default: voice_rag_kb.json)
    --collection: Qdrant collection name (default: university_kb)
    --qdrant-url: Qdrant server URL (default: http://localhost:6333)
    --force: Force recreate collection even if it exists
"""

import sys
import io
# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import json
import os
import argparse
from typing import List
from qdrant_client import QdrantClient, models
from llama_index.core import Document, Settings, VectorStoreIndex, StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


class QdrantDataIngestion:
    def __init__(self, kb_path: str, collection_name: str, qdrant_url: str):
        self.kb_path = kb_path
        self.collection_name = collection_name
        self.qdrant_url = qdrant_url
        self.client = None
        self.embeddings = None
        
    def connect_to_qdrant(self):
        """Connect to Qdrant server."""
        print(f"\n{'='*60}")
        print("CONNECTING TO QDRANT")
        print(f"{'='*60}")
        print(f"URL: {self.qdrant_url}")
        
        try:
            self.client = QdrantClient(
                url=self.qdrant_url,
                prefer_grpc=False,
                timeout=10.0
            )
            # Test connection
            collections = self.client.get_collections()
            print(f"✓ Connected successfully!")
            print(f"  Existing collections: {len(collections.collections)}")
            return True
        except Exception as e:
            print(f"❌ Failed to connect to Qdrant: {e}")
            print("\nMake sure Qdrant is running:")
            print("  - Docker: docker run -p 6333:6333 qdrant/qdrant")
            print("  - Or check if Qdrant service is running")
            return False
    
    def initialize_embeddings(self):
        """Initialize the embedding model."""
        print(f"\n{'='*60}")
        print("INITIALIZING EMBEDDING MODEL")
        print(f"{'='*60}")
        print("Model: BAAI/bge-small-en-v1.5")
        print("Dimension: 384")
        
        try:
            self.embeddings = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
            Settings.embed_model = self.embeddings
            
            # Test embedding
            test_embedding = self.embeddings.get_text_embedding("test")
            print(f"✓ Embedding model loaded successfully!")
            print(f"  Test embedding dimension: {len(test_embedding)}")
            return True
        except Exception as e:
            print(f"❌ Failed to load embedding model: {e}")
            return False
    
    def load_knowledge_base(self) -> List[Document]:
        """Load and parse the knowledge base JSON file."""
        print(f"\n{'='*60}")
        print("LOADING KNOWLEDGE BASE")
        print(f"{'='*60}")
        print(f"File: {os.path.abspath(self.kb_path)}")
        
        if not os.path.exists(self.kb_path):
            print(f"❌ Knowledge base file not found: {self.kb_path}")
            return []
        
        try:
            with open(self.kb_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            documents = []
            stats = {}
            
            # Process Programs
            programs = data.get('programs', [])
            if programs:
                stats['programs'] = len(programs)
                
                # Create summary document
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
                
                documents.append(Document(
                    text=summary_text,
                    metadata={"type": "programs_summary", "id": "programs_summary", "count": str(len(programs))}
                ))
                
                # Individual program documents
                for program in programs:
                    text = f"""Program: {program.get('name', 'N/A')}
Duration: {program.get('duration', 'N/A')}
Degree: {program.get('degree', 'N/A')}
Description: {program.get('description', 'N/A')}
Eligibility: {program.get('eligibility_summary', 'N/A')}"""
                    documents.append(Document(
                        text=text,
                        metadata={
                            "type": "program",
                            "id": program.get('id', ''),
                            "name": program.get('name', '')
                        }
                    ))
            
            # Process other sections
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
            
            for section_name, doc_type in sections:
                items = data.get(section_name, [])
                if not items:
                    continue
                
                stats[section_name] = len(items)
                
                # Create summary document
                item_list = []
                for i, item in enumerate(items, 1):
                    name = item.get('name', item.get('title', 'N/A'))
                    item_list.append(f"{i}. {name}")
                
                section_summary = f"""Question: How many {section_name} are there?
Answer: There are {len(items)} {section_name} in total.

Question: What are all the {section_name}?
Answer: Here is the complete list of all {len(items)} {section_name}:
{chr(10).join(item_list)}

Question: List all {section_name}
Answer: The university offers {len(items)} {section_name}:
{chr(10).join(item_list)}

SUMMARY: This university has exactly {len(items)} {section_name}."""
                
                documents.append(Document(
                    text=section_summary,
                    metadata={
                        "type": f"{doc_type}_summary",
                        "id": f"{section_name}_summary",
                        "count": str(len(items))
                    }
                ))
                
                # Individual documents
                for item in items:
                    text_lines = [f"{k.replace('_', ' ').title()}: {v}" 
                                for k, v in item.items() 
                                if v and not k.startswith('_')]
                    
                    documents.append(Document(
                        text='\n'.join(text_lines),
                        metadata={
                            "type": doc_type,
                            "id": str(item.get('id', '')),
                            "name": str(item.get('name', item.get('title', '')))[:100]
                        }
                    ))
            
            print(f"\n✓ Knowledge base loaded successfully!")
            print(f"\nData Statistics:")
            for key, value in stats.items():
                print(f"  - {key.capitalize()}: {value}")
            print(f"\nTotal documents created: {len(documents)}")
            
            return documents
            
        except Exception as e:
            print(f"❌ Error loading knowledge base: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def create_collection(self, force: bool = False):
        """Create or recreate the Qdrant collection."""
        print(f"\n{'='*60}")
        print("CREATING QDRANT COLLECTION")
        print(f"{'='*60}")
        print(f"Collection name: {self.collection_name}")
        
        try:
            # Check if collection exists
            try:
                existing = self.client.get_collection(self.collection_name)
                if force:
                    print(f"⚠ Collection exists. Deleting due to --force flag...")
                    self.client.delete_collection(self.collection_name)
                    print("✓ Deleted existing collection")
                else:
                    print(f"⚠ Collection already exists!")
                    print(f"  Use --force to recreate it")
                    return False
            except:
                print("No existing collection found. Creating new one...")
            
            # Create collection with vector configuration
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=384,  # BAAI/bge-small-en-v1.5 dimension
                    distance=models.Distance.COSINE
                )
            )
            print(f"✓ Collection '{self.collection_name}' created successfully!")
            return True
            
        except Exception as e:
            print(f"❌ Error creating collection: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def upload_documents(self, documents: List[Document]):
        """Upload documents to Qdrant with embeddings."""
        print(f"\n{'='*60}")
        print("UPLOADING DOCUMENTS TO QDRANT")
        print(f"{'='*60}")
        print(f"Total documents: {len(documents)}")
        
        try:
            # Create vector store
            vector_store = QdrantVectorStore(
                client=self.client,
                collection_name=self.collection_name,
                batch_size=32
            )
            
            # Create storage context
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            
            # Create index and upload
            print("\nGenerating embeddings and uploading...")
            index = VectorStoreIndex.from_documents(
                documents,
                storage_context=storage_context,
                show_progress=True,
                embed_model=self.embeddings
            )
            
            # Verify upload
            collection_info = self.client.get_collection(self.collection_name)
            
            # Get vector count safely (different Qdrant versions use different attributes)
            vectors_count = 0
            if hasattr(collection_info, 'vectors_count'):
                vectors_count = collection_info.vectors_count or 0
            elif hasattr(collection_info, 'points_count'):
                vectors_count = collection_info.points_count or 0
            
            # Alternative: Count directly from Qdrant
            if vectors_count == 0:
                try:
                    count_result = self.client.count(collection_name=self.collection_name)
                    vectors_count = count_result.count if hasattr(count_result, 'count') else count_result
                except:
                    vectors_count = len(documents)  # Fallback to document count
            
            print(f"\n✓ Upload completed successfully!")
            print(f"  Documents processed: {len(documents)}")
            print(f"  Vectors stored in Qdrant: {vectors_count}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error uploading documents: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def run(self, force: bool = False):
        """Run the complete ingestion pipeline."""
        print(f"\n{'#'*60}")
        print("QDRANT DATA INGESTION PIPELINE")
        print(f"{'#'*60}")
        
        # Step 1: Connect to Qdrant
        if not self.connect_to_qdrant():
            return False
        
        # Step 2: Initialize embeddings
        if not self.initialize_embeddings():
            return False
        
        # Step 3: Load knowledge base
        documents = self.load_knowledge_base()
        if not documents:
            print("\n❌ No documents to upload. Exiting.")
            return False
        
        # Step 4: Create collection
        if not self.create_collection(force=force):
            if not force:
                print("\n⚠ Skipping upload. Use --force to recreate collection.")
                return False
        
        # Step 5: Upload documents
        if not self.upload_documents(documents):
            return False
        
        print(f"\n{'#'*60}")
        print("✓ DATA INGESTION COMPLETED SUCCESSFULLY!")
        print(f"{'#'*60}")
        print(f"\nYour knowledge base is now ready in Qdrant!")
        print(f"Collection: {self.collection_name}")
        print(f"URL: {self.qdrant_url}")
        print(f"\nYou can now use this data in your AI Voice Assistant.")
        
        return True


def main():
    parser = argparse.ArgumentParser(
        description="Ingest knowledge base data into Qdrant vector database"
    )
    parser.add_argument(
        '--kb-path',
        type=str,
        default='voice_rag_kb.json',
        help='Path to knowledge base JSON file (default: voice_rag_kb.json)'
    )
    parser.add_argument(
        '--collection',
        type=str,
        default='university_kb',
        help='Qdrant collection name (default: university_kb)'
    )
    parser.add_argument(
        '--qdrant-url',
        type=str,
        default='http://localhost:6333',
        help='Qdrant server URL (default: http://localhost:6333)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force recreate collection even if it exists'
    )
    
    args = parser.parse_args()
    
    # Run ingestion
    ingestion = QdrantDataIngestion(
        kb_path=args.kb_path,
        collection_name=args.collection,
        qdrant_url=args.qdrant_url
    )
    
    success = ingestion.run(force=args.force)
    
    if not success:
        exit(1)


if __name__ == "__main__":
    main()
