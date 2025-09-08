import os
import json
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from datetime import datetime
import hashlib

# Configuration
EMB_MODEL = r"C:\TextSummarization\TS\all-MiniLM-L6-v2"
DB_DIR = "storage/chroma"
CHUNK_SIZE = 1000  # Characters per chunk
CHUNK_OVERLAP = 200  # Overlap between chunks

class UnderwritingVectorDB:
    """Manages vector database for underwriting documents"""
    
    def __init__(self):
        self.embed_model = None
        self.client = None
        self.collection = None
        self._initialize()
    
    def _initialize(self):
        """Initialize embedding model and ChromaDB"""
        # Load embedding model
        print("Loading embedding model for underwriting...")
        self.embed_model = SentenceTransformer(EMB_MODEL)
        
        # Initialize ChromaDB
        os.makedirs(DB_DIR, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=DB_DIR, 
            settings=Settings(allow_reset=False)
        )
        
        # Create/get underwriting collection
        self.collection = self.client.get_or_create_collection(
            name="underwriting_documents",
            metadata={"description": "Insurance application documents for underwriting"}
        )
    
    def chunk_document(self, content: str, doc_id: str, doc_type: str) -> List[Dict[str, Any]]:
        """
        Split document into chunks for embedding
        
        Args:
            content: Document text content
            doc_id: Unique document identifier
            doc_type: Type of insurance (health, life, motor, commercial)
        
        Returns:
            List of chunks with metadata
        """
        chunks = []
        
        # Split content into chunks
        for i in range(0, len(content), CHUNK_SIZE - CHUNK_OVERLAP):
            chunk_text = content[i:i + CHUNK_SIZE]
            
            # Create chunk ID
            chunk_id = f"{doc_id}_chunk_{i // (CHUNK_SIZE - CHUNK_OVERLAP)}"
            
            # Extract key information from chunk for metadata
            chunk_metadata = self._extract_chunk_metadata(chunk_text, doc_type)
            
            chunks.append({
                "id": chunk_id,
                "text": chunk_text,
                "metadata": {
                    "doc_id": doc_id,
                    "doc_type": doc_type,
                    "chunk_index": i // (CHUNK_SIZE - CHUNK_OVERLAP),
                    "timestamp": datetime.now().isoformat(),
                    **chunk_metadata
                }
            })
        
        return chunks
    
    def _extract_chunk_metadata(self, chunk_text: str, doc_type: str) -> Dict[str, Any]:
        """Extract relevant metadata from chunk text"""
        import re
        
        metadata = {}
        chunk_lower = chunk_text.lower()
        
        # Extract common information
        if "age:" in chunk_lower or "age " in chunk_lower:
            age_match = re.search(r'age[:\s]*(\d+)', chunk_lower)
            if age_match:
                metadata["age"] = int(age_match.group(1))
        
        if "income" in chunk_lower:
            income_match = re.search(r'income[:\s]*(?:rs\.?\s*)?([\d,]+)', chunk_lower)
            if income_match:
                metadata["income"] = income_match.group(1).replace(',', '')
        
        # Document type specific extraction
        if doc_type == "health":
            if "bmi" in chunk_lower:
                bmi_match = re.search(r'bmi[:\s]*(\d+\.?\d*)', chunk_lower)
                if bmi_match:
                    metadata["bmi"] = float(bmi_match.group(1))
            
            if "medical condition" in chunk_lower or "diagnosis" in chunk_lower:
                metadata["has_medical_info"] = True
            
            if "medication" in chunk_lower or "prescription" in chunk_lower:
                metadata["has_medication_info"] = True
        
        elif doc_type == "commercial":
            if "turnover" in chunk_lower:
                metadata["has_financial_info"] = True
            
            if "employees" in chunk_lower:
                emp_match = re.search(r'employees[:\s]*(\d+)', chunk_lower)
                if emp_match:
                    metadata["employee_count"] = int(emp_match.group(1))
        
        elif doc_type == "motor":
            if "vehicle" in chunk_lower or "registration" in chunk_lower:
                metadata["has_vehicle_info"] = True
            
            if "accident" in chunk_lower or "claim" in chunk_lower:
                metadata["has_claims_history"] = True
        
        return metadata
    
    def index_document(self, content: str, filename: str, doc_type: str) -> Dict[str, Any]:
        """
        Index a document into the vector database
        
        Args:
            content: Document text content
            filename: Original filename
            doc_type: Type of insurance document
        
        Returns:
            Indexing result with statistics
        """
        try:
            # Generate document ID
            doc_id = hashlib.md5(f"{filename}_{datetime.now().isoformat()}".encode()).hexdigest()[:12]
            
            # Chunk the document
            chunks = self.chunk_document(content, doc_id, doc_type)
            
            if not chunks:
                return {"error": "No chunks created from document"}
            
            # Prepare data for ChromaDB
            ids = [chunk["id"] for chunk in chunks]
            texts = [chunk["text"] for chunk in chunks]
            metadatas = [chunk["metadata"] for chunk in chunks]
            
            # Generate embeddings
            embeddings = self.embed_model.encode(texts, normalize_embeddings=True).tolist()
            
            # Add to collection
            self.collection.upsert(
                ids=ids,
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas
            )
            
            return {
                "status": "success",
                "doc_id": doc_id,
                "filename": filename,
                "doc_type": doc_type,
                "chunks_indexed": len(chunks),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"error": f"Failed to index document: {str(e)}"}
    
    def search_documents(self, query: str, doc_type: Optional[str] = None, 
                        k: int = 5) -> List[Dict[str, Any]]:
        """
        Search indexed documents using semantic search
        
        Args:
            query: Search query
            doc_type: Optional filter by document type
            k: Number of results to return
        
        Returns:
            List of relevant document chunks
        """
        try:
            # Generate query embedding
            query_embedding = self.embed_model.encode([query], normalize_embeddings=True).tolist()[0]
            
            # Build where clause for filtering
            where_clause = {"doc_type": doc_type} if doc_type else None
            
            # Search collection
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(k, self.collection.count()),
                where=where_clause
            )
            
            # Format results
            formatted_results = []
            if results["metadatas"] and results["documents"]:
                for metadata, document, distance in zip(
                    results["metadatas"][0], 
                    results["documents"][0], 
                    results["distances"][0]
                ):
                    formatted_results.append({
                        "doc_id": metadata.get("doc_id"),
                        "doc_type": metadata.get("doc_type"),
                        "chunk_index": metadata.get("chunk_index"),
                        "content": document,
                        "metadata": metadata,
                        "relevance_score": round(1 / (1 + distance), 3)
                    })
            
            return formatted_results
            
        except Exception as e:
            return [{"error": f"Search failed: {str(e)}"}]
    
    def get_document_context(self, doc_id: str, chunk_index: int, 
                           context_size: int = 2) -> str:
        """
        Get surrounding context for a specific chunk
        
        Args:
            doc_id: Document ID
            chunk_index: Index of the target chunk
            context_size: Number of chunks before/after to include
        
        Returns:
            Combined text of target chunk with context
        """
        try:
            # Get chunks around the target
            chunk_ids = []
            for i in range(max(0, chunk_index - context_size), 
                          chunk_index + context_size + 1):
                chunk_ids.append(f"{doc_id}_chunk_{i}")
            
            # Retrieve chunks
            results = self.collection.get(ids=chunk_ids)
            
            if results["documents"]:
                return "\n".join(results["documents"])
            
            return ""
            
        except Exception as e:
            print(f"Error getting context: {e}")
            return ""
    
    def analyze_with_rag(self, query: str, doc_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Perform RAG-based analysis for underwriting queries
        
        Args:
            query: Analysis query
            doc_type: Optional document type filter
        
        Returns:
            Analysis results with relevant document sections
        """
        # Search for relevant chunks
        search_results = self.search_documents(query, doc_type, k=10)
        
        if not search_results or "error" in search_results[0]:
            return {"error": "No relevant documents found"}
        
        # Group results by document
        docs_data = {}
        for result in search_results:
            doc_id = result["doc_id"]
            if doc_id not in docs_data:
                docs_data[doc_id] = []
            docs_data[doc_id].append(result)
        
        # Get expanded context for top documents
        relevant_sections = []
        for doc_id, chunks in list(docs_data.items())[:3]:  # Top 3 documents
            # Get the most relevant chunk
            top_chunk = max(chunks, key=lambda x: x["relevance_score"])
            
            # Get surrounding context
            context = self.get_document_context(
                doc_id, 
                top_chunk["chunk_index"],
                context_size=2
            )
            
            relevant_sections.append({
                "doc_id": doc_id,
                "doc_type": top_chunk["doc_type"],
                "content": context,
                "relevance_score": top_chunk["relevance_score"]
            })
        
        return {
            "query": query,
            "relevant_sections": relevant_sections,
            "total_matches": len(search_results)
        }
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the indexed documents"""
        try:
            count = self.collection.count()
            
            # Get unique documents
            all_metadata = self.collection.get()["metadatas"]
            unique_docs = set()
            doc_types = {}
            
            for metadata in all_metadata:
                doc_id = metadata.get("doc_id")
                doc_type = metadata.get("doc_type")
                
                if doc_id:
                    unique_docs.add(doc_id)
                
                if doc_type:
                    doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
            
            return {
                "total_chunks": count,
                "unique_documents": len(unique_docs),
                "documents_by_type": doc_types
            }
            
        except Exception as e:
            return {"error": f"Failed to get stats: {str(e)}"}
