# -*- coding: utf-8 -*-
"""
Document Analyzer Agent - RAG Foundation

This agent parses research papers (PDF, arXiv, HTML, Markdown) and extracts
structured knowledge including:
- Algorithms and methodologies
- Model architectures
- Hyperparameters and training details
- Dataset information
- Code snippets and pseudocode
- Implementation notes

The extracted knowledge is stored in a ChromaDB vector database for
efficient retrieval during code generation.

Authors: GaneshS 271123
gmail:ganeshnaik214@gmail.com
"""

import os
import re
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from loguru import logger
import google.generativeai as genai
from sentence_transformers import SentenceTransformer

# Import BaseAgent for inheritance
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from agents.base_agent import BaseAgent


class DocumentAnalyzerAgent(BaseAgent):
    """
    Agent for parsing and analyzing research papers and technical documents.
    
    Capabilities:
    - Parse PDF files (research papers)
    - Extract text from arXiv papers
    - Parse Markdown and HTML documentation
    - Extract structured knowledge (algorithms, architectures, etc.)
    - Store embeddings in ChromaDB vector database
    - Support multi-document context
    
    Example:
        >>> agent = DocumentAnalyzerAgent()
        >>> knowledge = await agent.analyze_document("ResNet_paper.pdf")
        >>> print(knowledge['algorithms'])
        ['Residual Learning', 'Batch Normalization']
    """
    
    def __init__(
        self,
        llm_client=None,
        prompt_manager=None,
        vector_db_path: str = "data/vector_db",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        """
        Initialize DocumentAnalyzerAgent.
        
        Args:
            llm_client: LLM client for text analysis (uses Gemini)
            prompt_manager: Prompt manager for loading prompts
            vector_db_path: Path to ChromaDB database
            chunk_size: Size of text chunks for embedding
            chunk_overlap: Overlap between chunks
            embedding_model: Sentence Transformer model name (default: all-MiniLM-L6-v2)
        """
        # Initialize base agent
        if llm_client is None or prompt_manager is None:
            from llm.gemini_client import GeminiClient
            from llm.prompt_manager import PromptManager
            llm_client = llm_client or GeminiClient()
            prompt_manager = prompt_manager or PromptManager()
        
        super().__init__(
            agent_type="document_analysis",
            name="document_analyzer_agent",
            role="Expert Document Analyst and Knowledge Extractor for Research Papers",
            llm_client=llm_client,
            prompt_manager=prompt_manager
        )
        
        self.vector_db_path = Path(vector_db_path)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model_name = embedding_model
        
        # Initialize Sentence Transformers embedding model
        self.embedding_model = self._init_embedding_model()
        
        # Initialize ChromaDB
        self._init_vector_db()
        
        # Supported file types
        self.supported_formats = ['.pdf', '.md', '.html', '.txt', '.arxiv']
        
        logger.info(f"{self.name} ready for document analysis")
        logger.info(f"   - Vector DB: {self.vector_db_path}")
        logger.info(f"   - Chunk size: {self.chunk_size}")
        logger.info(f"   - Embedding model: {self.embedding_model_name} (Sentence Transformers)")
    
    def _init_embedding_model(self):
        """Initialize Sentence Transformers embedding model."""
        try:
            logger.info(f"Loading Sentence Transformer model: {self.embedding_model_name}")
            model = SentenceTransformer(self.embedding_model_name)
            logger.info(f"✅ Embedding model loaded (dimension: {model.get_sentence_embedding_dimension()})")
            return model
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise
    
    def _init_vector_db(self):
        """Initialize ChromaDB client and collection with proper embedding dimensions."""
        try:
            import chromadb
            from chromadb.config import Settings
            
            # Create vector DB directory
            self.vector_db_path.mkdir(parents=True, exist_ok=True)
            
            # Initialize ChromaDB client
            self.chroma_client = chromadb.PersistentClient(
                path=str(self.vector_db_path),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Get embedding dimension from model
            embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
            
            # Try to get existing collection, delete if dimension mismatch
            collection_name = "research_papers"
            try:
                existing_collection = self.chroma_client.get_collection(name=collection_name)
                # Delete and recreate to ensure correct dimensions
                logger.info(f"Deleting existing collection '{collection_name}' to update dimensions...")
                self.chroma_client.delete_collection(name=collection_name)
            except:
                pass  # Collection doesn't exist yet
            
            # Create collection without default embedding function
            # We'll provide embeddings explicitly from Sentence Transformers
            self.collection = self.chroma_client.create_collection(
                name=collection_name,
                metadata={
                    "description": "Research papers and technical documents",
                    "embedding_model": self.embedding_model_name,
                    "embedding_dimension": embedding_dim
                }
            )
            
            logger.info("ChromaDB initialized successfully")
            logger.info(f"   - Collection: {collection_name}")
            logger.info(f"   - Embedding dimension: {embedding_dim} ({self.embedding_model_name})")
            
        except ImportError:
            logger.error("ChromaDB not installed. Run: pip install chromadb")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise
    
    def _generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for text using Sentence Transformers.
        
        Args:
            text: Text to embed
            
        Returns:
            List of embedding values (384 dimensions for all-MiniLM-L6-v2)
        """
        try:
            # Generate embedding using Sentence Transformers
            embedding = self.embedding_model.encode(text, convert_to_numpy=True)
            return embedding.tolist()
            
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            # Return zero vector as fallback
            dim = self.embedding_model.get_sentence_embedding_dimension()
            return [0.0] * dim
    
    def _generate_query_embedding(self, query: str) -> List[float]:
        """
        Generate embedding for search query using Sentence Transformers.
        
        Args:
            query: Search query
            
        Returns:
            List of embedding values
        """
        try:
            # Generate embedding (same method for query and document)
            embedding = self.embedding_model.encode(query, convert_to_numpy=True)
            return embedding.tolist()
            
        except Exception as e:
            logger.error(f"Failed to generate query embedding: {e}")
            # Return zero vector as fallback
            dim = self.embedding_model.get_sentence_embedding_dimension()
            return [0.0] * dim
    
    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute document analysis task.
        
        Args:
            task: Task dictionary with keys:
                - task_type: "analyze_document" or "query_knowledge"
                - data: Task-specific data
        
        Returns:
            Task result dictionary
        """
        task_type = task.get("task_type", "")
        data = task.get("data", {})
        
        if task_type == "analyze_document":
            filepath = data.get("filepath")
            return await self.analyze_document(filepath)
        
        elif task_type == "query_knowledge":
            query = data.get("query")
            top_k = data.get("top_k", 5)
            return await self.query_knowledge(query, top_k)
        
        elif task_type == "list_documents":
            return self.list_indexed_documents()
        
        else:
            return {
                "status": "error",
                "message": f"Unknown task type: {task_type}"
            }
    
    async def analyze_document(
        self,
        filepath: Union[str, Path],
        document_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze a document and extract structured knowledge.
        
        Args:
            filepath: Path to document file
            document_id: Optional unique ID for document
        
        Returns:
            Dictionary with extracted knowledge:
            {
                "status": "success",
                "document_id": "doc_123",
                "title": "Deep Residual Learning for Image Recognition",
                "authors": ["Kaiming He", ...],
                "year": 2015,
                "summary": "...",
                "algorithms": [...],
                "architectures": [...],
                "hyperparameters": {...},
                "datasets": [...],
                "code_snippets": [...],
                "chunks_indexed": 45,
                "total_tokens": 15000
            }
        """
        filepath = Path(filepath)
        
        logger.info(f"Analyzing document: {filepath.name}")
        
        # Validate file
        if not filepath.exists():
            return {
                "status": "error",
                "message": f"File not found: {filepath}"
            }
        
        if filepath.suffix not in self.supported_formats:
            return {
                "status": "error",
                "message": f"Unsupported format: {filepath.suffix}. Supported: {self.supported_formats}"
            }
        
        # Generate document ID
        document_id = document_id or self._generate_document_id(filepath)
        
        # Step 1: Extract text from document
        logger.info(f"Step 1: Extracting text from {filepath.suffix} file...")
        text_content = await self._extract_text(filepath)
        
        if not text_content:
            return {
                "status": "error",
                "message": "Failed to extract text from document"
            }
        
        logger.info(f"   ✓ Extracted {len(text_content)} characters")
        
        # Step 2: Chunk text for embedding
        logger.info("Step 2: Chunking text for embedding...")
        chunks = self._chunk_text(text_content)
        logger.info(f"   ✓ Created {len(chunks)} chunks")
        
        # Step 3: Extract structured knowledge using LLM
        logger.info("Step 3: Extracting structured knowledge with LLM...")
        knowledge = await self._extract_knowledge(text_content, filepath.name)
        logger.info(f"   ✓ Extracted: {len(knowledge.get('algorithms', []))} algorithms, "
                   f"{len(knowledge.get('architectures', []))} architectures")
        
        # Step 4: Store chunks in vector database
        logger.info("Step 4: Storing embeddings in ChromaDB...")
        num_indexed = await self._index_chunks(chunks, document_id, knowledge)
        logger.info(f"   ✓ Indexed {num_indexed} chunks")
        
        # Build result
        result = {
            "status": "success",
            "document_id": document_id,
            "filename": filepath.name,
            "chunks_indexed": num_indexed,
            "total_characters": len(text_content),
            **knowledge
        }
        
        logger.info(f"✅ Document analysis complete: {document_id}")
        
        return result
    
    async def _extract_text(self, filepath: Path) -> str:
        """
        Extract text from document based on file type.
        
        Args:
            filepath: Path to document
        
        Returns:
            Extracted text content
        """
        suffix = filepath.suffix.lower()
        
        try:
            if suffix == '.pdf':
                return await self._extract_pdf(filepath)
            elif suffix in ['.md', '.txt']:
                return await self._extract_text_file(filepath)
            elif suffix == '.html':
                return await self._extract_html(filepath)
            elif suffix == '.arxiv':
                return await self._extract_arxiv(filepath)
            else:
                logger.error(f"Unsupported file type: {suffix}")
                return ""
                
        except Exception as e:
            logger.error(f"Error extracting text: {e}")
            return ""
    
    async def _extract_pdf(self, filepath: Path) -> str:
        """Extract text from PDF file using PyMuPDF."""
        try:
            import fitz  # PyMuPDF
            
            text_content = []
            doc = fitz.open(filepath)
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                text_content.append(text)
            
            doc.close()
            
            full_text = "\n\n".join(text_content)
            return full_text
            
        except ImportError:
            logger.error("PyMuPDF not installed. Run: pip install PyMuPDF")
            return ""
        except Exception as e:
            logger.error(f"Error reading PDF: {e}")
            return ""
    
    async def _extract_text_file(self, filepath: Path) -> str:
        """Extract text from .txt or .md file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error reading text file: {e}")
            return ""
    
    async def _extract_html(self, filepath: Path) -> str:
        """Extract text from HTML file."""
        try:
            from bs4 import BeautifulSoup
            
            with open(filepath, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            text = soup.get_text()
            
            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)
            
            return text
            
        except ImportError:
            logger.error("BeautifulSoup not installed. Run: pip install beautifulsoup4")
            return ""
        except Exception as e:
            logger.error(f"Error reading HTML: {e}")
            return ""
    
    async def _extract_arxiv(self, filepath: Path) -> str:
        """
        Extract text from arXiv paper (contains arXiv ID).
        
        Format: File contains arXiv ID like "2015.12345"
        Downloads and extracts PDF content.
        """
        try:
            import arxiv
            
            # Read arXiv ID from file
            with open(filepath, 'r') as f:
                arxiv_id = f.read().strip()
            
            # Search arXiv
            search = arxiv.Search(id_list=[arxiv_id])
            paper = next(search.results())
            
            # Download PDF to temp location
            temp_pdf = filepath.parent / f"{arxiv_id}.pdf"
            paper.download_pdf(filename=str(temp_pdf))
            
            # Extract text from PDF
            text = await self._extract_pdf(temp_pdf)
            
            # Clean up temp file
            temp_pdf.unlink()
            
            return text
            
        except ImportError:
            logger.error("arxiv not installed. Run: pip install arxiv")
            return ""
        except Exception as e:
            logger.error(f"Error downloading arXiv paper: {e}")
            return ""
    
    def _chunk_text(self, text: str) -> List[Dict[str, Any]]:
        """
        Split text into overlapping chunks for embedding.
        
        Args:
            text: Full document text
        
        Returns:
            List of chunks with metadata
        """
        chunks = []
        
        # Split by paragraphs first
        paragraphs = text.split('\n\n')
        
        current_chunk = ""
        chunk_id = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # Check if adding paragraph exceeds chunk size
            if len(current_chunk) + len(para) > self.chunk_size:
                if current_chunk:
                    # Save current chunk
                    chunks.append({
                        "chunk_id": chunk_id,
                        "text": current_chunk.strip(),
                        "length": len(current_chunk)
                    })
                    chunk_id += 1
                    
                    # Start new chunk with overlap
                    words = current_chunk.split()
                    overlap_words = words[-self.chunk_overlap//5:] if len(words) > self.chunk_overlap//5 else words
                    current_chunk = " ".join(overlap_words) + "\n\n" + para
                else:
                    current_chunk = para
            else:
                current_chunk += "\n\n" + para if current_chunk else para
        
        # Add final chunk
        if current_chunk:
            chunks.append({
                "chunk_id": chunk_id,
                "text": current_chunk.strip(),
                "length": len(current_chunk)
            })
        
        return chunks
    
    async def _extract_knowledge(
        self,
        text: str,
        filename: str
    ) -> Dict[str, Any]:
        """
        Extract structured knowledge from document using LLM.
        
        Args:
            text: Full document text
            filename: Document filename
        
        Returns:
            Dictionary with extracted knowledge
        """
        # Truncate text if too long (use first 8000 chars for analysis)
        analysis_text = text[:8000] if len(text) > 8000 else text
        
        # Build prompt for knowledge extraction
        prompt = self._build_extraction_prompt(analysis_text, filename)
        
        # Generate response
        response = await self.generate_response(prompt, temperature=0.2)
        
        # Parse JSON response
        try:
            knowledge = self._parse_knowledge_json(response)
            return knowledge
        except Exception as e:
            logger.error(f"Failed to parse knowledge JSON: {e}")
            return self._get_fallback_knowledge(filename)
    
    def _build_extraction_prompt(self, text: str, filename: str) -> str:
        """Build prompt for knowledge extraction."""
        return f"""You are an expert at analyzing research papers and technical documents.

Extract structured knowledge from the following document excerpt.

Document: {filename}

Excerpt:
{text}

Extract the following information and return as JSON:
1. title: Paper title
2. authors: List of author names
3. year: Publication year (integer)
4. summary: 2-3 sentence summary
5. algorithms: List of algorithms/methods described
6. architectures: List of model architectures
7. hyperparameters: Dictionary of hyperparameters mentioned (e.g., {{"learning_rate": 0.001, "batch_size": 32}})
8. datasets: List of datasets used
9. code_snippets: List of code snippets or pseudocode (if any)
10. implementation_notes: List of implementation details/tips
11. key_equations: List of important equations (LaTeX format)
12. metrics: Dictionary of evaluation metrics mentioned

Return ONLY valid JSON, no markdown formatting.

Example:
{{
  "title": "Deep Residual Learning for Image Recognition",
  "authors": ["Kaiming He", "Xiangyu Zhang"],
  "year": 2015,
  "summary": "Introduces residual learning framework...",
  "algorithms": ["Residual Learning", "Batch Normalization"],
  "architectures": ["ResNet-50", "ResNet-101"],
  "hyperparameters": {{"learning_rate": 0.1, "momentum": 0.9}},
  "datasets": ["ImageNet", "CIFAR-10"],
  "code_snippets": [],
  "implementation_notes": ["Use SGD optimizer", "Weight decay: 0.0001"],
  "key_equations": ["y = F(x) + x"],
  "metrics": {{"top1_accuracy": 75.3, "top5_accuracy": 92.2}}
}}

Now extract from the given document:"""
    
    def _parse_knowledge_json(self, response: str) -> Dict[str, Any]:
        """Parse JSON from LLM response."""
        # Remove markdown code blocks if present
        json_str = response.strip()
        if json_str.startswith("```json"):
            json_str = json_str[7:]
        if json_str.startswith("```"):
            json_str = json_str[3:]
        if json_str.endswith("```"):
            json_str = json_str[:-3]
        json_str = json_str.strip()
        
        # Parse JSON
        knowledge = json.loads(json_str)
        
        return knowledge
    
    def _get_fallback_knowledge(self, filename: str) -> Dict[str, Any]:
        """Return fallback knowledge structure if extraction fails."""
        return {
            "title": filename,
            "authors": [],
            "year": None,
            "summary": "Failed to extract summary",
            "algorithms": [],
            "architectures": [],
            "hyperparameters": {},
            "datasets": [],
            "code_snippets": [],
            "implementation_notes": [],
            "key_equations": [],
            "metrics": {}
        }
    
    async def _index_chunks(
        self,
        chunks: List[Dict[str, Any]],
        document_id: str,
        knowledge: Dict[str, Any]
    ) -> int:
        """
        Index document chunks in ChromaDB using Gemini embeddings.
        
        Args:
            chunks: List of text chunks
            document_id: Unique document ID
            knowledge: Extracted knowledge metadata
        
        Returns:
            Number of chunks indexed
        """
        if not chunks:
            return 0
        
        # Prepare data for ChromaDB
        ids = [f"{document_id}_chunk_{chunk['chunk_id']}" for chunk in chunks]
        documents = [chunk['text'] for chunk in chunks]
        metadatas = [
            {
                "document_id": document_id,
                "chunk_id": chunk['chunk_id'],
                "title": knowledge.get('title', 'Unknown'),
                "authors": str(knowledge.get('authors', [])),
                "year": knowledge.get('year', 0),
                "chunk_length": chunk['length']
            }
            for chunk in chunks
        ]
        
        # Generate embeddings using Gemini API
        logger.info(f"Generating embeddings for {len(chunks)} chunks using Gemini...")
        embeddings = []
        for chunk in chunks:
            embedding = self._generate_embedding(chunk['text'])
            embeddings.append(embedding)
        
        # Add to collection with embeddings
        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )
        
        return len(chunks)
    
    async def query_knowledge(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Query vector database for relevant passages using Gemini embeddings.
        
        Args:
            query: Search query
            top_k: Number of results to return
            filter_metadata: Optional metadata filters
        
        Returns:
            Dictionary with search results
        """
        logger.info(f"Querying knowledge base: '{query}' (top_k={top_k})")
        
        # Generate query embedding using Gemini
        query_embedding = self._generate_query_embedding(query)
        
        # Query ChromaDB with embedding
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filter_metadata
        )
        
        # Format results
        passages = []
        if results and results['documents']:
            for i, doc in enumerate(results['documents'][0]):
                passages.append({
                    "text": doc,
                    "metadata": results['metadatas'][0][i] if results['metadatas'] else {},
                    "distance": results['distances'][0][i] if results['distances'] else 0,
                    "rank": i + 1
                })
        
        logger.info(f"   ✓ Retrieved {len(passages)} passages")
        
        return {
            "status": "success",
            "query": query,
            "num_results": len(passages),
            "passages": passages
        }
    
    def list_indexed_documents(self) -> Dict[str, Any]:
        """
        List all documents in vector database.
        
        Returns:
            Dictionary with document list
        """
        # Get all documents
        results = self.collection.get()
        
        # Extract unique document IDs
        document_ids = set()
        documents_info = {}
        
        if results and results['metadatas']:
            for metadata in results['metadatas']:
                doc_id = metadata.get('document_id')
                if doc_id and doc_id not in document_ids:
                    document_ids.add(doc_id)
                    documents_info[doc_id] = {
                        "title": metadata.get('title', 'Unknown'),
                        "authors": metadata.get('authors', '[]'),
                        "year": metadata.get('year', 0)
                    }
        
        return {
            "status": "success",
            "num_documents": len(document_ids),
            "documents": documents_info
        }
    
    def _generate_document_id(self, filepath: Path) -> str:
        """Generate unique document ID from filepath."""
        import hashlib
        
        # Use filename + timestamp hash
        timestamp = str(filepath.stat().st_mtime)
        content = f"{filepath.name}_{timestamp}"
        doc_id = hashlib.md5(content.encode()).hexdigest()[:12]
        
        return f"doc_{doc_id}"
