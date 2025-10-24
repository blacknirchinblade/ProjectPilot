"""
Simple RAG Orchestrator - Direct Agent Orchestration Without CrewAI

This module provides a lightweight alternative to CrewAI for orchestrating
the RAG workflow. It directly uses DocumentAnalyzerAgent without LLM complexity.

Author: GaneshS 271123
gmail:ganeshnaik214@gmail.com
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import sys

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.agents.document.document_analyzer_agent import DocumentAnalyzerAgent

logger = logging.getLogger(__name__)


@dataclass
class RAGResult:
    """Result from RAG workflow execution."""
    papers_analyzed: int
    chunks_indexed: int
    knowledge_retrieved: List[Dict[str, Any]]
    query: str
    success: bool
    error: Optional[str] = None


class SimpleRAGOrchestrator:
    """
    Simple orchestrator for RAG workflow without CrewAI dependency.
    
    This orchestrator:
    1. Analyzes research papers using DocumentAnalyzerAgent
    2. Extracts and stores knowledge in ChromaDB
    3. Retrieves relevant passages for user queries
    
    Benefits over CrewAI:
    - No LLM compatibility issues
    - Direct control over workflow
    - Faster execution
    - Simpler debugging
    """
    
    def __init__(
        self,
        vector_db_path: str = "data/vector_db",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        verbose: bool = True
    ):
        """
        Initialize the RAG orchestrator.
        
        Args:
            vector_db_path: Path to ChromaDB database
            chunk_size: Size of text chunks for embedding
            chunk_overlap: Overlap between chunks
            verbose: Enable verbose logging
        """
        self.vector_db_path = vector_db_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.verbose = verbose
        
        # Initialize document analyzer
        self.doc_analyzer = DocumentAnalyzerAgent(
            vector_db_path=vector_db_path,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        logger.info(f"✅ Simple RAG Orchestrator initialized")
        logger.info(f"   - Vector DB: {vector_db_path}")
        logger.info(f"   - Chunk size: {chunk_size}")
        logger.info(f"   - Embeddings: Gemini text-embedding-004 (768-dim)")
    
    def execute(
        self,
        papers: List[str],
        user_query: str,
        top_k: int = 10
    ) -> RAGResult:
        """
        Execute the RAG workflow.
        
        Workflow:
        1. Analyze papers → Extract knowledge → Store embeddings
        2. Search vector DB for relevant passages
        3. Return organized results
        
        Args:
            papers: List of paper file paths
            user_query: User's question/query
            top_k: Number of passages to retrieve
            
        Returns:
            RAGResult with analysis and retrieval results
        """
        try:
            logger.info("=" * 70)
            logger.info("🚀 STARTING RAG WORKFLOW")
            logger.info("=" * 70)
            
            # Phase 1: Analyze papers
            logger.info(f"\n📄 Phase 1: Analyzing {len(papers)} paper(s)...")
            papers_analyzed = 0
            total_chunks = 0
            
            for paper_path in papers:
                logger.info(f"\n   Analyzing: {paper_path}")
                result = self.doc_analyzer.analyze_document(paper_path)
                
                if result.get("success"):
                    papers_analyzed += 1
                    chunks = result.get("chunks_indexed", 0)
                    total_chunks += chunks
                    
                    logger.info(f"   ✅ Success!")
                    logger.info(f"      - Algorithms: {len(result.get('algorithms', []))}")
                    logger.info(f"      - Architectures: {len(result.get('architectures', []))}")
                    logger.info(f"      - Hyperparameters: {len(result.get('hyperparameters', []))}")
                    logger.info(f"      - Chunks indexed: {chunks}")
                else:
                    logger.error(f"   ❌ Failed to analyze {paper_path}")
            
            logger.info(f"\n   📊 Summary:")
            logger.info(f"      - Papers analyzed: {papers_analyzed}/{len(papers)}")
            logger.info(f"      - Total chunks indexed: {total_chunks}")
            
            # Phase 2: Retrieve knowledge
            logger.info(f"\n🔍 Phase 2: Retrieving knowledge for query...")
            logger.info(f"   Query: {user_query}")
            
            passages = self.doc_analyzer.query_knowledge(
                query=user_query,
                top_k=top_k
            )
            
            logger.info(f"   ✅ Retrieved {len(passages)} relevant passages")
            
            # Organize results by relevance
            if passages:
                logger.info(f"\n   📋 Top passages:")
                for i, passage in enumerate(passages[:5], 1):
                    relevance = (1 - passage.get('distance', 1)) * 100
                    logger.info(f"      {i}. Relevance: {relevance:.1f}%")
                    logger.info(f"         Source: {passage.get('metadata', {}).get('title', 'Unknown')}")
                    preview = passage.get('text', '')[:100].replace('\n', ' ')
                    logger.info(f"         Preview: {preview}...")
            
            logger.info("\n" + "=" * 70)
            logger.info("✅ RAG WORKFLOW COMPLETE")
            logger.info("=" * 70)
            
            return RAGResult(
                papers_analyzed=papers_analyzed,
                chunks_indexed=total_chunks,
                knowledge_retrieved=passages,
                query=user_query,
                success=True
            )
            
        except Exception as e:
            logger.error(f"❌ RAG workflow failed: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            
            return RAGResult(
                papers_analyzed=0,
                chunks_indexed=0,
                knowledge_retrieved=[],
                query=user_query,
                success=False,
                error=str(e)
            )
    
    def get_indexed_documents(self) -> List[Dict[str, Any]]:
        """
        Get list of all documents indexed in the vector database.
        
        Returns:
            List of document metadata
        """
        try:
            # Query ChromaDB for all documents
            collection = self.doc_analyzer.collection
            if collection:
                results = collection.get()
                
                # Extract unique documents
                documents = {}
                if results and 'metadatas' in results:
                    for metadata in results['metadatas']:
                        if metadata:
                            doc_id = metadata.get('document_id')
                            if doc_id and doc_id not in documents:
                                documents[doc_id] = {
                                    'title': metadata.get('title', 'Unknown'),
                                    'authors': metadata.get('authors', []),
                                    'year': metadata.get('year', 'Unknown'),
                                    'source': metadata.get('source', 'Unknown')
                                }
                
                return list(documents.values())
            
            return []
            
        except Exception as e:
            logger.error(f"Error getting indexed documents: {str(e)}")
            return []
    
    def clear_database(self):
        """Clear all documents from the vector database."""
        try:
            if self.doc_analyzer.chroma_client:
                collection_name = f"research_papers"
                self.doc_analyzer.chroma_client.delete_collection(name=collection_name)
                logger.info(f"✅ Cleared vector database: {collection_name}")
                
                # Reinitialize
                self.doc_analyzer._init_vector_db()
        except Exception as e:
            logger.error(f"Error clearing database: {str(e)}")


def create_simple_rag_orchestrator(
    vector_db_path: str = "data/vector_db",
    verbose: bool = True
) -> SimpleRAGOrchestrator:
    """
    Factory function to create a Simple RAG Orchestrator.
    
    Args:
        vector_db_path: Path to ChromaDB database
        verbose: Enable verbose logging
        
    Returns:
        Configured SimpleRAGOrchestrator
        
    Example:
        >>> orchestrator = create_simple_rag_orchestrator()
        >>> result = orchestrator.execute(
        ...     papers=["ResNet_2015.pdf"],
        ...     user_query="How to implement ResNet-50?"
        ... )
        >>> print(f"Found {len(result.knowledge_retrieved)} passages")
    """
    return SimpleRAGOrchestrator(
        vector_db_path=vector_db_path,
        verbose=verbose
    )


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(name)s:%(funcName)s:%(lineno)d - %(message)s'
    )
    
    # Create orchestrator
    orchestrator = create_simple_rag_orchestrator(vector_db_path="data/test_vector_db")
    
    # Execute RAG workflow
    result = orchestrator.execute(
        papers=["test_paper.md"],
        user_query="How to implement ResNet-50 for image classification? What hyperparameters should I use?"
    )
    
    if result.success:
        print(f"\n✅ Success!")
        print(f"   - Papers analyzed: {result.papers_analyzed}")
        print(f"   - Chunks indexed: {result.chunks_indexed}")
        print(f"   - Passages retrieved: {len(result.knowledge_retrieved)}")
        
        # Show top passages
        print(f"\n📋 Top 3 Relevant Passages:")
        for i, passage in enumerate(result.knowledge_retrieved[:3], 1):
            relevance = (1 - passage.get('distance', 1)) * 100
            print(f"\n{i}. Relevance: {relevance:.1f}%")
            print(f"   Text: {passage.get('text', '')[:200]}...")
    else:
        print(f"\n❌ Failed: {result.error}")
