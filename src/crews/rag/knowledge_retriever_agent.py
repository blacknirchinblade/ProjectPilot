# -*- coding: utf-8 -*-
"""
Knowledge Retriever Agent - CrewAI RAG Component

This agent performs semantic search in the ChromaDB vector database to retrieve
relevant passages from research papers. It uses Gemini embeddings for consistency
with the DocumentAnalyzerAgent.

Features:
- Semantic similarity search using Gemini embeddings
- Multi-query fusion for better recall
- Context ranking and relevance scoring
- Citation tracking

Author: AutoCoder Team
Date: October 19, 2025
"""

from typing import Dict, List, Any, Optional
from crewai import Agent
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from loguru import logger
import sys
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from agents.document.document_analyzer_agent import DocumentAnalyzerAgent


class ChromaDBSearchInput(BaseModel):
    """Input schema for ChromaDB search tool."""
    query: str = Field(..., description="Search query to find relevant passages")
    top_k: int = Field(default=5, description="Number of results to return")


class ChromaDBSearchTool(BaseTool):
    """
    Custom CrewAI tool for searching ChromaDB vector database.
    
    This tool uses the DocumentAnalyzerAgent's query_knowledge method
    to perform semantic search using Gemini embeddings.
    """
    name: str = "ChromaDB Knowledge Search"
    description: str = """
    Search the vector database for relevant passages from research papers.
    Use this tool to find:
    - Algorithm descriptions and implementations
    - Model architecture details
    - Hyperparameter recommendations
    - Dataset information
    - Code snippets and examples
    - Training methodologies
    
    Input: A natural language query describing what you're looking for
    Output: Top-K most relevant passages with metadata (title, authors, distance)
    """
    args_schema: type[BaseModel] = ChromaDBSearchInput
    
    document_analyzer: Optional[DocumentAnalyzerAgent] = None
    
    def __init__(self, doc_analyzer_instance: Any = None, **kwargs):
        """
        Initialize the ChromaDB search tool.
        
        Args:
            doc_analyzer_instance: Shared DocumentAnalyzerAgent instance (optional)
        """
        super().__init__(**kwargs)
        
        # Use shared instance or create new one
        if doc_analyzer_instance is not None:
            self.document_analyzer = doc_analyzer_instance
        else:
            # Fallback: create new instance
            self.document_analyzer = DocumentAnalyzerAgent(
                vector_db_path="data/vector_db",
                chunk_size=1000,
                chunk_overlap=200
            )
        logger.info("ChromaDB Search Tool initialized")
    
    def _run(self, query: str, top_k: int = 5) -> str:
        """
        Search ChromaDB for relevant passages.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            Formatted string with search results
        """
        try:
            logger.info(f"🔍 Searching ChromaDB: '{query}' (top_k={top_k})")
            
            # Use DocumentAnalyzerAgent's async query method
            import asyncio
            loop = asyncio.get_event_loop()
            result = loop.run_until_complete(
                self.document_analyzer.query_knowledge(query, top_k=top_k)
            )
            
            # Format results for LLM
            if not result.get('passages'):
                return f"No relevant passages found for query: '{query}'"
            
            formatted_output = f"Found {len(result['passages'])} relevant passages:\n\n"
            
            for passage in result['passages']:
                formatted_output += f"--- Passage {passage['rank']} ---\n"
                formatted_output += f"Text: {passage['text'][:500]}...\n"
                formatted_output += f"Source: {passage['metadata'].get('title', 'Unknown')}\n"
                formatted_output += f"Authors: {passage['metadata'].get('authors', 'Unknown')}\n"
                formatted_output += f"Relevance: {1 - passage['distance']:.2%}\n\n"
            
            logger.info(f"   ✅ Retrieved {len(result['passages'])} passages")
            return formatted_output
            
        except Exception as e:
            error_msg = f"Error searching ChromaDB: {str(e)}"
            logger.error(error_msg)
            return error_msg


class DocumentListInput(BaseModel):
    """Input schema for document listing tool."""
    pass  # No input needed


class DocumentListTool(BaseTool):
    """
    Custom CrewAI tool for listing indexed documents in ChromaDB.
    """
    name: str = "List Indexed Documents"
    description: str = """
    List all research papers currently indexed in the vector database.
    Use this tool to:
    - See what papers are available for search
    - Get document metadata (title, authors, year)
    - Understand the knowledge base coverage
    
    Input: None
    Output: List of all indexed documents with metadata
    """
    args_schema: type[BaseModel] = DocumentListInput
    
    document_analyzer: Optional[DocumentAnalyzerAgent] = None
    
    def __init__(self, doc_analyzer_instance: Any = None, **kwargs):
        """
        Initialize the document listing tool.
        
        Args:
            doc_analyzer_instance: Shared DocumentAnalyzerAgent instance (optional)
        """
        super().__init__(**kwargs)
        
        # Use shared instance or create new one
        if doc_analyzer_instance is not None:
            self.document_analyzer = doc_analyzer_instance
        else:
            # Fallback: create new instance
            self.document_analyzer = DocumentAnalyzerAgent(
                vector_db_path="data/vector_db"
            )
    
    def _run(self) -> str:
        """
        List all indexed documents.
        
        Returns:
            Formatted string with document list
        """
        try:
            logger.info("📚 Listing indexed documents...")
            
            result = self.document_analyzer.list_indexed_documents()
            
            if not result.get('documents'):
                return "No documents indexed in the database yet."
            
            formatted_output = f"Found {result['count']} indexed document(s):\n\n"
            
            for doc_id, metadata in result['documents'].items():
                formatted_output += f"--- Document: {doc_id} ---\n"
                formatted_output += f"Title: {metadata.get('title', 'Unknown')}\n"
                formatted_output += f"Authors: {metadata.get('authors', 'Unknown')}\n"
                formatted_output += f"Year: {metadata.get('year', 'Unknown')}\n\n"
            
            logger.info(f"   ✅ Found {result['count']} documents")
            return formatted_output
            
        except Exception as e:
            error_msg = f"Error listing documents: {str(e)}"
            logger.error(error_msg)
            return error_msg


def create_knowledge_retriever_agent(
    doc_analyzer_instance: Any = None,
    llm: Any = None,
    verbose: bool = True
) -> Agent:
    """
    Create a KnowledgeRetrieverAgent as a CrewAI Agent.
    
    This agent specializes in semantic search and knowledge retrieval from
    the vector database populated by DocumentAnalyzerAgent.
    
    Args:
        doc_analyzer_instance: Shared DocumentAnalyzerAgent instance (optional)
        llm: Language model to use (defaults to Gemini)
        verbose: Enable verbose logging
        
    Returns:
        CrewAI Agent configured for knowledge retrieval
    
    Example:
        >>> from crewai import Task
        >>> retriever = create_knowledge_retriever_agent()
        >>> task = Task(
        ...     description="Find information about ResNet-50 architecture",
        ...     agent=retriever,
        ...     expected_output="Architecture details with layer specifications"
        ... )
    """
    # Create custom tools with shared instance
    chromadb_search_tool = ChromaDBSearchTool(doc_analyzer_instance=doc_analyzer_instance)
    document_list_tool = DocumentListTool(doc_analyzer_instance=doc_analyzer_instance)
    
    # Configure LLM (default to Gemini)
    if llm is None:
        from langchain_google_genai import ChatGoogleGenerativeAI
        import os
        
        # Use LangChain's Gemini integration (proven to work)
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.2,
            convert_system_message_to_human=True
        )
    
    # Create CrewAI Agent
    agent = Agent(
        role="Knowledge Retrieval Specialist",
        goal="""
        Retrieve the most relevant information from research papers to help 
        generate high-quality code implementations. Focus on:
        - Algorithm descriptions and pseudocode
        - Model architecture specifications
        - Hyperparameter recommendations
        - Dataset preparation details
        - Implementation best practices
        """,
        backstory="""
        You are an expert research assistant with deep knowledge of machine learning
        and software engineering. You excel at finding relevant information from
        academic papers and technical documentation. Your semantic search skills
        help developers implement papers correctly by providing precise, contextual
        information about algorithms, architectures, and best practices.
        
        You use advanced vector search with Gemini embeddings to find the most
        relevant passages, and you understand how to rank and filter results to
        provide maximum value to code generation tasks.
        """,
        tools=[chromadb_search_tool, document_list_tool],
        llm=llm,
        verbose=verbose,
        allow_delegation=False,  # This agent doesn't delegate to others
        memory=False  # Disable to avoid OpenAI dependency
    )
    
    vector_db_path = doc_analyzer_instance.vector_db_path if doc_analyzer_instance else "data/vector_db"
    
    logger.info("✅ KnowledgeRetrieverAgent (CrewAI) created successfully")
    logger.info(f"   - Tools: ChromaDB Search, Document List")
    logger.info(f"   - Vector DB: {vector_db_path}")
    logger.info(f"   - LLM: Gemini 2.5-pro")
    
    return agent


# Example usage
if __name__ == "__main__":
    """
    Test the KnowledgeRetrieverAgent with sample queries.
    """
    from crewai import Task, Crew
    
    # Create agent
    retriever_agent = create_knowledge_retriever_agent(
        vector_db_path="data/test_vector_db",
        verbose=True
    )
    
    # Create test task
    test_task = Task(
        description="""
        Search for information about ResNet-50 architecture:
        1. What are the key architectural components?
        2. What are the recommended hyperparameters for training?
        3. What datasets is it typically trained on?
        
        Provide detailed information from the indexed research papers.
        """,
        expected_output="""
        A comprehensive summary including:
        - Architecture details (layers, skip connections, etc.)
        - Hyperparameters (learning rate, batch size, optimizer, etc.)
        - Datasets (ImageNet, CIFAR-10, etc.)
        - Any code snippets or implementation notes
        """,
        agent=retriever_agent
    )
    
    # Create single-agent crew for testing
    test_crew = Crew(
        agents=[retriever_agent],
        tasks=[test_task],
        verbose=True
    )
    
    # Execute
    print("\n" + "=" * 70)
    print("TESTING KNOWLEDGE RETRIEVER AGENT")
    print("=" * 70 + "\n")
    
    result = test_crew.kickoff()
    
    print("\n" + "=" * 70)
    print("RESULT")
    print("=" * 70)
    print(result)
