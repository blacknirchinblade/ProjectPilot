# -*- coding: utf-8 -*-
"""
RAG Crew - Research Paper Analysis Workflow

This crew orchestrates the RAG (Retrieval-Augmented Generation) workflow:
1. DocumentAnalyzerAgent: Parse and index research papers
2. KnowledgeRetrieverAgent: Search and retrieve relevant knowledge
3. PromptEngineerAgent: Enhance specifications with retrieved knowledge

The RAG Crew improves code generation quality by incorporating domain knowledge
from research papers into the specification phase.

Authors: GaneshS 271123
gmail:ganeshnaik214@gmail.com
"""

from typing import List, Dict, Any, Optional
from crewai import Crew, Agent, Task, Process
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from loguru import logger
from pathlib import Path
import sys

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from agents.document.document_analyzer_agent import DocumentAnalyzerAgent
from agents.prompt_engineering.prompt_engineer_agent import PromptEngineerAgent

from .knowledge_retriever_agent import create_knowledge_retriever_agent


class DocumentAnalyzerInput(BaseModel):
    """Input schema for document analyzer tool."""
    filepath: str = Field(..., description="Path to the research paper (PDF/MD/HTML/arXiv)")


class DocumentAnalyzerTool(BaseTool):
    """
    Custom CrewAI tool for analyzing research papers.
    
    Wraps the DocumentAnalyzerAgent to make it available as a CrewAI tool.
    """
    name: str = "Research Paper Analyzer"
    description: str = """
    Analyze a research paper and extract structured knowledge.
    Use this tool to:
    - Parse PDF, Markdown, HTML, or arXiv papers
    - Extract algorithms, architectures, hyperparameters
    - Store embeddings in vector database for later retrieval
    
    Input: Path to research paper file
    Output: Extracted knowledge (title, authors, algorithms, etc.)
    """
    args_schema: type[BaseModel] = DocumentAnalyzerInput
    
    document_analyzer: Optional[DocumentAnalyzerAgent] = None
    
    def __init__(self, doc_analyzer_instance: Any = None, **kwargs):
        """
        Initialize the document analyzer tool.
        
        Args:
            doc_analyzer_instance: Shared DocumentAnalyzerAgent instance (optional)
        """
        super().__init__(**kwargs)
        
        if doc_analyzer_instance is not None:
            self.document_analyzer = doc_analyzer_instance
        else:
            # Fallback: create new instance
            self.document_analyzer = DocumentAnalyzerAgent(
                vector_db_path="data/vector_db",
                chunk_size=1000,
                chunk_overlap=200
            )
        logger.info("Document Analyzer Tool initialized")
    
    def _run(self, filepath: str) -> str:
        """
        Analyze a research paper.
        
        Args:
            filepath: Path to the paper
            
        Returns:
            Formatted string with extracted knowledge
        """
        try:
            logger.info(f"📄 Analyzing paper: {filepath}")
            
            # Use DocumentAnalyzerAgent's async analyze method
            import asyncio
            loop = asyncio.get_event_loop()
            result = loop.run_until_complete(
                self.document_analyzer.analyze_document(filepath)
            )
            
            # Format results for LLM
            knowledge = result.get('knowledge', {})
            
            formatted_output = f"Successfully analyzed: {filepath}\n\n"
            formatted_output += f"📊 EXTRACTED KNOWLEDGE:\n\n"
            formatted_output += f"Title: {knowledge.get('title', 'Unknown')}\n"
            formatted_output += f"Authors: {', '.join(knowledge.get('authors', []))}\n"
            formatted_output += f"Year: {knowledge.get('year', 'Unknown')}\n\n"
            
            formatted_output += f"Summary: {knowledge.get('summary', 'N/A')}\n\n"
            
            if knowledge.get('algorithms'):
                formatted_output += f"Algorithms ({len(knowledge['algorithms'])}):\n"
                for algo in knowledge['algorithms']:
                    formatted_output += f"  - {algo}\n"
                formatted_output += "\n"
            
            if knowledge.get('architectures'):
                formatted_output += f"Architectures ({len(knowledge['architectures'])}):\n"
                for arch in knowledge['architectures']:
                    formatted_output += f"  - {arch}\n"
                formatted_output += "\n"
            
            if knowledge.get('hyperparameters'):
                formatted_output += f"Hyperparameters:\n"
                for key, value in knowledge['hyperparameters'].items():
                    formatted_output += f"  - {key}: {value}\n"
                formatted_output += "\n"
            
            if knowledge.get('datasets'):
                formatted_output += f"Datasets: {', '.join(knowledge['datasets'])}\n\n"
            
            formatted_output += f"📦 Indexed {result.get('num_chunks', 0)} chunks in vector database\n"
            
            logger.info(f"   ✅ Paper analyzed successfully")
            return formatted_output
            
        except Exception as e:
            error_msg = f"Error analyzing paper: {str(e)}"
            logger.error(error_msg)
            return error_msg


def create_document_analyzer_agent(
    doc_analyzer_instance: Any = None,
    llm: Any = None,
    verbose: bool = True
) -> Agent:
    """
    Create a DocumentAnalyzerAgent as a CrewAI Agent.
    
    Args:
        doc_analyzer_instance: Shared DocumentAnalyzerAgent instance (optional)
        llm: Language model to use (defaults to Gemini)
        verbose: Enable verbose logging
        
    Returns:
        CrewAI Agent configured for document analysis
    """
    # Create custom tool with shared instance
    doc_analyzer_tool = DocumentAnalyzerTool(doc_analyzer_instance=doc_analyzer_instance)
    
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
        role="Research Paper Analysis Expert",
        goal="""
        Parse and extract structured knowledge from research papers including
        algorithms, model architectures, hyperparameters, datasets, and
        implementation details. Store this knowledge in a vector database
        for efficient retrieval during code generation.
        """,
        backstory="""
        You are a meticulous research analyst with expertise in machine learning
        and deep learning papers. You excel at reading academic papers and
        extracting actionable information that developers need to implement
        the described methods. You understand how to identify key components:
        algorithms, architectures, training procedures, hyperparameters, and
        datasets. Your work enables accurate code generation from research papers.
        """,
        tools=[doc_analyzer_tool],
        llm=llm,
        verbose=verbose,
        allow_delegation=False,
        memory=False  # Disable to avoid OpenAI dependency
    )
    
    logger.info("✅ DocumentAnalyzerAgent (CrewAI) created successfully")
    return agent


def create_rag_crew(
    vector_db_path: str = "data/vector_db",
    llm: Any = None,
    verbose: bool = True
) -> Crew:
    """
    Create the RAG Crew for research paper processing and knowledge retrieval.
    
    This crew orchestrates:
    1. Document analysis (parse papers → extract knowledge → store in vector DB)
    2. Knowledge retrieval (semantic search for relevant passages)
    3. Specification enhancement (integrate knowledge into code specs)
    
    Args:
        vector_db_path: Path to ChromaDB database
        llm: Language model to use (defaults to Gemini)
        verbose: Enable verbose logging
        
    Returns:
        Configured RAG Crew ready to process papers and enhance specifications
    
    Example:
        >>> rag_crew = create_rag_crew()
        >>> result = rag_crew.kickoff(inputs={
        ...     "papers": ["ResNet_2015.pdf"],
        ...     "user_query": "Implement ResNet-50 for CIFAR-10"
        ... })
    """
    # Create agents
    logger.info("🚀 Creating RAG Crew...")
    
    # Create a SHARED DocumentAnalyzerAgent instance for both tools
    from src.agents.document.document_analyzer_agent import DocumentAnalyzerAgent
    
    shared_doc_analyzer = DocumentAnalyzerAgent(
        vector_db_path=vector_db_path
    )
    
    doc_analyzer = create_document_analyzer_agent(
        doc_analyzer_instance=shared_doc_analyzer,
        llm=llm,
        verbose=verbose
    )
    
    knowledge_retriever = create_knowledge_retriever_agent(
        doc_analyzer_instance=shared_doc_analyzer,
        llm=llm,
        verbose=verbose
    )
    
    # Note: PromptEngineerAgent enhancement will be in Phase 2
    # For now, we focus on document analysis + retrieval
    
    # Define tasks
    analyze_papers_task = Task(
        description="""
        Analyze all research papers provided in {papers}.
        
        For each paper:
        1. Parse the document (PDF/Markdown/HTML/arXiv)
        2. Extract structured knowledge:
           - Title, authors, publication year
           - Algorithms and methodologies
           - Model architectures
           - Hyperparameters and training details
           - Datasets used
           - Code snippets or pseudocode
        3. Store embeddings in the vector database
        
        Papers to analyze: {papers}
        """,
        expected_output="""
        A summary report containing:
        - Number of papers analyzed
        - List of extracted algorithms
        - List of model architectures
        - Key hyperparameters identified
        - Number of chunks indexed in vector database
        """,
        agent=doc_analyzer
    )
    
    retrieve_knowledge_task = Task(
        description="""
        Based on the user query: "{user_query}"
        
        Search the vector database for the most relevant information:
        1. Identify key concepts in the user query
        2. Perform semantic search using Gemini embeddings
        3. Retrieve top 10 most relevant passages
        4. Rank passages by relevance
        5. Organize findings by category:
           - Algorithms
           - Architectures
           - Hyperparameters
           - Implementation details
        
        Focus on information that will help generate accurate code.
        """,
        expected_output="""
        A structured knowledge report containing:
        - Top 10 relevant passages with sources
        - Categorized findings (algorithms, architectures, etc.)
        - Specific implementation recommendations
        - Hyperparameter suggestions
        - Dataset information
        """,
        agent=knowledge_retriever,
        context=[analyze_papers_task]  # Depends on Task 1
    )
    
    # Create crew
    crew = Crew(
        agents=[doc_analyzer, knowledge_retriever],
        tasks=[analyze_papers_task, retrieve_knowledge_task],
        process=Process.sequential,  # Run tasks in order
        verbose=verbose,
        memory=False  # Disable memory to avoid OpenAI API key requirement
        # Note: Using custom Gemini embeddings in tools instead
    )
    
    logger.info("✅ RAG Crew created successfully")
    logger.info(f"   - Agents: DocumentAnalyzer, KnowledgeRetriever")
    logger.info(f"   - Tasks: analyze_papers → retrieve_knowledge")
    logger.info(f"   - Process: Sequential")
    logger.info(f"   - Vector DB: {vector_db_path}")
    
    return crew


# Example usage
if __name__ == "__main__":
    """
    Test the RAG Crew with a sample workflow.
    """
    import os
    
    # Set API key (should be in environment)
    if not os.getenv("GOOGLE_API_KEY"):
        print("⚠️  Warning: GOOGLE_API_KEY not set in environment")
        print("   Set it with: export GOOGLE_API_KEY='your-key-here'")
        exit(1)
    
    # Create RAG Crew
    rag_crew = create_rag_crew(
        vector_db_path="data/test_vector_db",
        verbose=True
    )
    
    # Test inputs
    test_inputs = {
        "papers": ["test_paper.md"],  # Use the test paper from earlier
        "user_query": "Implement ResNet-50 for image classification with best practices"
    }
    
    print("\n" + "=" * 70)
    print("TESTING RAG CREW")
    print("=" * 70)
    print(f"Papers: {test_inputs['papers']}")
    print(f"Query: {test_inputs['user_query']}")
    print("=" * 70 + "\n")
    
    # Execute crew
    result = rag_crew.kickoff(inputs=test_inputs)
    
    print("\n" + "=" * 70)
    print("RAG CREW RESULT")
    print("=" * 70)
    print(result)
