"""
GenAI Agent - Generates GenAI/LLM Projects

This agent creates complete GenAI applications including:
- RAG (Retrieval-Augmented Generation) systems
- LLM fine-tuning pipelines (LoRA, QLoRA, PEFT)
- LLM agent systems (ReAct, function calling, multi-agent)
- Prompt engineering frameworks
- LLM evaluation and benchmarking
- Vector database integration (ChromaDB, Pinecone, Weaviate, FAISS)
- Embeddings and semantic search

Authors: GaneshS 271123
gmail:ganeshnaik214@gmail.com
"""

import asyncio
from typing import Dict, List, Optional, Any
from pathlib import Path
import json
from loguru import logger

from src.agents.base_agent import BaseAgent
from src.llm.gemini_client import GeminiClient
from src.llm.prompt_manager import PromptManager


class GenAIAgent(BaseAgent):
    """
    GenAIAgent generates complete GenAI/LLM applications and pipelines.
    
    Features:
    - RAG system generation (embeddings + vector DB + retrieval)
    - LLM fine-tuning scripts (LoRA, QLoRA, full fine-tuning)
    - LLM agents (ReAct, function calling, multi-agent orchestration)
    - Prompt engineering templates and optimization
    - LLM evaluation frameworks
    - Integration with OpenAI, Anthropic, HuggingFace, Ollama
    """
    
    def __init__(
        self,
        llm_client: Optional[GeminiClient] = None,
        prompt_manager: Optional[PromptManager] = None
    ):
        """
        Initialize the GenAI Agent.
        
        Args:
            llm_client: The initialized Gemini client.
            prompt_manager: The initialized Prompt manager.
        """
        super().__init__(
            name="genai_agent",
            role="Expert GenAI and LLM Application Developer",
            agent_type="coding", # Use 'coding' temp
            llm_client=llm_client,
            prompt_manager=prompt_manager
        )
        
        # Project type templates (helper methods)
        self.templates = {
            "rag_system": self._generate_rag_system,
            "llm_finetuning": self._generate_llm_finetuning,
            "llm_agent": self._generate_llm_agent,
            "prompt_engineering": self._generate_prompt_engineering,
            "llm_evaluation": self._generate_llm_evaluation,
            "semantic_search": self._generate_semantic_search,
            "chatbot": self._generate_chatbot,
        }
        logger.info(f"{self.name} initialized for GenAI project generation")
    
    async def generate_genai_project(
        self,
        project_type: str,
        project_name: str,
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, str]:
        """
        Generate a complete GenAI project.
        
        Args:
            project_type: Type of GenAI project (rag_system, llm_finetuning, etc.)
            project_name: Name of the project
            config: Project configuration
        
        Returns:
            Dictionary with generated files: {filename: code}
        """

        # Ensure clients are available if not passed during init
        if not self.llm_client or not self.prompt_manager:
            logger.warning("Re-initializing LLM client for GenAIAgent. Should be passed in __init__.")
            self.llm_client = GeminiClient()
            self.prompt_manager = PromptManager()
            
        config = config or {}
        
        # Get template generator
        template_func = self.templates.get(project_type, self._generate_rag_system)
        
        # Generate project files
        result = template_func(project_name, config)
        
        # Add common files
        result.update(self._generate_common_files(project_name, project_type, config))
        
        return result
    
    def _generate_rag_system(
        self,
        project_name: str,
        config: Dict[str, Any]
    ) -> Dict[str, str]:
        """Generate RAG (Retrieval-Augmented Generation) system."""
        
        llm_provider = config.get("llm_provider", "OpenAI")
        vector_db = config.get("vector_db", "Chroma")
        embedding_model = config.get("embedding_model", "OpenAI")
        
        files = {}
        

        files["rag/pipeline.py"] = f'''"""
RAG Pipeline for {project_name}

Retrieval-Augmented Generation system using {llm_provider} and {vector_db}.
"""

from typing import List, Dict, Optional
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.vectorstores import Chroma, Pinecone, Weaviate, FAISS
from langchain.llms import OpenAI, Anthropic
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from pathlib import Path
import os


class RAGPipeline:
    """
    Retrieval-Augmented Generation pipeline.
    
    Components:
    - Document loading and chunking
    - Embeddings generation
    - Vector store (similarity search)
    - LLM for answer generation
    - Retrieval QA chain
    """
    
    def __init__(
        self,
        docs_path: Optional[str] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        top_k: int = 3,
        temperature: float = 0.0,
        embedding_model: str = "OpenAI",
        llm_provider: str = "OpenAI",
        vector_db: str = "Chroma"
    ):
        """
        Initialize RAG pipeline.
        
        Args:
            docs_path: Path to documents directory
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            top_k: Number of documents to retrieve
            temperature: LLM temperature
            embedding_model: Embedding model type
            llm_provider: LLM provider
            vector_db: Vector database type
        """
        self.docs_path = docs_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.temperature = temperature
        self.embedding_model = embedding_model
        self.llm_provider = llm_provider
        self.vector_db = vector_db
        
        # Initialize components
        self.embeddings = self._init_embeddings()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
        self.vectorstore = None
        self.llm = self._init_llm()
        self.qa_chain = None
        
        # Load documents if path provided
        if docs_path:
            self.load_documents(docs_path)

    def _init_embeddings(self):
        """Initialize embeddings model."""
        if self.embedding_model == "OpenAI":
            return OpenAIEmbeddings(model="text-embedding-ada-002")
        else:
            return HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    def _init_llm(self):
        """Initialize LLM."""
        if self.llm_provider == "OpenAI":
            return OpenAI(temperature=self.temperature, model="gpt-3.5-turbo-instruct")
        else:
            return Anthropic(temperature=self.temperature, model="claude-3-sonnet-20240229")
    
    def load_documents(self, docs_path: str) -> int:
        """
        Load and process documents from directory.
        
        Args:
            docs_path: Path to documents
        
        Returns:
            Number of chunks created
        """
        docs_path = Path(docs_path)
        
        if not docs_path.exists():
            raise ValueError(f"Documents path does not exist: {{docs_path}}")
        
        # Load documents based on file types
        documents = []
        
        # Load PDFs
        pdf_loader = DirectoryLoader(
            str(docs_path),
            glob="**/*.pdf",
            loader_cls=PyPDFLoader
        )
        documents.extend(pdf_loader.load())
        
        # Load text files
        text_loader = DirectoryLoader(
            str(docs_path),
            glob="**/*.txt",
            loader_cls=TextLoader
        )
        documents.extend(text_loader.load())
        
        # Split into chunks
        chunks = self.text_splitter.split_documents(documents)
        
        # Create vector store
        self.vectorstore = {vector_db}.from_documents(
            documents=chunks,
            embedding=self.embeddings
        )
        
        # Create QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(
                search_kwargs={{"k": self.top_k}}
            ),
            return_source_documents=True
        )
        
        print(f"✓ Loaded {{len(documents)}} documents, created {{len(chunks)}} chunks")
        return len(chunks)
    
    def add_documents(self, documents: List[str]) -> int:
        """
        Add new documents to the vector store.
        
        Args:
            documents: List of document texts
        
        Returns:
            Number of chunks added
        """
        if not self.vectorstore:
            # Create new vector store
            from langchain_core.documents import Document
            docs = [Document(page_content=doc) for doc in documents]
            chunks = self.text_splitter.split_documents(docs)
            
            self.vectorstore = {vector_db}.from_documents(
                documents=chunks,
                embedding=self.embeddings
            )
        else:
            # Add to existing vector store
            from langchain_core.documents import Document
            docs = [Document(page_content=doc) for doc in documents]
            chunks = self.text_splitter.split_documents(docs)
            
            self.vectorstore.add_documents(chunks)
        
        return len(chunks)
    
    def query(self, question: str) -> Dict[str, Any]:
        """
        Query the RAG system.
        
        Args:
            question: User question
        
        Returns:
            Dictionary with answer and source documents
        """
        if not self.qa_chain:
            raise ValueError("No documents loaded. Call load_documents() first.")
        
        result = self.qa_chain({{"query": question}})
        
        return {{
            "question": question,
            "answer": result["result"],
            "sources": [
                {{
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "relevance_score": 0.0  # TODO: Add scoring
                }}
                for doc in result.get("source_documents", [])
            ]
        }}
    
    def similarity_search(self, query: str, k: int = 5) -> List[Dict]:
        """
        Perform similarity search without LLM.
        
        Args:
            query: Search query
            k: Number of results
        
        Returns:
            List of similar documents
        """
        if not self.vectorstore:
            raise ValueError("No documents loaded.")
        
        docs = self.vectorstore.similarity_search(query, k=k)
        
        return [
            {{
                "content": doc.page_content,
                "metadata": doc.metadata
            }}
            for doc in docs
        ]


# Example usage
if __name__ == "__main__":
    # Initialize RAG pipeline
    rag = RAGPipeline(
        docs_path="./documents",
        chunk_size=1000,
        top_k=3
    )
    
    # Query
    result = rag.query("What is the main topic of the documents?")
    
    print(f"Question: {{result['question']}}")
    print(f"Answer: {{result['answer']}}")
    print(f"\\nSources ({{len(result['sources'])}}):")
    for i, source in enumerate(result['sources'], 1):
        print(f"  {{i}}. {{source['content'][:200]}}...")
'''
        
        # Document loader utility
        files["rag/document_loader.py"] = '''"""
Document Loader Utilities

Supports multiple file formats: PDF, DOCX, TXT, MD, HTML, CSV
"""

from typing import List
from pathlib import Path
from langchain.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredMarkdownLoader,
    UnstructuredHTMLLoader,
    CSVLoader
)
from langchain_core.documents import Document


class DocumentLoader:
    """Utility for loading various document formats."""
    
    SUPPORTED_FORMATS = {
        '.pdf': PyPDFLoader,
        '.txt': TextLoader,
        '.md': UnstructuredMarkdownLoader,
        '.docx': UnstructuredWordDocumentLoader,
        '.html': UnstructuredHTMLLoader,
        '.csv': CSVLoader
    }
    
    @classmethod
    def load_file(cls, file_path: str) -> List[Document]:
        """
        Load a single file.
        
        Args:
            file_path: Path to file
        
        Returns:
            List of Document objects
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        ext = path.suffix.lower()
        
        if ext not in cls.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format: {ext}")
        
        loader_cls = cls.SUPPORTED_FORMATS[ext]
        loader = loader_cls(str(path))
        
        return loader.load()
    
    @classmethod
    def load_directory(cls, dir_path: str, recursive: bool = True) -> List[Document]:
        """
        Load all supported files from a directory.
        
        Args:
            dir_path: Path to directory
            recursive: Search subdirectories
        
        Returns:
            List of all loaded documents
        """
        path = Path(dir_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Directory not found: {dir_path}")
        
        documents = []
        pattern = "**/*" if recursive else "*"
        
        for file_path in path.glob(pattern):
            if file_path.is_file() and file_path.suffix.lower() in cls.SUPPORTED_FORMATS:
                try:
                    docs = cls.load_file(str(file_path))
                    documents.extend(docs)
                    print(f"✓ Loaded: {file_path.name}")
                except Exception as e:
                    print(f"✗ Failed to load {file_path.name}: {e}")
        
        return documents
'''
        
        # API endpoints for RAG
        files["rag/api.py"] = '''"""
FastAPI endpoints for RAG system
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from pathlib import Path
import shutil
from .pipeline import RAGPipeline
from .document_loader import DocumentLoader

app = FastAPI(title="RAG API")

# Initialize RAG pipeline
rag_pipeline = RAGPipeline()

class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 3

class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[dict]

@app.post("/index")
async def index_document(file: UploadFile = File(...)):
    """Upload and index a document."""
    try:
        # Save uploaded file
        upload_dir = Path("uploads")
        upload_dir.mkdir(exist_ok=True)
        file_path = upload_dir / file.filename
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Load and index
        documents = DocumentLoader.load_file(str(file_path))
        num_chunks = rag_pipeline.add_documents([doc.page_content for doc in documents])
        
        return {
            "message": f"Indexed {file.filename}",
            "chunks": num_chunks
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Query the RAG system."""
    try:
        result = rag_pipeline.query(request.question)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/search")
async def search(query: str, k: int = 5):
    """Similarity search without LLM."""
    try:
        results = rag_pipeline.similarity_search(query, k=k)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    """Health check."""
    return {"status": "healthy", "has_documents": rag_pipeline.vectorstore is not None}
'''
        
        return files
    
    def _generate_llm_finetuning(
        self,
        project_name: str,
        config: Dict[str, Any]
    ) -> Dict[str, str]:
        """Generate LLM fine-tuning pipeline."""
        
        model_name = config.get("model", "mistral-7b")
        method = config.get("method", "LoRA")  # LoRA, QLoRA, or full
        
        files = {}
        
        # Fine-tuning script
        files["finetuning/train.py"] = f'''"""
LLM Fine-tuning with {method} for {project_name}

Uses PEFT library for parameter-efficient fine-tuning.
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
import os


class LLMFinetuner:
    """Fine-tune LLM using {method}."""
    def __init__(
        self,
        model_name: str = "{model_name}",
        output_dir: str = "./finetuned_model",
        dataset_name: Optional[str] = None,
        method: str = "{method}"
    ):
        """
        Initialize fine-tuner.
        Args:
            model_name: HuggingFace model name
            output_dir: Directory to save fine-tuned model
            dataset_name: HuggingFace dataset name or path
            method: Fine-tuning method (LoRA, QLoRA, Full)
        """
        self.model_name = model_name
        self.output_dir = output_dir
        self.dataset_name = dataset_name
        self.method = method
        # Device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {{self.device}}")
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        # Load model
        self.model = self._load_model()
        # Apply PEFT
        self.model = self._apply_peft()
        # Dataset
        self.dataset = None
        if dataset_name:
            self.load_dataset(dataset_name)

    def _load_model(self):
        """Load base model."""
        if self.method == "QLoRA":
            # Load in 8-bit for QLoRA
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                load_in_8bit=True,
                device_map="auto",
                trust_remote_code=True
            )
            return prepare_model_for_kbit_training(model)
        else:
            # Load normally for LoRA/Full
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto",
                trust_remote_code=True
            )
            return model

    def _apply_peft(self):
        """Apply PEFT configuration."""
        if self.method in ["LoRA", "QLoRA"]:
            # LoRA configuration
            peft_config = LoraConfig(
                r=16,                          # Rank
                lora_alpha=32,                 # Alpha parameter
                target_modules=["q_proj", "v_proj"],  # Which modules to adapt
                lora_dropout=0.05,             # Dropout
                bias="none",
                task_type="CAUSAL_LM"
            )
            return get_peft_model(self.model, peft_config)
        else:
            # Full fine-tuning (no PEFT)
            return self.model
    
    def load_dataset(self, dataset_name: str):
        """
        Load and prepare dataset.
        
        Args:
            dataset_name: HuggingFace dataset or local path
        """
        # Load dataset
        self.dataset = load_dataset(dataset_name)
        
        # Tokenize
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=512,
                padding="max_length"
            )
        
        self.dataset = self.dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=self.dataset["train"].column_names
        )
        
        print(f"✓ Loaded dataset: {{len(self.dataset['train'])}} training examples")
    
    def train(
        self,
        num_epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 2e-4,
        gradient_accumulation_steps: int = 4,
        warmup_steps: int = 100,
        logging_steps: int = 10
    ):
        """
        Fine-tune the model.
        
        Args:
            num_epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            gradient_accumulation_steps: Gradient accumulation
            warmup_steps: Warmup steps
            logging_steps: Logging frequency
        """
        if not self.dataset:
            raise ValueError("No dataset loaded. Call load_dataset() first.")
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            logging_steps=logging_steps,
            save_steps=500,
            save_total_limit=2,
            fp16=torch.cuda.is_available(),
            logging_dir=f"{{self.output_dir}}/logs",
            report_to="tensorboard",
            load_best_model_at_end=True
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.dataset["train"],
            eval_dataset=self.dataset.get("validation"),
            data_collator=data_collator
        )
        
        # Train
        print("🚀 Starting training...")
        trainer.train()
        
        # Save
        trainer.save_model(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        
        print(f"✓ Model saved to {{self.output_dir}}")
    
    def generate(self, prompt: str, max_length: int = 100) -> str:
        """
        Generate text using fine-tuned model.
        
        Args:
            prompt: Input prompt
            max_length: Maximum generation length
        
        Returns:
            Generated text
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True
            )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


# Example usage
if __name__ == "__main__":
    # Initialize
    finetuner = LLMFinetuner(
        model_name="{model_name}",
        dataset_name="databricks/databricks-dolly-15k"
    )
    
    # Train
    finetuner.train(
        num_epochs=3,
        batch_size=4,
        learning_rate=2e-4
    )
    
    # Test
    result = finetuner.generate("What is machine learning?")
    print(result)
'''
        
        return files
    
    def _generate_llm_agent(self, project_name: str, config: Dict) -> Dict[str, str]:
        """Generate LLM agent system."""
        return {"agents/main.py": "# LLM Agent System\n# TODO: Implement ReAct pattern"}
    
    def _generate_prompt_engineering(self, project_name: str, config: Dict) -> Dict[str, str]:
        """Generate prompt engineering framework."""
        return {"prompts/main.py": "# Prompt Engineering\n# TODO: Implement"}
    
    def _generate_llm_evaluation(self, project_name: str, config: Dict) -> Dict[str, str]:
        """Generate LLM evaluation framework."""
        return {"evaluation/main.py": "# LLM Evaluation\n# TODO: Implement"}
    
    def _generate_semantic_search(self, project_name: str, config: Dict) -> Dict[str, str]:
        """Generate semantic search system."""
        return {"search/main.py": "# Semantic Search\n# TODO: Implement"}
    
    def _generate_chatbot(self, project_name: str, config: Dict) -> Dict[str, str]:
        """Generate chatbot application."""
        return {"chatbot/main.py": "# Chatbot\n# TODO: Implement"}
    
    def _generate_common_files(
        self,
        project_name: str,
        project_type: str,
        config: Dict[str, Any]
    ) -> Dict[str, str]:
        """Generate common files for all GenAI projects."""
        
        files = {}
        
        # Requirements
        files["requirements.txt"] = self._create_requirements(project_type, config)
        
        # Environment template
        files[".env.example"] = self._create_env_template(project_type, config)
        
        # README
        files["README.md"] = self._create_readme(project_name, project_type)
        
        # Docker
        files["Dockerfile"] = self._create_dockerfile(project_type)
        
        return files
    
    def _create_requirements(self, project_type: str, config: Dict) -> str:
        """Create requirements.txt."""
        
        base_requirements = [
            "# Core dependencies",
            "python-dotenv>=1.0.0",
            "pydantic>=2.0.0",
            ""
        ]
        
        if project_type == "rag_system":
            base_requirements.extend([
                "# RAG dependencies",
                "langchain>=0.1.0",
                "chromadb>=0.4.0",
                "sentence-transformers>=2.2.0",
                "pypdf>=3.0.0",
                "python-docx>=1.0.0",
                "unstructured>=0.11.0",
                "fastapi>=0.104.0",
                "uvicorn>=0.24.0",
                ""
            ])
            
            llm_provider = config.get("llm_provider", "OpenAI")
            if llm_provider == "OpenAI":
                base_requirements.append("openai>=1.0.0")
            elif llm_provider == "Anthropic":
                base_requirements.append("anthropic>=0.7.0")
        
        elif project_type == "llm_finetuning":
            base_requirements.extend([
                "# Fine-tuning dependencies",
                "transformers>=4.35.0",
                "peft>=0.7.0",
                "datasets>=2.15.0",
                "accelerate>=0.24.0",
                "bitsandbytes>=0.41.0",
                "tensorboard>=2.15.0",
                ""
            ])
        
        return "\n".join(base_requirements)
    
    def _create_env_template(self, project_type: str, config: Dict) -> str:
        """Create .env.example file."""
        
        env_vars = ["# API Keys"]
        
        if config.get("llm_provider") == "OpenAI":
            env_vars.append("OPENAI_API_KEY=your_openai_api_key_here")
        elif config.get("llm_provider") == "Anthropic":
            env_vars.append("ANTHROPIC_API_KEY=your_anthropic_api_key_here")
        
        env_vars.extend([
            "",
            "# Configuration",
            "ENVIRONMENT=development",
            "LOG_LEVEL=INFO"
        ])
        
        return "\n".join(env_vars)
    
    def _create_readme(self, project_name: str, project_type: str) -> str:
        """Create README.md."""
        
        return f'''# {project_name.replace("_", " ").title()}

{project_type.replace("_", " ").title()} project generated by AutoCoder.

## Features

- GenAI/LLM powered application
- Production-ready code
- Docker deployment support

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

Copy `.env.example` to `.env` and fill in your API keys:

```bash
cp .env.example .env
```

## Usage

```bash
# Run the application
python main.py
```

## Docker

```bash
docker build -t {project_name} .
docker run -p 8000:8000 {project_name}
```

## License

MIT
'''
    
    def _create_dockerfile(self, project_type: str) -> str:
        """Create Dockerfile."""
        
        return '''FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Expose port
EXPOSE 8000

# Run application
CMD ["python", "main.py"]
'''


# Example usage
async def main():
    """Example usage of GenAIAgent."""
    agent = GenAIAgent()
    
    # Generate RAG system
    result = await agent.generate_genai_project(
        project_type="rag_system",
        project_name="document_qa_system",
        config={
            "llm_provider": "OpenAI",
            "vector_db": "Chroma",
            "embedding_model": "OpenAI"
        }
    )
    
    print(f"Generated {len(result)} files:")
    for filename in result.keys():
        print(f"  - {filename}")


if __name__ == "__main__":
    asyncio.run(main())
