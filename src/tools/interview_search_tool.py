"""
Interview Preparation Search Tool - Advanced search capabilities for interview prep.

Features:
- Semantic search across project documentation
- Code example finder
- Metrics and statistics lookup
- Similar question finder
- Topic-based filtering

Author: GaneshS 271123
gmail:ganeshnaik214@gmail.com
"""

import asyncio
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import json
import re
from datetime import datetime
from dataclasses import dataclass, asdict

try:
    from sentence_transformers import SentenceTransformer
    import chromadb
    from chromadb.config import Settings
except ImportError:
    print("Warning: sentence-transformers or chromadb not installed")


@dataclass
class SearchResult:
    """Single search result with metadata."""
    content: str
    source: str
    relevance_score: float
    category: str
    has_code: bool
    has_metrics: bool
    keywords: List[str]
    
    def to_dict(self) -> Dict:
        return asdict(self)


class InterviewSearchTool:
    """Advanced search tool for interview preparation."""
    
    def __init__(
        self,
        docs_path: Path = Path("docs"),
        vector_db_path: Path = Path("data/interview_prep_db")
    ):
        """
        Initialize search tool.
        
        Args:
            docs_path: Path to project documentation
            vector_db_path: Path to vector database
        """
        self.docs_path = docs_path
        self.vector_db_path = vector_db_path
        
        # Initialize embedder
        print("🔧 Loading embedding model...")
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize vector DB
        print("🔧 Connecting to vector database...")
        self.vector_db = chromadb.Client(Settings(
            persist_directory=str(vector_db_path),
            anonymized_telemetry=False
        ))
        
        # Create/get collection
        try:
            self.collection = self.vector_db.get_collection("interview_prep")
            print(f"✅ Loaded existing collection with {self.collection.count()} documents")
        except:
            print("📚 Creating new collection...")
            self.collection = self.vector_db.create_collection(
                name="interview_prep",
                metadata={"description": "Interview preparation content"}
            )
            self._index_documentation()
        
        # Load keyword index
        self.keyword_index = self._build_keyword_index()
        
        # Common interview topics
        self.topics = {
            "architecture": ["design", "structure", "agents", "orchestrator", "system"],
            "rag": ["retrieval", "embedding", "chromadb", "vector", "sentence transformers"],
            "quality": ["review", "scoring", "testing", "validation"],
            "optimization": ["performance", "speed", "cost", "improvement"],
            "error_fixing": ["error", "bug", "fix", "debugging", "exception"],
            "code_generation": ["coding", "generation", "llm", "prompt"],
            "multi_agent": ["agents", "coordination", "communication", "workflow"],
            "metrics": ["score", "accuracy", "precision", "latency", "tokens"]
        }
    
    def _index_documentation(self):
        """Index all documentation for semantic search."""
        print("📖 Indexing documentation...")
        
        # Find all markdown files
        md_files = list(self.docs_path.glob("*.md"))
        
        if not md_files:
            print("⚠️  No documentation files found")
            return
        
        all_chunks = []
        all_embeddings = []
        all_metadatas = []
        all_ids = []
        
        for md_file in md_files:
            print(f"   Processing {md_file.name}...")
            
            try:
                content = md_file.read_text(encoding="utf-8")
                
                # Split into sections (by headings)
                sections = self._split_into_sections(content, md_file.name)
                
                for i, section in enumerate(sections):
                    chunk_id = f"{md_file.stem}_{i}"
                    
                    # Determine category
                    category = self._categorize_content(section["content"])
                    
                    # Check for code and metrics
                    has_code = "```" in section["content"] or "def " in section["content"]
                    has_metrics = bool(re.search(r'\d+(\.\d+)?%|\d+/\d+|\$\d+', section["content"]))
                    
                    # Extract keywords
                    keywords = self._extract_keywords(section["content"])
                    
                    all_chunks.append(section["content"])
                    all_ids.append(chunk_id)
                    all_metadatas.append({
                        "source": md_file.name,
                        "title": section["title"],
                        "category": category,
                        "has_code": has_code,
                        "has_metrics": has_metrics,
                        "keywords": ",".join(keywords[:10])  # Top 10 keywords
                    })
                
            except Exception as e:
                print(f"   ⚠️  Error processing {md_file.name}: {e}")
        
        if not all_chunks:
            print("⚠️  No content to index")
            return
        
        # Generate embeddings
        print(f"🧮 Generating embeddings for {len(all_chunks)} chunks...")
        all_embeddings = self.embedder.encode(
            all_chunks,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True
        ).tolist()
        
        # Add to vector DB
        print("💾 Storing in vector database...")
        self.collection.add(
            ids=all_ids,
            documents=all_chunks,
            embeddings=all_embeddings,
            metadatas=all_metadatas
        )
        
        print(f"✅ Indexed {len(all_chunks)} sections from {len(md_files)} documents")
    
    def _split_into_sections(self, content: str, filename: str) -> List[Dict[str, str]]:
        """Split document into sections by headings."""
        sections = []
        current_section = {"title": filename, "content": ""}
        
        for line in content.split('\n'):
            if line.startswith('#'):
                # New section
                if current_section["content"].strip():
                    sections.append(current_section)
                
                # Extract title
                title = line.lstrip('#').strip()
                current_section = {"title": title, "content": ""}
            else:
                current_section["content"] += line + "\n"
        
        # Add last section
        if current_section["content"].strip():
            sections.append(current_section)
        
        return sections
    
    def _categorize_content(self, content: str) -> str:
        """Categorize content based on keywords."""
        content_lower = content.lower()
        
        category_scores = {}
        for category, keywords in self.topics.items():
            score = sum(1 for keyword in keywords if keyword in content_lower)
            category_scores[category] = score
        
        # Return category with highest score
        if category_scores:
            return max(category_scores, key=category_scores.get)
        return "general"
    
    def _extract_keywords(self, content: str) -> List[str]:
        """Extract important keywords from content."""
        # Remove markdown and code blocks
        clean_content = re.sub(r'```.*?```', '', content, flags=re.DOTALL)
        clean_content = re.sub(r'[#*`]', '', clean_content)
        
        # Split into words
        words = re.findall(r'\b\w+\b', clean_content.lower())
        
        # Filter common words
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'}
        keywords = [w for w in words if w not in stopwords and len(w) > 3]
        
        # Count frequencies
        keyword_counts = {}
        for keyword in keywords:
            keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1
        
        # Sort by frequency
        sorted_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)
        
        return [kw for kw, count in sorted_keywords]
    
    def _build_keyword_index(self) -> Dict[str, List[str]]:
        """Build inverted index for keyword search."""
        keyword_index = {}
        
        # Get all documents from vector DB
        try:
            results = self.collection.get()
            
            for doc_id, metadata in zip(results['ids'], results['metadatas']):
                if 'keywords' in metadata:
                    keywords = metadata['keywords'].split(',')
                    for keyword in keywords:
                        keyword = keyword.strip()
                        if keyword not in keyword_index:
                            keyword_index[keyword] = []
                        keyword_index[keyword].append(doc_id)
        except:
            pass
        
        return keyword_index
    
    def semantic_search(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """
        Semantic search across documentation.
        
        Args:
            query: Search query
            top_k: Number of results to return
            filters: Optional filters (category, has_code, has_metrics)
            
        Returns:
            List of search results
        """
        print(f"🔍 Searching for: '{query}'")
        
        # Generate query embedding
        query_embedding = self.embedder.encode([query]).tolist()
        
        # Build where clause for filters
        where_clause = {}
        if filters:
            if 'category' in filters:
                where_clause['category'] = filters['category']
            if 'has_code' in filters:
                where_clause['has_code'] = filters['has_code']
            if 'has_metrics' in filters:
                where_clause['has_metrics'] = filters['has_metrics']
        
        # Query vector DB
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            where=where_clause if where_clause else None
        )
        
        # Convert to SearchResult objects
        search_results = []
        for i in range(len(results['ids'][0])):
            doc_id = results['ids'][0][i]
            content = results['documents'][0][i]
            metadata = results['metadatas'][0][i]
            distance = results['distances'][0][i]
            
            # Convert distance to relevance score (0-1)
            relevance_score = 1.0 / (1.0 + distance)
            
            # Extract keywords from metadata
            keywords = metadata.get('keywords', '').split(',')
            keywords = [k.strip() for k in keywords if k.strip()]
            
            result = SearchResult(
                content=content[:500] + "..." if len(content) > 500 else content,
                source=metadata.get('source', 'Unknown'),
                relevance_score=relevance_score,
                category=metadata.get('category', 'general'),
                has_code=metadata.get('has_code', False),
                has_metrics=metadata.get('has_metrics', False),
                keywords=keywords[:5]
            )
            search_results.append(result)
        
        return search_results
    
    def keyword_search(
        self,
        keywords: List[str],
        match_all: bool = False
    ) -> List[SearchResult]:
        """
        Keyword-based search.
        
        Args:
            keywords: List of keywords to search for
            match_all: If True, match all keywords (AND), else any keyword (OR)
            
        Returns:
            List of search results
        """
        print(f"🔍 Keyword search: {keywords} (match_all={match_all})")
        
        # Find documents matching keywords
        matching_doc_ids = set()
        
        for keyword in keywords:
            keyword_lower = keyword.lower()
            if keyword_lower in self.keyword_index:
                doc_ids = set(self.keyword_index[keyword_lower])
                
                if not matching_doc_ids:
                    matching_doc_ids = doc_ids
                elif match_all:
                    matching_doc_ids &= doc_ids  # Intersection (AND)
                else:
                    matching_doc_ids |= doc_ids  # Union (OR)
        
        # Get documents
        if not matching_doc_ids:
            return []
        
        results = self.collection.get(ids=list(matching_doc_ids))
        
        # Convert to SearchResult objects
        search_results = []
        for i in range(len(results['ids'])):
            content = results['documents'][i]
            metadata = results['metadatas'][i]
            
            # Calculate relevance based on keyword matches
            content_lower = content.lower()
            keyword_matches = sum(1 for kw in keywords if kw.lower() in content_lower)
            relevance_score = keyword_matches / len(keywords)
            
            result = SearchResult(
                content=content[:500] + "..." if len(content) > 500 else content,
                source=metadata.get('source', 'Unknown'),
                relevance_score=relevance_score,
                category=metadata.get('category', 'general'),
                has_code=metadata.get('has_code', False),
                has_metrics=metadata.get('has_metrics', False),
                keywords=metadata.get('keywords', '').split(',')[:5]
            )
            search_results.append(result)
        
        # Sort by relevance
        search_results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        return search_results
    
    def find_code_examples(
        self,
        topic: str,
        language: str = "python"
    ) -> List[SearchResult]:
        """
        Find code examples for a specific topic.
        
        Args:
            topic: Topic to find examples for
            language: Programming language
            
        Returns:
            List of search results with code
        """
        print(f"💻 Finding {language} code examples for: '{topic}'")
        
        return self.semantic_search(
            query=f"{topic} {language} code example",
            top_k=5,
            filters={"has_code": True}
        )
    
    def find_metrics(
        self,
        topic: str
    ) -> List[SearchResult]:
        """
        Find metrics and statistics for a topic.
        
        Args:
            topic: Topic to find metrics for
            
        Returns:
            List of search results with metrics
        """
        print(f"📊 Finding metrics for: '{topic}'")
        
        return self.semantic_search(
            query=f"{topic} metrics performance statistics",
            top_k=5,
            filters={"has_metrics": True}
        )
    
    def find_by_category(
        self,
        category: str,
        limit: int = 10
    ) -> List[SearchResult]:
        """
        Get all content for a specific category.
        
        Args:
            category: Category name
            limit: Maximum results
            
        Returns:
            List of search results
        """
        print(f"📁 Finding content in category: '{category}'")
        
        # Query all documents in category
        results = self.collection.get(
            where={"category": category},
            limit=limit
        )
        
        # Convert to SearchResult objects
        search_results = []
        for i in range(len(results['ids'])):
            content = results['documents'][i]
            metadata = results['metadatas'][i]
            
            result = SearchResult(
                content=content[:500] + "..." if len(content) > 500 else content,
                source=metadata.get('source', 'Unknown'),
                relevance_score=1.0,  # All equally relevant
                category=metadata.get('category', 'general'),
                has_code=metadata.get('has_code', False),
                has_metrics=metadata.get('has_metrics', False),
                keywords=metadata.get('keywords', '').split(',')[:5]
            )
            search_results.append(result)
        
        return search_results
    
    def suggest_topics(self, query: str) -> List[Tuple[str, float]]:
        """
        Suggest relevant topics based on a query.
        
        Args:
            query: Search query
            
        Returns:
            List of (topic, relevance_score) tuples
        """
        query_lower = query.lower()
        
        topic_scores = []
        for topic, keywords in self.topics.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            if score > 0:
                # Normalize by number of keywords
                normalized_score = score / len(keywords)
                topic_scores.append((topic, normalized_score))
        
        # Sort by score
        topic_scores.sort(key=lambda x: x[1], reverse=True)
        
        return topic_scores
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about indexed content."""
        total_docs = self.collection.count()
        
        # Get all metadatas
        results = self.collection.get()
        metadatas = results['metadatas']
        
        # Count by category
        category_counts = {}
        docs_with_code = 0
        docs_with_metrics = 0
        
        for metadata in metadatas:
            category = metadata.get('category', 'general')
            category_counts[category] = category_counts.get(category, 0) + 1
            
            if metadata.get('has_code', False):
                docs_with_code += 1
            
            if metadata.get('has_metrics', False):
                docs_with_metrics += 1
        
        return {
            "total_documents": total_docs,
            "categories": category_counts,
            "documents_with_code": docs_with_code,
            "documents_with_metrics": docs_with_metrics,
            "code_percentage": (docs_with_code / total_docs * 100) if total_docs > 0 else 0,
            "metrics_percentage": (docs_with_metrics / total_docs * 100) if total_docs > 0 else 0
        }


# CLI interface for search tool
def interactive_search():
    """Run interactive search session."""
    print("🚀 Initializing Interview Search Tool...")
    search_tool = InterviewSearchTool()
    
    print("\n" + "=" * 60)
    print("🔍 Interview Preparation Search Tool")
    print("=" * 60)
    print("\nCommands:")
    print("  - search [query] - Semantic search")
    print("  - keyword [word1,word2,...] - Keyword search")
    print("  - code [topic] - Find code examples")
    print("  - metrics [topic] - Find metrics/statistics")
    print("  - category [name] - Browse by category")
    print("  - topics - List available topics")
    print("  - stats - Show database statistics")
    print("  - quit - Exit")
    print("=" * 60)
    
    while True:
        print("\n")
        user_input = input("🔍 Command: ").strip()
        
        if not user_input:
            continue
        
        parts = user_input.split(maxsplit=1)
        command = parts[0].lower()
        query = parts[1] if len(parts) > 1 else ""
        
        if command == 'quit':
            print("👋 Good luck with your preparation!")
            break
        
        elif command == 'search':
            if not query:
                print("❌ Please provide a search query")
                continue
            
            results = search_tool.semantic_search(query, top_k=5)
            print(f"\n✅ Found {len(results)} results:\n")
            
            for i, result in enumerate(results, 1):
                print(f"{i}. [{result.category}] {result.source}")
                print(f"   Relevance: {result.relevance_score:.2%}")
                print(f"   {result.content[:200]}...")
                print(f"   🏷️  Keywords: {', '.join(result.keywords)}")
                print()
        
        elif command == 'keyword':
            if not query:
                print("❌ Please provide keywords (comma-separated)")
                continue
            
            keywords = [k.strip() for k in query.split(',')]
            results = search_tool.keyword_search(keywords, match_all=False)
            print(f"\n✅ Found {len(results)} results:\n")
            
            for i, result in enumerate(results[:5], 1):
                print(f"{i}. [{result.category}] {result.source}")
                print(f"   Match: {result.relevance_score:.2%}")
                print(f"   {result.content[:200]}...")
                print()
        
        elif command == 'code':
            if not query:
                print("❌ Please provide a topic")
                continue
            
            results = search_tool.find_code_examples(query)
            print(f"\n✅ Found {len(results)} code examples:\n")
            
            for i, result in enumerate(results, 1):
                print(f"{i}. {result.source}")
                print(f"   {result.content[:300]}...")
                print()
        
        elif command == 'metrics':
            if not query:
                print("❌ Please provide a topic")
                continue
            
            results = search_tool.find_metrics(query)
            print(f"\n✅ Found {len(results)} results with metrics:\n")
            
            for i, result in enumerate(results, 1):
                print(f"{i}. {result.source}")
                print(f"   {result.content[:300]}...")
                print()
        
        elif command == 'category':
            if not query:
                print("❌ Please provide a category name")
                print(f"Available: {', '.join(search_tool.topics.keys())}")
                continue
            
            results = search_tool.find_by_category(query, limit=10)
            print(f"\n✅ Found {len(results)} documents in '{query}':\n")
            
            for i, result in enumerate(results, 1):
                print(f"{i}. {result.source}")
                print(f"   {result.content[:200]}...")
                print()
        
        elif command == 'topics':
            print("\n📚 Available Topics:")
            for topic, keywords in search_tool.topics.items():
                print(f"  • {topic}: {', '.join(keywords[:5])}")
        
        elif command == 'stats':
            stats = search_tool.get_statistics()
            print("\n📊 Database Statistics:")
            print(f"  Total documents: {stats['total_documents']}")
            print(f"  Documents with code: {stats['documents_with_code']} ({stats['code_percentage']:.1f}%)")
            print(f"  Documents with metrics: {stats['documents_with_metrics']} ({stats['metrics_percentage']:.1f}%)")
            print("\n  Categories:")
            for category, count in stats['categories'].items():
                print(f"    • {category}: {count}")
        
        else:
            print(f"❌ Unknown command: {command}")


if __name__ == "__main__":
    interactive_search()
