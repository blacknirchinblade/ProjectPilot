"""
Interview Q&A Agent - Intelligent interview preparation assistant.

This agent helps you prepare for technical interviews by:
- Answering questions about your AutoCoder project
- Providing detailed explanations with code examples
- Simulating interview scenarios
- Suggesting follow-up questions
- Generating STAR-format behavioral answers

Authors: GaneshS 271123
gmail:ganeshnaik214@gmail.com
"""

import asyncio
from typing import Dict, List, Optional, Any
from pathlib import Path
import json
from datetime import datetime

from src.agents.base_agent import BaseAgent
from src.tools.shared_memory import SharedMemory


class InterviewQAAgent(BaseAgent):
    """Agent specialized in answering interview questions about the project."""
    
    def __init__(self, project_docs_path: Optional[Path] = None):
        """
        Initialize Interview Q&A Agent.
        
        Args:
            project_docs_path: Path to project documentation directory
        """
        super().__init__(agent_type="interview_qa")
        self.project_docs_path = project_docs_path or Path("docs")
        self.memory = SharedMemory()
        
        # Load project context
        self.project_context = self._load_project_context()
        
        # Interview question categories
        self.question_categories = {
            "technical": "Technical implementation details",
            "architectural": "System design and architecture",
            "problem_solving": "Problem-solving approaches",
            "optimization": "Performance and optimization",
            "behavioral": "Behavioral and STAR-format answers",
            "debugging": "Debugging and error handling",
            "testing": "Testing strategies and quality",
            "future": "Future improvements and scalability"
        }
        
        # Common interview questions database
        self.common_questions = self._load_common_questions()
        
    def _load_project_context(self) -> Dict[str, Any]:
        """Load project documentation and context."""
        context = {
            "project_name": "AutoCoder",
            "description": "Multi-agent AI system for code generation",
            "tech_stack": ["Python", "Gemini 2.5 Pro", "ChromaDB", "Sentence Transformers"],
            "key_features": [
                "Dynamic architecture design (DynamicArchitectAgent - NO templates)",
                "RAG-powered generation (Sentence Transformers + ChromaDB)",
                "Multi-agent system (15+ agents)",
                "Quality review (6 reviewers)",
                "Error fixing (20+ types)",
                "Natural language modifications",
                "Task tracking with resume (TaskTracker + TODO.json)",
                "Import map (100% correct imports)",
                "Dependency-aware build order"
            ],
            "agents": {
                "planning": ["PlanningAgent", "DynamicArchitectAgent", "DocumentAnalyzerAgent"],
                "generation": ["CodingAgent", "TestingAgent", "DocumentationAgent"],
                "quality": ["ReviewOrchestrator (6 reviewers)", "ErrorFixingAgent", "RefactoringAgent"],
                "interactive": ["ContextualChangeAgent", "ModificationAgent"],
                "support": ["TaskTracker", "SharedMemory", "PromptManager"]
            },
            "metrics": {
                "quality_score": "85-90/100 (improved from 58/100)",
                "code_generated": "66,000+ lines",
                "cost_savings": "$780/year (RAG migration)",
                "embedding_speed": "24.76 it/s (146% improvement)",
                "files_per_project": "20+ small files (100-250 lines each)",
                "import_accuracy": "100% (with import_map)"
            },
            "innovations": {
                "DynamicArchitectAgent": "LLM designs structure - NO hardcoded templates",
                "TaskTracker": "Persistent task management with resume capability",
                "Import Map": "All agents know correct import paths",
                "Dependency Resolution": "Topological sort for correct build order",
                "Quality System": "6 specialized reviewers with weighted scoring"
            }
        }
        
        # Try to load from actual docs
        try:
            doc_file = self.project_docs_path / "PROJECT_DOCUMENTATION_AND_INTERVIEW_GUIDE.md"
            if doc_file.exists():
                context["full_documentation"] = doc_file.read_text(encoding="utf-8")
        except Exception as e:
            self.logger.warning(f"Could not load full documentation: {e}")
        
        return context
    
    def _load_common_questions(self) -> Dict[str, List[Dict]]:
        """Load common interview questions by category."""
        return {
            "technical": [
                {
                    "question": "How does your RAG system work?",
                    "key_points": ["Sentence Transformers", "ChromaDB", "Embedding pipeline", "Similarity search"],
                    "code_example": True
                },
                {
                    "question": "Explain your multi-agent architecture.",
                    "key_points": ["Agent types", "Communication", "Coordination", "Shared memory"],
                    "code_example": True
                },
                {
                    "question": "How do you handle dependencies between files?",
                    "key_points": ["Dependency graph", "Topological sort", "Build order"],
                    "code_example": True
                },
                {
                    "question": "How does DynamicArchitectAgent work?",
                    "key_points": ["No templates", "LLM-based design", "Import map", "Dependency resolution", "Small files"],
                    "code_example": True
                },
                {
                    "question": "How do you track tasks and enable resume?",
                    "key_points": ["TaskTracker", "TODO.json", "Dependencies", "Progress tracking", "Resume capability"],
                    "code_example": True
                }
            ],
            "architectural": [
                {
                    "question": "Why did you choose a multi-agent approach?",
                    "key_points": ["Separation of concerns", "Modularity", "Scalability", "Maintainability"],
                    "code_example": False
                },
                {
                    "question": "How does the Orchestrator coordinate agents?",
                    "key_points": ["Workflow phases", "Task tracking", "Agent communication"],
                    "code_example": True
                }
            ],
            "problem_solving": [
                {
                    "question": "What was the most challenging problem you solved?",
                    "key_points": ["RAG migration", "Cost optimization", "Performance improvement"],
                    "star_format": True
                },
                {
                    "question": "How do you handle code quality?",
                    "key_points": ["Multi-reviewer system", "Weighted scoring", "Quality gates"],
                    "code_example": True
                }
            ],
            "optimization": [
                {
                    "question": "How did you optimize the RAG system?",
                    "key_points": ["Batch processing", "Caching", "Model quantization"],
                    "metrics": {"speed": "146% improvement", "cost": "$780 → $0"}
                },
                {
                    "question": "What performance metrics do you track?",
                    "key_points": ["Quality scores", "Generation time", "Token usage"],
                    "code_example": False
                }
            ]
        }
    
    async def answer_question(
        self,
        question: str,
        context: Optional[str] = None,
        format: str = "detailed"  # "brief", "detailed", "star"
    ) -> Dict[str, Any]:
        """
        Answer an interview question intelligently.
        
        Args:
            question: The interview question
            context: Additional context about the question
            format: Response format (brief/detailed/star)
            
        Returns:
            Dictionary with answer, examples, and follow-up suggestions
        """
        self.logger.info(f"Answering interview question: {question}")
        
        # Determine question category
        category = await self._categorize_question(question)
        
        # Build context-rich prompt
        prompt = self._build_answer_prompt(question, context, category, format)
        
        # Generate answer
        answer = await self.generate_response(prompt, temperature=0.3)
        
        # Extract code examples if present
        code_examples = self._extract_code_examples(answer)
        
        # Generate follow-up questions
        follow_ups = await self._generate_follow_ups(question, category)
        
        # Create structured response
        response = {
            "question": question,
            "category": category,
            "answer": answer,
            "code_examples": code_examples,
            "follow_up_questions": follow_ups,
            "timestamp": datetime.now().isoformat(),
            "format": format
        }
        
        # Store in memory for conversation context
        await self.memory.store(
            key=f"interview_qa_{datetime.now().timestamp()}",
            value=response,
            agent_name=self.agent_name
        )
        
        return response
    
    async def _categorize_question(self, question: str) -> str:
        """Categorize the interview question."""
        prompt = f"""
Categorize this interview question into ONE of these categories:
- technical (implementation details, code explanations)
- architectural (system design, structure)
- problem_solving (challenges overcome)
- optimization (performance, cost improvements)
- behavioral (STAR format, team dynamics)
- debugging (error handling, troubleshooting)
- testing (quality assurance, testing strategies)
- future (improvements, scalability)

Question: {question}

Respond with ONLY the category name.
"""
        
        category = await self.generate_response(prompt, temperature=0.1)
        category = category.strip().lower()
        
        # Validate category
        if category not in self.question_categories:
            category = "technical"  # Default
        
        return category
    
    def _build_answer_prompt(
        self,
        question: str,
        context: Optional[str],
        category: str,
        format: str
    ) -> str:
        """Build comprehensive prompt for answering questions."""
        
        # Base prompt
        prompt = f"""
You are an expert technical interviewer helping prepare answers about the AutoCoder project.

PROJECT CONTEXT:
{json.dumps(self.project_context, indent=2)}

QUESTION CATEGORY: {category} ({self.question_categories[category]})

QUESTION: {question}
"""
        
        if context:
            prompt += f"\nADDITIONAL CONTEXT: {context}\n"
        
        # Format-specific instructions
        if format == "brief":
            prompt += """
Provide a BRIEF answer (2-3 sentences max).
Focus on the key point only.
No code examples unless essential.
"""
        
        elif format == "detailed":
            prompt += """
Provide a DETAILED answer with:
1. Clear explanation (2-3 paragraphs)
2. Code examples (if relevant)
3. Technical details and metrics
4. Real-world context from the project

Use markdown formatting for code blocks.
"""
        
        elif format == "star":
            prompt += """
Provide answer in STAR format:

SITUATION: (What was the context?)
TASK: (What needed to be done?)
ACTION: (What did you do? Be specific with steps)
RESULT: (What was the outcome? Include metrics)

Include code snippets where relevant.
Make it interview-ready (confident, specific, quantified).
"""
        
        # Category-specific guidance
        if category == "technical":
            prompt += """
Technical questions require:
- Architecture diagrams (in text/ascii)
- Code examples
- Performance metrics
- Trade-off discussions
"""
        
        elif category == "problem_solving":
            prompt += """
Problem-solving questions require:
- Clear problem statement
- Multiple approaches considered
- Decision rationale
- Measurable results
"""
        
        elif category == "optimization":
            prompt += """
Optimization questions require:
- Before/after comparisons
- Specific metrics (%, ms, $)
- Profiling/benchmarking approach
- Trade-offs made
"""
        
        prompt += "\nYour answer:"
        
        return prompt
    
    def _extract_code_examples(self, answer: str) -> List[str]:
        """Extract code blocks from the answer."""
        code_examples = []
        
        # Simple extraction (looking for code blocks)
        lines = answer.split('\n')
        in_code_block = False
        current_code = []
        
        for line in lines:
            if line.strip().startswith('```'):
                if in_code_block:
                    # End of code block
                    code_examples.append('\n'.join(current_code))
                    current_code = []
                in_code_block = not in_code_block
            elif in_code_block:
                current_code.append(line)
        
        return code_examples
    
    async def _generate_follow_ups(
        self,
        original_question: str,
        category: str
    ) -> List[str]:
        """Generate relevant follow-up questions."""
        prompt = f"""
Given this interview question: "{original_question}"
Category: {category}

Generate 3 natural follow-up questions that an interviewer might ask.
Make them progressively deeper/more challenging.

Format as a numbered list.
"""
        
        response = await self.generate_response(prompt, temperature=0.5)
        
        # Parse follow-up questions
        follow_ups = []
        for line in response.split('\n'):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-')):
                # Remove numbering
                question = line.split('.', 1)[-1].strip()
                question = question.lstrip('- ')
                if question:
                    follow_ups.append(question)
        
        return follow_ups[:3]  # Max 3 follow-ups
    
    async def simulate_interview(
        self,
        topic: str = "general",
        duration_minutes: int = 30,
        difficulty: str = "medium"  # "easy", "medium", "hard"
    ) -> Dict[str, Any]:
        """
        Simulate a technical interview session.
        
        Args:
            topic: Interview focus (general, architecture, optimization, etc.)
            duration_minutes: Estimated interview duration
            difficulty: Question difficulty level
            
        Returns:
            Interview simulation with questions and suggested answers
        """
        self.logger.info(f"Simulating {duration_minutes}-minute interview on {topic}")
        
        # Determine number of questions
        questions_count = duration_minutes // 5  # ~5 mins per question
        
        prompt = f"""
Generate a realistic technical interview for the AutoCoder project.

INTERVIEW PARAMETERS:
- Topic: {topic}
- Duration: {duration_minutes} minutes
- Difficulty: {difficulty}
- Questions: {questions_count}

PROJECT: AutoCoder (multi-agent AI code generation system)

Generate {questions_count} interview questions covering:
1. Opening question (warm-up)
2-{questions_count-1}. Technical deep-dive questions
{questions_count}. Closing question (future plans, questions for interviewer)

Format each question as:
Q[number]: [Question]
Difficulty: [Easy/Medium/Hard]
Expected time: [X minutes]
Key points to cover: [bullet points]

Make it realistic and progressively challenging.
"""
        
        interview_plan = await self.generate_response(prompt, temperature=0.6)
        
        # Parse questions
        questions = self._parse_interview_questions(interview_plan)
        
        # Generate suggested answers for each
        answers = []
        for q in questions:
            answer = await self.answer_question(
                question=q["question"],
                format="detailed" if q["difficulty"] != "Easy" else "brief"
            )
            answers.append(answer)
        
        simulation = {
            "topic": topic,
            "duration_minutes": duration_minutes,
            "difficulty": difficulty,
            "questions": questions,
            "answers": answers,
            "timestamp": datetime.now().isoformat()
        }
        
        return simulation
    
    def _parse_interview_questions(self, interview_plan: str) -> List[Dict]:
        """Parse interview questions from generated plan."""
        questions = []
        current_question = None
        
        for line in interview_plan.split('\n'):
            line = line.strip()
            
            if line.startswith('Q'):
                # New question
                if current_question:
                    questions.append(current_question)
                
                # Extract question text
                question_text = line.split(':', 1)[-1].strip()
                current_question = {
                    "question": question_text,
                    "difficulty": "Medium",
                    "expected_time": 5,
                    "key_points": []
                }
            
            elif line.startswith('Difficulty:') and current_question:
                current_question["difficulty"] = line.split(':', 1)[-1].strip()
            
            elif line.startswith('Expected time:') and current_question:
                time_str = line.split(':', 1)[-1].strip()
                # Extract number
                import re
                match = re.search(r'\d+', time_str)
                if match:
                    current_question["expected_time"] = int(match.group())
            
            elif line.startswith('Key points') and current_question:
                continue  # Header line
            
            elif line.startswith('-') and current_question:
                # Key point
                point = line.lstrip('- ').strip()
                if point:
                    current_question["key_points"].append(point)
        
        # Add last question
        if current_question:
            questions.append(current_question)
        
        return questions
    
    async def get_question_by_topic(
        self,
        topic: str,
        count: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Get interview questions filtered by topic.
        
        Args:
            topic: Topic/category of questions
            count: Number of questions to return
            
        Returns:
            List of questions with metadata
        """
        # Map topic to category
        category = topic.lower()
        if category not in self.question_categories:
            # Try to find best match
            category = "technical"
        
        # Get questions from database
        questions = self.common_questions.get(category, [])
        
        # Limit to requested count
        return questions[:count]
    
    async def practice_mode(
        self,
        category: Optional[str] = None,
        count: int = 10
    ) -> Dict[str, Any]:
        """
        Interactive practice mode for interview preparation.
        
        Args:
            category: Focus category (None for mixed)
            count: Number of questions
            
        Returns:
            Practice session results
        """
        self.logger.info(f"Starting practice mode: {count} questions, category={category}")
        
        # Get questions
        if category:
            questions = await self.get_question_by_topic(category, count)
        else:
            # Mixed questions from all categories
            questions = []
            for cat in self.common_questions:
                questions.extend(self.common_questions[cat][:2])
            questions = questions[:count]
        
        practice_session = {
            "category": category or "mixed",
            "total_questions": len(questions),
            "questions": questions,
            "started_at": datetime.now().isoformat(),
            "status": "ready"
        }
        
        return practice_session


# CLI interface for interactive practice
async def interactive_qa_session():
    """Run interactive Q&A session in terminal."""
    agent = InterviewQAAgent()
    
    print("=" * 60)
    print("🎤 AutoCoder Interview Q&A Agent")
    print("=" * 60)
    print("\nCommands:")
    print("  - Type your question to get an answer")
    print("  - 'simulate' - Run full interview simulation")
    print("  - 'practice [category]' - Practice specific category")
    print("  - 'topics' - List question categories")
    print("  - 'quit' - Exit")
    print("=" * 60)
    
    while True:
        print("\n")
        user_input = input("❓ Your question: ").strip()
        
        if not user_input:
            continue
        
        if user_input.lower() == 'quit':
            print("👋 Good luck with your interviews!")
            break
        
        elif user_input.lower() == 'topics':
            print("\n📚 Question Categories:")
            for category, description in agent.question_categories.items():
                print(f"  • {category}: {description}")
        
        elif user_input.lower() == 'simulate':
            print("\n🎬 Starting interview simulation...")
            simulation = await agent.simulate_interview(
                topic="general",
                duration_minutes=30,
                difficulty="medium"
            )
            
            print(f"\n📋 Interview Plan ({len(simulation['questions'])} questions):")
            for i, q in enumerate(simulation['questions'], 1):
                print(f"\nQ{i}: {q['question']}")
                print(f"   Difficulty: {q['difficulty']} | Time: {q['expected_time']} min")
                print(f"   Key points: {', '.join(q['key_points'])}")
            
            print("\n💡 Tip: Ask each question individually to get detailed answers!")
        
        elif user_input.lower().startswith('practice'):
            parts = user_input.split()
            category = parts[1] if len(parts) > 1 else None
            
            print(f"\n📝 Practice Mode: {category or 'Mixed'}")
            practice = await agent.practice_mode(category=category, count=5)
            
            print(f"\n{practice['total_questions']} questions loaded:")
            for i, q in enumerate(practice['questions'], 1):
                print(f"\n{i}. {q['question']}")
        
        else:
            # Answer the question
            print("\n🤔 Thinking...")
            response = await agent.answer_question(
                question=user_input,
                format="detailed"
            )
            
            print(f"\n📁 Category: {response['category']}")
            print(f"\n✅ Answer:\n{response['answer']}")
            
            if response['code_examples']:
                print(f"\n💻 {len(response['code_examples'])} code example(s) included")
            
            if response['follow_up_questions']:
                print("\n🔄 Follow-up questions to prepare for:")
                for fq in response['follow_up_questions']:
                    print(f"  • {fq}")


if __name__ == "__main__":
    # Run interactive session
    asyncio.run(interactive_qa_session())
