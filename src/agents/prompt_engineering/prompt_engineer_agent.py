"""
Prompt Engineering Agent

Enhances vague user inputs into detailed, actionable specifications.
This agent analyzes requirements, extracts intent, and builds comprehensive
specifications that guide all downstream agents.

Key Features:
- Requirement analysis and extraction
- Ambiguity detection
- Specification generation
- Context enrichment

Authors: GaneshS 271123
gmail:ganeshnaik214@gmail.com
"""

from typing import Dict, Any, List, Optional
from loguru import logger
import json

from ..base_agent import BaseAgent
from ..document.document_analyzer_agent import DocumentAnalyzerAgent


class PromptEngineerAgent(BaseAgent):
    """
    Intelligent prompt engineering agent that transforms vague inputs
    into detailed specifications.
    
    This agent acts as the first layer in the system, ensuring that
    all downstream agents have clear, complete requirements to work with.
    """
    
    def __init__(self, llm_client, prompt_manager, rag_agent: Optional[DocumentAnalyzerAgent] = None):
        """
        Initialize PromptEngineerAgent
        
        Args:
            llm_client: LLM client for generation
            prompt_manager: Prompt template manager
            rag_agent: Optional DocumentAnalyzerAgent for research paper knowledge retrieval
        """
        super().__init__(
            agent_type="prompt_engineering",
            name="prompt_engineer_agent",
            role="Expert Requirements Analyst and Specification Engineer",
            llm_client=llm_client,
            prompt_manager=prompt_manager
        )
        
        self.rag_agent = rag_agent
        
        logger.info(f"{self.name} ready for prompt engineering tasks")
        if self.rag_agent:
            logger.info(f"{self.name} RAG support enabled (research paper knowledge retrieval)")
    
    async def enhance_user_input(
        self, 
        user_input: str, 
        context: Optional[Dict[str, Any]] = None,
        research_papers: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Enhance vague user input into detailed specification
        
        Args:
            user_input: Raw user requirement (can be vague)
            context: Optional context (project type, constraints, etc.)
            research_papers: Optional list of research paper paths for RAG knowledge retrieval
        
        Returns:
            Enhanced specification with:
            - analyzed_task: What the user wants to accomplish
            - requirements: Detailed requirements list
            - technical_specs: Technical specifications
            - ambiguities: Detected ambiguities needing clarification
            - suggested_architecture: Recommended project structure
            - research_knowledge: Retrieved knowledge from papers (if RAG enabled)
        """
        try:
            logger.info(f"{self.name} analyzing user input: {user_input[:100]}...")
            
            # Step 0: Retrieve research knowledge if papers provided
            research_knowledge = None
            if research_papers and self.rag_agent:
                research_knowledge = await self._retrieve_research_knowledge(user_input, research_papers)
                logger.info(f"{self.name} retrieved knowledge from {len(research_papers)} papers")
            
            # Step 1: Analyze the input
            analysis = await self._analyze_input(user_input, context)
            
            # Step 2: Extract requirements
            requirements = await self._extract_requirements(user_input, analysis)
            
            # Step 3: Detect ambiguities
            ambiguities = await self._detect_ambiguities(requirements)
            
            # Step 4: Build technical specification (with research knowledge)
            technical_specs = await self._build_technical_specs(
                requirements, 
                analysis,
                research_knowledge=research_knowledge
            )
            
            # Step 5: Suggest architecture
            suggested_architecture = await self._suggest_architecture(technical_specs)
            
            result = {
                "status": "success",
                "original_input": user_input,
                "analyzed_task": analysis,
                "requirements": requirements,
                "technical_specs": technical_specs,
                "ambiguities": ambiguities,
                "suggested_architecture": suggested_architecture,
                "research_knowledge": research_knowledge,
                "confidence": self._calculate_confidence(ambiguities, research_knowledge)
            }
            
            logger.info(f"{self.name} enhanced specification (confidence: {result['confidence']}%)")
            return result
            
        except Exception as e:
            logger.error(f"{self.name} error enhancing input: {str(e)}")
            return {
                "status": "error",
                "message": str(e),
                "original_input": user_input
            }
    
    async def _analyze_input(self, user_input: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze user input to understand task type, domain, and intent
        
        Returns:
            {
                "task_type": "classification|regression|generation|...",
                "domain": "computer_vision|nlp|tabular|...",
                "ml_framework": "pytorch|tensorflow|sklearn|...",
                "deployment_target": "cloud|edge|research|...",
                "complexity": "simple|moderate|complex"
            }
        """
        prompt = self.get_prompt(
            category="prompt_engineering_prompts",
            prompt_name="analyze_user_input",
            variables={
                "user_input": user_input,
                "context": json.dumps(context or {}, indent=2)
            }
        )
        
        response = await self.generate_response(prompt, temperature=0.3)
        
        # Parse JSON response
        try:
            analysis = json.loads(response)
            logger.debug(f"Input analysis: {analysis}")
            return analysis
        except json.JSONDecodeError:
            # Fallback: basic analysis
            return {
                "task_type": "unknown",
                "domain": "general",
                "ml_framework": "pytorch",
                "deployment_target": "research",
                "complexity": "moderate"
            }
    
    async def _extract_requirements(self, user_input: str, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract detailed requirements from user input
        
        Returns list of requirements, each with:
            {
                "id": "REQ-001",
                "category": "functional|non-functional|technical",
                "description": "Clear requirement statement",
                "priority": "must|should|could",
                "measurable": True/False
            }
        """
        prompt = self.get_prompt(
            category="prompt_engineering_prompts",
            prompt_name="extract_requirements",
            variables={
                "user_input": user_input,
                "analysis": json.dumps(analysis, indent=2)
            }
        )
        
        response = await self.generate_response(prompt, temperature=0.3)
        
        try:
            requirements = json.loads(response)
            logger.debug(f"Extracted {len(requirements)} requirements")
            return requirements
        except json.JSONDecodeError:
            # Fallback: basic requirement
            return [{
                "id": "REQ-001",
                "category": "functional",
                "description": user_input,
                "priority": "must",
                "measurable": False
            }]
    
    async def _detect_ambiguities(self, requirements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Detect ambiguities and gaps in requirements
        
        Returns list of ambiguities:
            {
                "id": "AMB-001",
                "category": "missing_info|vague|conflicting",
                "description": "What's ambiguous",
                "impact": "high|medium|low",
                "suggestions": ["Option 1", "Option 2"]
            }
        """
        prompt = self.get_prompt(
            category="prompt_engineering_prompts",
            prompt_name="detect_ambiguities",
            variables={
                "requirements": json.dumps(requirements, indent=2)
            }
        )
        
        response = await self.generate_response(prompt, temperature=0.3)
        
        try:
            ambiguities = json.loads(response)
            logger.debug(f"Detected {len(ambiguities)} ambiguities")
            return ambiguities
        except json.JSONDecodeError:
            return []
    
    async def _retrieve_research_knowledge(
        self, 
        user_input: str, 
        research_papers: List[str]
    ) -> Dict[str, Any]:
        """
        Retrieve relevant knowledge from research papers using RAG
        
        Args:
            user_input: User's requirement/query
            research_papers: List of paper file paths to analyze
        
        Returns:
            {
                "papers_analyzed": int,
                "algorithms": List[Dict],
                "architectures": List[Dict],
                "hyperparameters": List[Dict],
                "relevant_passages": List[Dict],
                "success": bool
            }
        """
        if not self.rag_agent:
            logger.warning("RAG agent not available for knowledge retrieval")
            return {
                "papers_analyzed": 0,
                "algorithms": [],
                "architectures": [],
                "hyperparameters": [],
                "relevant_passages": [],
                "success": False
            }
        
        try:
            logger.info(f"Retrieving knowledge from {len(research_papers)} papers...")
            
            all_algorithms = []
            all_architectures = []
            all_hyperparameters = []
            papers_analyzed = 0
            
            # Analyze each paper
            for paper_path in research_papers:
                try:
                    result = await self.rag_agent.analyze_document(paper_path)
                    if result.get("success") or result.get("status") == "success":
                        # Extract knowledge from result
                        algorithms = result.get("algorithms", [])
                        architectures = result.get("architectures", [])
                        hyperparams = result.get("hyperparameters", [])
                        
                        logger.debug(f"Paper {paper_path}: {len(algorithms)} algorithms, {len(architectures)} architectures, {len(hyperparams)} hyperparameters")
                        
                        all_algorithms.extend(algorithms)
                        all_architectures.extend(architectures)
                        all_hyperparameters.extend(hyperparams)
                        papers_analyzed += 1
                        logger.debug(f"Analyzed: {paper_path}")
                except Exception as e:
                    logger.warning(f"Failed to analyze {paper_path}: {e}")
                    continue
            
            # Query for relevant passages based on user input
            relevant_passages = []
            if papers_analyzed > 0:
                query_result = await self.rag_agent.query_knowledge(
                    query=user_input,
                    top_k=5
                )
                relevant_passages = query_result.get("passages", [])
            
            knowledge = {
                "papers_analyzed": papers_analyzed,
                "algorithms": all_algorithms,
                "architectures": all_architectures,
                "hyperparameters": all_hyperparameters,
                "relevant_passages": relevant_passages,
                "success": papers_analyzed > 0
            }
            
            logger.info(
                f"Retrieved: {len(all_algorithms)} algorithms, "
                f"{len(all_architectures)} architectures, "
                f"{len(all_hyperparameters)} hyperparameters, "
                f"{len(relevant_passages)} relevant passages"
            )
            
            return knowledge
            
        except Exception as e:
            logger.error(f"Error retrieving research knowledge: {e}")
            return {
                "papers_analyzed": 0,
                "algorithms": [],
                "architectures": [],
                "hyperparameters": [],
                "relevant_passages": [],
                "success": False,
                "error": str(e)
            }
    
    async def _build_technical_specs(
        self, 
        requirements: List[Dict[str, Any]], 
        analysis: Dict[str, Any],
        research_knowledge: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Build technical specifications from requirements
        
        Args:
            requirements: Extracted requirements
            analysis: Input analysis
            research_knowledge: Optional retrieved knowledge from research papers
        
        Returns:
            {
                "data": {...},
                "model": {...},
                "training": {...},
                "evaluation": {...},
                "deployment": {...}
            }
        """
        # Build enhanced context with research knowledge
        context_vars = {
            "requirements": json.dumps(requirements, indent=2),
            "analysis": json.dumps(analysis, indent=2)
        }
        
        # Add research knowledge if available
        if research_knowledge and research_knowledge.get("success"):
            context_vars["research_knowledge"] = json.dumps({
                "algorithms": research_knowledge.get("algorithms", []),
                "architectures": research_knowledge.get("architectures", []),
                "hyperparameters": research_knowledge.get("hyperparameters", []),
                "relevant_passages": [
                    {
                        "text": p.get("text", "")[:300],  # Truncate for prompt size
                        "relevance": (1 - p.get("distance", 1)) * 100
                    }
                    for p in research_knowledge.get("relevant_passages", [])[:3]
                ]
            }, indent=2)
            logger.info("Including research knowledge in technical specs generation")
        
        prompt = self.get_prompt(
            category="prompt_engineering_prompts",
            prompt_name="build_technical_specs",
            variables=context_vars
        )
        
        response = await self.generate_response(prompt, temperature=0.4)
        
        try:
            specs = json.loads(response)
            logger.debug(f"Built technical specs with {len(specs)} sections")
            
            # Annotate if research-informed
            if research_knowledge and research_knowledge.get("success"):
                specs["_research_informed"] = True
                specs["_papers_analyzed"] = research_knowledge.get("papers_analyzed", 0)
            
            return specs
        except json.JSONDecodeError:
            # Fallback minimal specs
            return {
                "data": {"source": "unknown"},
                "model": {"type": "unknown"},
                "training": {"epochs": 10},
                "evaluation": {"metrics": ["accuracy"]},
                "deployment": {"format": "python"}
            }
    
    async def _suggest_architecture(self, technical_specs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Suggest optimal project architecture based on technical specs
        
        Returns:
            {
                "recommended_structure": "monolithic|modular|microservices",
                "estimated_files": 5,
                "suggested_modules": [...],
                "justification": "Why this structure?"
            }
        """
        prompt = self.get_prompt(
            category="prompt_engineering_prompts",
            prompt_name="suggest_architecture",
            variables={
                "technical_specs": json.dumps(technical_specs, indent=2)
            }
        )
        
        response = await self.generate_response(prompt, temperature=0.4)
        
        try:
            architecture = json.loads(response)
            logger.debug(f"Suggested architecture: {architecture.get('recommended_structure')}")
            return architecture
        except json.JSONDecodeError:
            # Fallback: modular structure
            return {
                "recommended_structure": "modular",
                "estimated_files": 5,
                "suggested_modules": ["data", "model", "training", "evaluation"],
                "justification": "Standard ML project structure"
            }
    
    def _calculate_confidence(
        self, 
        ambiguities: List[Dict[str, Any]],
        research_knowledge: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Calculate confidence score based on ambiguities and research knowledge
        
        Args:
            ambiguities: Detected ambiguities
            research_knowledge: Retrieved research knowledge (boosts confidence)
        
        Returns:
            Confidence percentage (0-100)
        """
        # Base confidence
        base_confidence = 95
        
        # Boost if research knowledge available
        if research_knowledge and research_knowledge.get("success"):
            papers_count = research_knowledge.get("papers_analyzed", 0)
            passages_count = len(research_knowledge.get("relevant_passages", []))
            hyperparams_count = len(research_knowledge.get("hyperparameters", []))
            
            # +3% boost for research-informed specs (max 98%)
            if papers_count > 0 and passages_count > 0:
                base_confidence = 98
                logger.debug(f"Confidence boosted to 98% (research-informed)")
        
        if not ambiguities:
            return base_confidence
        
        # Reduce confidence based on ambiguity impact
        confidence = base_confidence
        for amb in ambiguities:
            impact = amb.get("impact", "low")
            if impact == "high":
                confidence -= 15
            elif impact == "medium":
                confidence -= 10
            else:
                confidence -= 5
        
        return max(confidence, 20)  # Minimum 20% confidence
    
    async def generate_enhanced_prompt(self, specification: Dict[str, Any]) -> str:
        """
        Generate enhanced prompt for downstream agents
        
        Args:
            specification: Enhanced specification from enhance_user_input()
        
        Returns:
            Rich prompt text for code generation agents
        """
        prompt = self.get_prompt(
            category="prompt_engineering_prompts",
            prompt_name="generate_enhanced_prompt",
            variables={
                "specification": json.dumps(specification, indent=2)
            }
        )
        
        response = await self.generate_response(prompt, temperature=0.3)
        return response.strip()
    
    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a prompt engineering task
        
        Args:
            task: Task dictionary with type and data
        
        Returns:
            Task execution result
        """
        task_type = task.get("task_type", "")
        data = task.get("data", {})
        
        if task_type == "enhance_user_input":
            user_input = data.get("user_input", "")
            context = data.get("context", {})
            return await self.enhance_user_input(user_input, context)
        
        elif task_type == "generate_enhanced_prompt":
            specification = data.get("specification", {})
            prompt = await self.generate_enhanced_prompt(specification)
            return {"status": "success", "prompt": prompt}
        
        else:
            return {
                "status": "error",
                "message": f"Unknown task type: {task_type}"
            }
