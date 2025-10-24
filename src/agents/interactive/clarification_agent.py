"""
Clarification Agent

Interactive agent that asks clarifying questions before project generation
to ensure requirements are well-understood.
Authors: GaneshS 271123
gmail:ganeshnaik214@gmail.com
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from src.agents.base_agent import BaseAgent
from src.agents.interactive.data_structures import (
    Answer,
    Priority,
    ProjectContext,
    Question,
    QuestionCategory,
)
from src.agents.interactive.template_loader import get_template_loader

logger = logging.getLogger(__name__)


class ClarificationAgent(BaseAgent):
    """
    Asks clarifying questions before project generation.
    
    Workflow:
    1. Analyze user requirements for ambiguities
    2. Generate prioritized questions
    3. Conduct interactive Q&A session
    4. Parse and validate answers
    5. Update project context
    6. Validate requirement completeness
    
    Attributes:
        question_templates: Pre-defined question patterns
        answer_validators: Validation rules for answers
        min_questions: Minimum questions to ask (default: 3)
        max_questions: Maximum questions to ask (default: 10)
    """
    
    def __init__(self, name: str = "clarification_agent"):
        """Initialize clarification agent"""
        super().__init__(
            name=name,
            role="Interactive Requirement Clarification Specialist",
            agent_type="clarification"
        )
        
        self.min_questions = 3
        self.max_questions = 12  # Increased from 10 to 12 for complex projects
        self.question_templates: Dict[str, Question] = {}
        self.answer_validators: Dict[str, Any] = {}
        
        # Load question templates
        self.template_loader = get_template_loader()
        logger.info(f"Loaded {len(self.template_loader.templates)} question templates")
        
        logger.info(f"Initialized ClarificationAgent: {name}")
    
    async def generate_smart_questions(
        self,
        specification: Dict[str, Any],
        ambiguities: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Generate smart, context-aware clarifying questions (NEW Week 3 method)
        
        Instead of using template questions, this method uses LLM to generate
        questions specific to the user's request and detected ambiguities.
        
        Args:
            specification: Enhanced specification from PromptEngineerAgent
            ambiguities: List of detected ambiguities
        
        Returns:
            List of smart questions:
            [
                {
                    "id": "Q-001",
                    "question": "Specific, actionable question?",
                    "category": "data|model|training|deployment|other",
                    "priority": "high|medium|low",
                    "options": ["Option 1", "Option 2", "Let agent decide"],
                    "default": "Option 1",
                    "rationale": "Why this question is important",
                    "impact": "What happens if not answered"
                },
                ...
            ]
        """
        try:
            logger.info("Generating smart contextual questions...")
            
            prompt = self.get_prompt(
                "clarification_prompts",
                "generate_smart_questions",
                {
                    "specification": json.dumps(specification, indent=2),
                    "ambiguities": json.dumps(ambiguities, indent=2)
                }
            )
            
            logger.debug(f"Prompt length: {len(prompt)} chars")
            logger.debug(f"Specification: {specification.get('description', 'N/A')[:100]}")
            logger.debug(f"Ambiguities count: {len(ambiguities)}")
            
            response = await self.generate_response(prompt)
            logger.debug(f"Response received. Length: {len(response) if response else 0} chars")
            logger.debug(f"Response type: {type(response)}")
            
            # Check if response is empty
            if not response or not response.strip():
                logger.warning("LLM returned empty response! Using fallback questions.")
                return self._get_fallback_smart_questions(ambiguities)
            
            # Check if response is empty
            if not response or not response.strip():
                logger.warning("LLM returned empty response! Using fallback questions.")
                return self._get_fallback_smart_questions(ambiguities)
            
            # Parse JSON response (extract JSON from markdown if needed)
            try:
                # First, try direct JSON parsing
                logger.debug("Attempting direct JSON parse...")
                questions = json.loads(response)
                logger.info(f"✅ Generated {len(questions)} smart questions via direct parse")
                return questions
            except json.JSONDecodeError as e:
                logger.warning(f"Direct JSON parse failed: {e}. Trying to extract JSON from response...")
                print(f"\n{'='*80}")
                print(f"DEBUG: Response first 500 chars:")
                print(response[:500])
                print(f"{'='*80}\n")
                
                # Try to extract JSON from markdown code blocks
                import re
                print("DEBUG: Attempting markdown extraction with regex...")
                # Use greedy matching to capture entire JSON array
                json_match = re.search(r'```(?:json)?\s*(\[.*\])\s*```', response, re.DOTALL)
                print(f"DEBUG: Regex match found: {json_match is not None}")
                if json_match:
                    print(f"DEBUG: Match group 1 length: {len(json_match.group(1))}")
                    logger.info("📦 Found JSON in markdown code block!")
                    try:
                        extracted = json_match.group(1)
                        print(f"DEBUG: About to parse extracted JSON (length: {len(extracted)})")
                        print(f"DEBUG: First 100 chars of extracted: {extracted[:100]}")
                        questions = json.loads(extracted)
                        print(f"DEBUG: Successfully parsed {len(questions)} questions!")
                        logger.info(f"✅ Extracted {len(questions)} smart questions from markdown")
                        return questions
                    except json.JSONDecodeError as e2:
                        print(f"DEBUG: Failed to parse extracted JSON: {e2}")
                        logger.debug(f"Markdown extraction parse failed: {e2}")
                        logger.debug(f"Extracted content preview: {json_match.group(1)[:200]}")
                else:
                    print("DEBUG: No markdown match found, trying plain regex...")
                
                # Try to find JSON array anywhere in response
                logger.debug("Trying to find JSON array with regex...")
                json_match = re.search(r'\[.*\]', response, re.DOTALL)
                if json_match:
                    try:
                        questions = json.loads(json_match.group(0))
                        logger.info(f"✅ Extracted {len(questions)} smart questions from text")
                        return questions
                    except json.JSONDecodeError as e3:
                        logger.debug(f"Regex extraction parse failed: {e3}")
                
                # Log the actual response for debugging
                logger.error(f"❌ All JSON extraction methods failed!")
                logger.error(f"Response preview: {response[:500]}")
                logger.info("Using fallback questions based on detected ambiguities")
                return self._get_fallback_smart_questions(ambiguities)
                
        except Exception as e:
            logger.error(f"Error generating smart questions: {e}")
            return self._get_fallback_smart_questions(ambiguities)
    
    def _get_fallback_smart_questions(self, ambiguities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Fallback questions if smart generation fails.
        Creates well-formatted questions from detected ambiguities.
        Now with MORE options and custom input support.
        """
        logger.info(f"Creating fallback questions from {len(ambiguities)} ambiguities")
        questions = []
        
        # Dynamic question count based on ambiguity count
        max_questions = min(len(ambiguities), 8)  # Up to 8 questions
        
        for i, amb in enumerate(ambiguities[:max_questions], 1):
            category = amb.get("category", "other")
            description = amb.get("description", "Please provide more details")
            suggestions = amb.get("suggestions", [])
            
            # Ensure we have comprehensive options (5-8 options)
            if suggestions:
                # Add more options based on category
                if category == "data" and len(suggestions) < 5:
                    suggestions.extend([
                        "Public dataset (ImageNet, COCO, etc.)",
                        "Custom dataset (I'll provide)",
                        "Generate synthetic data"
                    ])
                elif category == "model" and len(suggestions) < 5:
                    suggestions.extend([
                        "Pretrained model (faster)",
                        "Custom architecture (more control)",
                        "Ensemble of models"
                    ])
                
                # Remove duplicates while preserving order
                seen = set()
                suggestions = [x for x in suggestions if not (x in seen or seen.add(x))]
                
                # Always add these at the end
                if "Let AI decide based on best practices" not in suggestions:
                    suggestions.append("Let AI decide based on best practices")
                if "Custom (I'll specify)" not in suggestions:
                    suggestions.append("Custom (I'll specify)")
            else:
                # Default comprehensive options
                suggestions = [
                    "Option 1 (recommended)",
                    "Option 2 (alternative)",
                    "Option 3 (advanced)",
                    "Let AI decide based on best practices",
                    "Custom (I'll specify)"
                ]
            
            # Format question based on category
            category_prefixes = {
                "data": "📊 Dataset/Data:",
                "model": "🤖 Model Architecture:",
                "api": "🔌 API Design:",
                "database": "💾 Database:",
                "configuration": "⚙️ Configuration:",
                "deployment": "🚀 Deployment:",
                "infrastructure": "🏗️ Infrastructure:",
                "optimization": "⚡ Optimization:"
            }
            question_prefix = category_prefixes.get(category, "❓ Clarification:")
            
            questions.append({
                "id": f"Q-{i:03d}",
                "question": f"{question_prefix} {description}",
                "category": category,
                "priority": amb.get("impact", "medium"),
                "options": suggestions,
                "default": suggestions[0] if suggestions else None,
                "rationale": f"This helps us understand your {category} requirements better",
                "impact": "Affects implementation approach and generated code structure",
                "custom_prompt": f"What {category} configuration would you like to use?",
                "validation": "Any valid configuration for this category"
            })
        
        logger.info(f"Created {len(questions)} fallback questions with {sum(len(q['options']) for q in questions)} total options")

        return questions
    
    async def prioritize_questions_intelligently(
        self,
        questions: List[Dict[str, Any]],
        user_context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Intelligently prioritize questions based on context (NEW Week 3 method)
        
        Uses LLM to understand which questions are most critical for this
        specific project, rather than using fixed priority rules.
        
        Args:
            questions: List of questions to prioritize
            user_context: User's background, preferences, constraints
        
        Returns:
            Sorted list of questions (most important first)
        """
        try:
            logger.info("Prioritizing questions intelligently...")
            
            prompt = self.get_prompt(
                "clarification_prompts",
                "prioritize_questions",
                {
                    "questions": json.dumps(questions, indent=2),
                    "user_context": json.dumps(user_context, indent=2)
                }
            )
            
            response = await self.generate_response(prompt)
            
            # Parse JSON response with prioritized questions
            try:
                prioritized = json.loads(response)
                logger.info(f"Questions prioritized: {len(prioritized)} returned")
                return prioritized
            except json.JSONDecodeError:
                # Fallback: return original order
                logger.warning("Failed to parse prioritization, using original order")
                return questions
                
        except Exception as e:
            logger.error(f"Error prioritizing questions: {e}")
            return questions
    
    async def validate_answer_contextually(
        self,
        question: Dict[str, Any],
        answer: str,
        specification: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate answer in context of the project (NEW Week 3 method)
        
        Instead of simple validation rules, uses LLM to understand if
        the answer makes sense given the project requirements.
        
        Args:
            question: The question that was asked
            answer: User's answer
            specification: Project specification
        
        Returns:
            {
                "valid": True/False,
                "message": "Validation message",
                "suggestion": "Alternative if invalid",
                "confidence": 0.95
            }
        """
        try:
            prompt = self.get_prompt(
                "clarification_prompts",
                "validate_answer",
                {
                    "question": json.dumps(question),
                    "answer": answer,
                    "specification": json.dumps(specification, indent=2)
                }
            )
            
            response = await self.generate_response(prompt)
            
            try:
                validation = json.loads(response)
                return validation
            except json.JSONDecodeError:
                # Fallback: accept answer
                return {
                    "valid": True,
                    "message": "Answer accepted",
                    "confidence": 0.5
                }
                
        except Exception as e:
            logger.error(f"Error validating answer: {e}")
            return {"valid": True, "message": "Validation skipped", "confidence": 0.5}
    
    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute clarification task.
        
        Task should contain:
        - user_request: Original user requirement
        - interactive: Whether to ask questions interactively (default: True)
        
        Returns:
            Dictionary containing:
            - success: Boolean
            - context: ProjectContext object
            - questions_asked: List of questions
            - answers: Dictionary of answers
        """
        try:
            user_request = task.get("user_request", "")
            interactive = task.get("interactive", True)
            
            if not user_request:
                return {
                    "success": False,
                    "error": "No user request provided",
                    "context": None
                }
            
            logger.info(f"Executing clarification task for: {user_request[:100]}...")
            
            # Step 1: Analyze and generate questions
            questions = await self.analyze_requirements(user_request)
            
            # Step 2: Ask questions
            answers = await self.ask_questions(questions, interactive=interactive)
            
            # Step 3: Update context
            context = await self.update_context(user_request, answers)
            
            return {
                "success": True,
                "context": context,
                "questions_asked": questions,
                "answers": answers,
                "confidence_score": context.confidence_score
            }
            
        except Exception as e:
            logger.error(f"Clarification task failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "context": None
            }
    
    async def analyze_requirements(
        self,
        user_request: str
    ) -> List[Question]:
        """
        Analyze user requirements and identify ambiguities.
        
        Process:
        1. Parse user request with LLM
        2. Identify missing information
        3. Categorize ambiguities
        4. Load relevant question templates
        5. Generate custom questions if needed
        6. Prioritize questions
        7. Return ordered question list
        
        Args:
            user_request: Original user requirement
            
        Returns:
            List of prioritized questions
        """
        logger.info("Analyzing user requirements for ambiguities...")
        
        # Step 1: Analyze with LLM
        analysis_prompt = self.get_prompt(
            "clarification_prompts",
            "analyze_requirements",
            {"request": user_request}
        )
        
        try:
            analysis_text = await self.generate_response(analysis_prompt)
            logger.debug(f"LLM Analysis: {analysis_text[:200]}...")
            
            # Step 2: Extract ambiguities from response
            ambiguities = self._extract_ambiguities(analysis_text)
            logger.info(f"Found {len(ambiguities)} ambiguities")
            
        except Exception as e:
            logger.warning(f"LLM analysis failed: {e}. Using fallback question set.")
            # Fallback to basic questions
            ambiguities = self._get_fallback_ambiguities(user_request)
        
        # Step 3: Generate questions from ambiguities
        questions = []
        
        # Template-based questions
        for ambiguity in ambiguities:
            category = ambiguity.get("category", "other")
            template_key = f"{category}_question"
            
            if template_key in self.question_templates:
                question = self.question_templates[template_key]
                questions.append(question)
            else:
                # Create custom question
                custom_q = self._create_question_from_ambiguity(ambiguity)
                if custom_q:
                    questions.append(custom_q)
        
        # Step 4: Add essential questions if missing
        questions = self._ensure_essential_questions(questions, user_request)
        
        # Step 5: Prioritize and limit
        questions = self._prioritize_questions(questions)
        questions = questions[:self.max_questions]
        
        logger.info(f"Generated {len(questions)} clarification questions")
        return questions
    
    def _extract_ambiguities(self, analysis_text: str) -> List[Dict[str, Any]]:
        """
        Extract ambiguities from LLM analysis response.
        
        Expects JSON format with 'ambiguities' key.
        """
        try:
            # Try to parse as JSON
            if "```json" in analysis_text:
                # Extract JSON from markdown code block
                json_match = re.search(r'```json\s*(\{.*?\})\s*```', analysis_text, re.DOTALL)
                if json_match:
                    analysis_text = json_match.group(1)
            
            data = json.loads(analysis_text)
            
            if isinstance(data, dict) and "ambiguities" in data:
                return data["ambiguities"]
            elif isinstance(data, list):
                return data
            else:
                logger.warning("Unexpected analysis format")
                return []
                
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse analysis as JSON: {e}")
            # Try to extract manually
            return self._extract_ambiguities_manually(analysis_text)
    
    def _extract_ambiguities_manually(self, text: str) -> List[Dict[str, Any]]:
        """Manually extract ambiguities from text"""
        ambiguities = []
        
        # Look for common patterns
        keywords = {
            "framework": QuestionCategory.FRAMEWORK,
            "pytorch": QuestionCategory.FRAMEWORK,
            "tensorflow": QuestionCategory.FRAMEWORK,
            "gpu": QuestionCategory.HARDWARE,
            "cuda": QuestionCategory.HARDWARE,
            "hardware": QuestionCategory.HARDWARE,
            "data": QuestionCategory.DATA,
            "augmentation": QuestionCategory.DATA,
            "model": QuestionCategory.ARCHITECTURE,
            "architecture": QuestionCategory.ARCHITECTURE,
            "deploy": QuestionCategory.DEPLOYMENT,
            "test": QuestionCategory.TESTING,
        }
        
        text_lower = text.lower()
        
        for keyword, category in keywords.items():
            if keyword in text_lower:
                ambiguities.append({
                    "category": category.value,
                    "severity": "HIGH",
                    "context": f"Mentioned {keyword} but details unclear"
                })
        
        return ambiguities[:5]  # Limit to 5
    
    def _get_fallback_ambiguities(self, user_request: str) -> List[Dict[str, Any]]:
        """Get fallback ambiguities when LLM analysis fails"""
        # Always ask these essential questions
        return [
            {"category": "framework", "severity": "BLOCKING"},
            {"category": "hardware", "severity": "HIGH"},
            {"category": "architecture", "severity": "HIGH"},
        ]
    
    def _create_question_from_ambiguity(self, ambiguity: Dict[str, Any]) -> Optional[Question]:
        """Create a Question object from ambiguity dict"""
        category_str = ambiguity.get("category", "other")
        
        try:
            category = QuestionCategory(category_str)
        except ValueError:
            category = QuestionCategory.OTHER
        
        # Try to find a template for this category
        templates_for_category = self.template_loader.get_templates_by_category(category)
        
        if templates_for_category:
            # Use the first template for this category
            # TODO: Could use LLM to select best template based on context
            return templates_for_category[0]
        
        # Fallback: create custom question
        severity = ambiguity.get("severity", "MEDIUM")
        try:
            priority = Priority[severity.upper()]
        except KeyError:
            priority = Priority.MEDIUM
        
        # Generate question text
        question_text = ambiguity.get("question", self._get_default_question_text(category))
        default_answer = ambiguity.get("default", None)
        
        return Question(
            id=f"Q_{category.value}_{id(ambiguity)}",
            category=category,
            text=question_text,
            priority=priority,
            context_key=category.value,
            default=default_answer,
            required=(priority == Priority.BLOCKING)
        )
    
    def _get_default_question_text(self, category: QuestionCategory) -> str:
        """Get default question text for a category"""
        defaults = {
            QuestionCategory.FRAMEWORK: "Which ML framework would you like to use?",
            QuestionCategory.HARDWARE: "What hardware will you use for training?",
            QuestionCategory.DATA: "How should data be handled?",
            QuestionCategory.ARCHITECTURE: "What type of model architecture do you prefer?",
            QuestionCategory.DEPLOYMENT: "Where will this model be deployed?",
            QuestionCategory.TESTING: "What level of testing do you need?",
            QuestionCategory.DOCUMENTATION: "What level of documentation do you need?",
        }
        return defaults.get(category, "Please provide more details about your requirements.")
    
    def _ensure_essential_questions(
        self,
        questions: List[Question],
        user_request: str
    ) -> List[Question]:
        """
        Ensure essential questions are included.
        
        Essential questions:
        - Framework selection (BLOCKING)
        - Hardware requirements (HIGH)
        """
        categories_present = {q.category for q in questions}
        
        # Add framework question if missing (use template)
        if QuestionCategory.FRAMEWORK not in categories_present:
            framework_template = self.template_loader.get_template('framework_selection')
            if framework_template:
                questions.insert(0, framework_template)
            else:
                # Fallback to hardcoded question
                questions.insert(0, Question(
                    id="Q_ESSENTIAL_FRAMEWORK",
                    category=QuestionCategory.FRAMEWORK,
                    text="Which machine learning framework would you like to use?",
                    priority=Priority.BLOCKING,
                    options=[
                        "PyTorch (recommended for research & flexibility)",
                        "TensorFlow (recommended for production)",
                        "JAX (recommended for performance)",
                        "Scikit-learn (for classical ML)"
                    ],
                    default="PyTorch",
                    context_key="framework",
                    required=True
                ))
        
        # Add hardware question if missing (use template)
        if QuestionCategory.HARDWARE not in categories_present:
            gpu_template = self.template_loader.get_template('gpu_support')
            if gpu_template:
                questions.append(gpu_template)
            else:
                # Fallback to hardcoded question
                questions.append(Question(
                    id="Q_ESSENTIAL_HARDWARE",
                    category=QuestionCategory.HARDWARE,
                    text="Do you need GPU/CUDA support for training?",
                    priority=Priority.HIGH,
                    options=[
                        "Yes - I have NVIDIA GPU",
                        "Yes - will use cloud GPU",
                        "No - CPU only",
                        "Optional - support both"
                    ],
                    default="Optional - support both",
                    context_key="gpu_support",
                    required=False
                ))
        
        return questions
    
    def _prioritize_questions(self, questions: List[Question]) -> List[Question]:
        """
        Prioritize questions by priority level.
        
        Order: BLOCKING -> HIGH -> MEDIUM -> LOW
        """
        return sorted(questions, key=lambda q: q.priority.value)
    
    async def ask_questions(
        self,
        questions: List[Question],
        interactive: bool = True
    ) -> Dict[str, Answer]:
        """
        Conduct interactive Q&A session.
        
        Modes:
        - Interactive: Ask user via CLI
        - Batch: Use defaults (for automation/testing)
        
        Args:
            questions: List of questions
            interactive: If True, prompt user; else use defaults
            
        Returns:
            Dictionary mapping question_id -> Answer
        """
        logger.info(f"Asking {len(questions)} questions (interactive={interactive})...")
        
        answers = {}
        
        for i, question in enumerate(questions, 1):
            logger.debug(f"Question {i}/{len(questions)}: {question.id}")
            
            if interactive:
                # Present question to user
                answer_text = await self._prompt_user(question, i, len(questions))
            else:
                # Use default answer
                answer_text = question.default or ""
                logger.info(f"Using default answer for {question.id}: {answer_text}")
            
            # Parse answer
            answer = await self._parse_answer(question, answer_text)
            
            # Validate
            if not answer.is_valid():
                logger.warning(f"Invalid answer for {question.id}: confidence={answer.confidence}")
                
                if question.required and interactive:
                    # Retry for required questions
                    logger.info("Retrying required question...")
                    answer_text = await self._retry_question(question)
                    answer = await self._parse_answer(question, answer_text)
                elif question.default:
                    # Use default
                    logger.info(f"Using default for {question.id}")
                    answer = await self._parse_answer(question, question.default)
            
            answers[question.id] = answer
            
            # Handle follow-up questions
            if question.follow_up_questions and answer.is_valid():
                follow_ups = self._get_follow_up_questions(question, answer)
                if follow_ups:
                    logger.info(f"Asking {len(follow_ups)} follow-up questions...")
                    follow_up_answers = await self.ask_questions(follow_ups, interactive)
                    answers.update(follow_up_answers)
        
        logger.info(f"Collected {len(answers)} answers")
        return answers
    
    async def _prompt_user(
        self,
        question: Question,
        current: int,
        total: int
    ) -> str:
        """
        Prompt user for answer (CLI implementation).
        
        In production, this would be replaced with UI integration.
        """
        print(f"\n{'='*80}")
        print(f"[Question {current}/{total}] Priority: {question.priority.name}")
        print(question.format_for_display())
        print(f"{'='*80}")
        
        answer = input("> ").strip()
        
        # If empty and has default, use default
        if not answer and question.default:
            answer = question.default
            print(f"  Using default: {answer}")
        
        return answer
    
    async def _retry_question(self, question: Question) -> str:
        """Retry asking a question after invalid answer"""
        print("\n⚠️  Invalid answer. Please try again.")
        return await self._prompt_user(question, 0, 0)
    
    async def _parse_answer(
        self,
        question: Question,
        answer_text: str
    ) -> Answer:
        """
        Parse user answer into structured Answer object.
        
        Parsing strategies:
        1. Exact match with options
        2. Fuzzy match with options
        3. Extract key information
        4. LLM-based parsing for complex answers
        
        Args:
            question: The question being answered
            answer_text: Raw user answer text
            
        Returns:
            Parsed Answer object with confidence score
        """
        if not answer_text or not answer_text.strip():
            return Answer(
                question_id=question.id,
                raw_text="",
                parsed_value=None,
                confidence=0.0
            )
        
        answer_text = answer_text.strip()
        
        # Strategy 1: Exact match with options
        if question.has_options():
            parsed, confidence = self._parse_from_options(question, answer_text)
            if parsed:
                return Answer(
                    question_id=question.id,
                    raw_text=answer_text,
                    parsed_value=parsed,
                    confidence=confidence
                )
        
        # Strategy 2: Category-specific parsing
        parsed, confidence = self._parse_by_category(question, answer_text)
        
        return Answer(
            question_id=question.id,
            raw_text=answer_text,
            parsed_value=parsed,
            confidence=confidence
        )
    
    def _parse_from_options(
        self,
        question: Question,
        answer_text: str
    ) -> Tuple[Optional[str], float]:
        """Parse answer by matching with predefined options"""
        if not question.options:
            return None, 0.0
        
        answer_lower = answer_text.lower()
        
        # Check if answer is a number (option index)
        if answer_text.isdigit():
            idx = int(answer_text) - 1
            if 0 <= idx < len(question.options):
                return question.options[idx], 1.0
        
        # Exact match
        for option in question.options:
            if answer_text == option:
                return option, 1.0
        
        # Case-insensitive match
        for option in question.options:
            if answer_lower == option.lower():
                return option, 0.95
        
        # Substring match
        for option in question.options:
            if answer_lower in option.lower() or option.lower() in answer_lower:
                return option, 0.85
        
        # Keyword match
        for option in question.options:
            option_keywords = set(option.lower().split())
            answer_keywords = set(answer_lower.split())
            overlap = option_keywords & answer_keywords
            
            if overlap:
                confidence = len(overlap) / max(len(option_keywords), len(answer_keywords))
                if confidence > 0.5:
                    return option, confidence
        
        return None, 0.0
    
    def _parse_by_category(
        self,
        question: Question,
        answer_text: str
    ) -> Tuple[Any, float]:
        """Parse answer based on question category"""
        answer_lower = answer_text.lower()
        
        if question.category == QuestionCategory.FRAMEWORK:
            return self._parse_framework(answer_lower)
        
        elif question.category == QuestionCategory.HARDWARE:
            return self._parse_hardware(answer_lower)
        
        elif question.category == QuestionCategory.DATA:
            return self._parse_data(answer_lower)
        
        elif question.category == QuestionCategory.ARCHITECTURE:
            return self._parse_architecture(answer_lower)
        
        elif question.category == QuestionCategory.DEPLOYMENT:
            return self._parse_deployment(answer_lower)
        
        elif question.category == QuestionCategory.TESTING:
            return self._parse_testing(answer_lower)
        
        else:
            # Generic string parsing
            return answer_text, 0.7
    
    def _parse_framework(self, answer: str) -> Tuple[str, float]:
        """Parse framework selection"""
        if "pytorch" in answer or "torch" in answer:
            return "pytorch", 0.95
        elif "tensorflow" in answer or "tf" in answer:
            return "tensorflow", 0.95
        elif "jax" in answer:
            return "jax", 0.95
        elif "sklearn" in answer or "scikit" in answer:
            return "sklearn", 0.95
        else:
            return "pytorch", 0.5  # Default with low confidence
    
    def _parse_hardware(self, answer: str) -> Tuple[Any, float]:
        """Parse hardware requirements"""
        if "gpu" in answer or "cuda" in answer or "nvidia" in answer:
            return True, 0.9
        elif "cpu" in answer or "no" in answer:
            return False, 0.9
        elif "both" in answer or "optional" in answer:
            return "optional", 0.85
        else:
            return "optional", 0.5
    
    def _parse_data(self, answer: str) -> Tuple[Any, float]:
        """Parse data handling preferences"""
        if "yes" in answer or "augment" in answer:
            return True, 0.85
        elif "no" in answer:
            return False, 0.85
        else:
            return True, 0.6  # Default to yes with medium confidence
    
    def _parse_architecture(self, answer: str) -> Tuple[str, float]:
        """Parse architecture preferences"""
        if "custom" in answer or "scratch" in answer:
            return "custom", 0.9
        elif "pretrained" in answer or "transfer" in answer:
            return "pretrained", 0.9
        elif "ensemble" in answer:
            return "ensemble", 0.9
        elif "automl" in answer or "auto" in answer:
            return "automl", 0.9
        else:
            return "custom", 0.6
    
    def _parse_deployment(self, answer: str) -> Tuple[str, float]:
        """Parse deployment target"""
        if "local" in answer or "development" in answer:
            return "local", 0.9
        elif "cloud" in answer or "aws" in answer or "gcp" in answer or "azure" in answer:
            return "cloud", 0.9
        elif "edge" in answer or "mobile" in answer or "iot" in answer:
            return "edge", 0.9
        elif "api" in answer or "production" in answer:
            return "api", 0.9
        else:
            return "local", 0.6
    
    def _parse_testing(self, answer: str) -> Tuple[str, float]:
        """Parse testing requirements"""
        if "comprehensive" in answer or "e2e" in answer or "end-to-end" in answer:
            return "comprehensive", 0.9
        elif "standard" in answer or "integration" in answer:
            return "standard", 0.9
        elif "basic" in answer or "unit" in answer:
            return "basic", 0.9
        elif "minimal" in answer or "smoke" in answer:
            return "minimal", 0.9
        else:
            return "standard", 0.6
    
    def _get_follow_up_questions(
        self,
        question: Question,
        answer: Answer
    ) -> List[Question]:
        """Get follow-up questions based on answer"""
        # TODO: Implement follow-up question logic
        # For now, return empty list
        return []
    
    async def update_context(
        self,
        original_request: str,
        answers: Dict[str, Answer]
    ) -> ProjectContext:
        """
        Update project context with clarified information.
        
        Process:
        1. Initialize context with original request
        2. Map answers to context fields
        3. Apply defaults for unanswered questions
        4. Validate completeness
        5. Calculate confidence score
        
        Args:
            original_request: Original user input
            answers: Answered questions
            
        Returns:
            Updated ProjectContext
        """
        logger.info("Updating project context with answers...")
        
        context = ProjectContext(original_request=original_request)
        
        # Map answers to context fields
        for answer in answers.values():
            if not answer.is_valid():
                continue
            
            # Find the question to get context_key
            question = self._find_question_by_id(answer.question_id)
            if not question:
                logger.warning(f"Question not found for answer: {answer.question_id}")
                continue
            
            context_key = question.context_key
            parsed_value = answer.parsed_value
            
            # Map to context field
            if hasattr(context, context_key):
                setattr(context, context_key, parsed_value)
                logger.debug(f"Set {context_key} = {parsed_value}")
            else:
                logger.warning(f"Context has no field: {context_key}")
        
        # Calculate confidence score
        valid_answers = [a for a in answers.values() if a.is_valid()]
        if valid_answers:
            context.confidence_score = sum(a.confidence for a in valid_answers) / len(valid_answers)
        else:
            context.confidence_score = 0.0
        
        # Validate context
        is_valid, errors = context.validate()
        context.clarification_complete = is_valid
        context.unanswered_questions = errors
        
        logger.info(f"Context updated. Confidence: {context.confidence_score:.2f}, Valid: {is_valid}")
        
        return context
    
    def _find_question_by_id(self, question_id: str) -> Optional[Question]:
        """Find question by ID from templates or cache"""
        # TODO: Implement question caching
        # For now, try to infer from question_id
        
        if "FRAMEWORK" in question_id:
            return Question(
                id=question_id,
                category=QuestionCategory.FRAMEWORK,
                text="Framework",
                priority=Priority.BLOCKING,
                context_key="framework"
            )
        elif "HARDWARE" in question_id or "GPU" in question_id:
            return Question(
                id=question_id,
                category=QuestionCategory.HARDWARE,
                text="Hardware",
                priority=Priority.HIGH,
                context_key="gpu_support"
            )
        else:
            # Try to extract category from ID
            for category in QuestionCategory:
                if category.value in question_id.lower():
                    return Question(
                        id=question_id,
                        category=category,
                        text="Question",
                        priority=Priority.MEDIUM,
                        context_key=category.value
                    )
        
        return None
    
    def load_question_templates(self, templates: Dict[str, Question]):
        """Load question templates from external source"""
        self.question_templates = templates
        logger.info(f"Loaded {len(templates)} question templates")
    
    def set_question_limits(self, min_questions: int, max_questions: int):
        """Set min/max number of questions to ask"""
        self.min_questions = min_questions
        self.max_questions = max_questions
        logger.info(f"Question limits: {min_questions}-{max_questions}")
