"""
Tools Integration:
- ProblemChecker: Validates generated code (via CodingAgent)
- TestFailureHandler: Analyzes test failures and suggests fixes
- Task Runner: Executes tests with proper output management

Authors: GaneshS 271123
gmail:ganeshnaik214@gmail.com
"""

import time
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
from loguru import logger

from ..agents.coding.coding_agent import CodingAgent
from ..agents.planning.planning_agent import PlanningAgent
from ..agents.interactive.clarification_agent import ClarificationAgent
from ..agents.prompt_engineering.prompt_engineer_agent import PromptEngineerAgent
from ..agents.review.readability_reviewer import ReadabilityReviewer
from ..agents.review.logic_flow_reviewer import LogicFlowReviewer
from ..agents.review.code_connectivity_reviewer import CodeConnectivityReviewer
from ..agents.review.project_connectivity_reviewer import ProjectConnectivityReviewer
from ..agents.review.performance_reviewer import PerformanceReviewer
from ..agents.review.security_reviewer import SecurityReviewer
from ..agents.review.scoring_system import ScoringSystem, AggregatedScore, QualityLevel
from ..tools.test_failure_handler import TestFailureHandler, FailureSeverity
from ..tools.task_runner import TaskRunner


class IterationStatus(Enum):
    """Status of iteration workflow."""
    NOT_STARTED = "not_started"
    GENERATING = "generating"
    REVIEWING = "reviewing"
    AGGREGATING = "aggregating"
    DECIDING = "deciding"
    COMPLETED = "completed"
    FAILED = "failed"
    MAX_ITERATIONS_REACHED = "max_iterations_reached"
    QUALITY_ACHIEVED = "quality_achieved"


@dataclass
class IterationResult:
    """
    Result from a single iteration.
    
    Attributes:
        iteration_number: Which iteration (1-based)
        code: Generated code
        aggregated_score: Score from ScoringSystem
        status: Iteration status
        timestamp: When iteration completed
        duration: Time taken in seconds
        improvements: Changes from previous iteration
    """
    iteration_number: int
    code: str
    aggregated_score: AggregatedScore
    status: IterationStatus
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    duration: float = 0.0
    improvements: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "iteration_number": self.iteration_number,
            "code_length": len(self.code),
            "overall_score": self.aggregated_score.overall_score,
            "quality_level": self.aggregated_score.quality_level.value,
            "status": self.status.value,
            "timestamp": self.timestamp,
            "duration": self.duration,
            "improvements": self.improvements,
            "critical_issues": self.aggregated_score.critical_issues,
            "total_issues": self.aggregated_score.total_issues
        }


@dataclass
class WorkflowResult:
    """
    Final result from auto-iteration workflow.
    
    Attributes:
        success: Whether workflow succeeded
        final_code: Best code generated
        best_score: Highest score achieved
        iterations: All iteration results
        total_duration: Total time taken
        stop_reason: Why workflow stopped
        improvement_trajectory: Score progression
    """
    success: bool
    final_code: str
    best_score: AggregatedScore
    iterations: List[IterationResult]
    total_duration: float
    stop_reason: str
    improvement_trajectory: List[float] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "success": self.success,
            "final_code_length": len(self.final_code),
            "best_overall_score": self.best_score.overall_score,
            "best_quality_level": self.best_score.quality_level.value,
            "total_iterations": len(self.iterations),
            "total_duration": self.total_duration,
            "stop_reason": self.stop_reason,
            "improvement_trajectory": self.improvement_trajectory,
            "iterations": [it.to_dict() for it in self.iterations]
        }


class AutoIterationWorkflow:
    """
    Automated code generation and quality improvement workflow.
    
    This workflow orchestrates:
    1. Code generation with CodingAgent
    2. Multi-dimensional review with 6 specialized reviewers (Phase 2.5)
    3. Score aggregation with ScoringSystem
    4. Quality-based iteration decisions
    5. Improvement tracking and optimization
    
    Reviewers (Phase 2.5):
    - ReadabilityReviewer: Code style, naming, comments
    - LogicFlowReviewer: Control flow, error handling
    - CodeConnectivityReviewer: Function dependencies, modularity
    - ProjectConnectivityReviewer: Import graph, architecture
    - PerformanceReviewer: Time/space complexity, optimization
    - SecurityReviewer: Vulnerabilities, security best practices
    
    Features:
    - Configurable quality threshold (default 75.0)
    - Max iteration limit (default 3)
    - Smart stopping criteria (quality achieved or diminishing returns)
    - Comprehensive improvement tracking
    - Best code selection across iterations
    """
    
    def __init__(
        self,
        prompt_engineer_agent: Optional[PromptEngineerAgent] = None,
        planning_agent: Optional[PlanningAgent] = None,
        clarification_agent: Optional[ClarificationAgent] = None,
        coding_agent: Optional[CodingAgent] = None,
        readability_reviewer: Optional[ReadabilityReviewer] = None,
        logic_flow_reviewer: Optional[LogicFlowReviewer] = None,
        code_connectivity_reviewer: Optional[CodeConnectivityReviewer] = None,
        project_connectivity_reviewer: Optional[ProjectConnectivityReviewer] = None,
        performance_reviewer: Optional[PerformanceReviewer] = None,
        security_reviewer: Optional[SecurityReviewer] = None,
        scoring_system: Optional[ScoringSystem] = None,
        quality_threshold: float = 75.0,
        max_iterations: int = 3,
        min_improvement: float = 2.0,
        enable_test_execution: bool = False,
        test_directory: Optional[Path] = None,
        enable_prompt_engineering: bool = True,
        enable_dynamic_planning: bool = True,
        enable_smart_clarification: bool = True
    ):
        """
        Initialize auto-iteration workflow (Week 3 Architecture).
        
        Args:
            prompt_engineer_agent: Agent for prompt enhancement (Week 3)
            planning_agent: Agent for dynamic architecture planning (Week 3)
            clarification_agent: Agent for smart clarification questions (Week 3)
            coding_agent: Agent for code generation (with ProblemChecker validation)
            readability_reviewer: Reviewer for code readability
            logic_flow_reviewer: Reviewer for logic flow
            code_connectivity_reviewer: Reviewer for code connectivity
            project_connectivity_reviewer: Reviewer for project connectivity
            performance_reviewer: Reviewer for performance (Phase 2.5)
            security_reviewer: Reviewer for security (Phase 2.5)
            scoring_system: System for score aggregation
            quality_threshold: Minimum acceptable score (0-100)
            max_iterations: Maximum iterations allowed
            min_improvement: Minimum score improvement to continue (points)
            enable_test_execution: Enable test execution with failure analysis
            test_directory: Directory containing test files (for test execution)
            enable_prompt_engineering: Enable Week 3 prompt engineering (default True)
            enable_dynamic_planning: Enable Week 3 dynamic planning (default True)
            enable_smart_clarification: Enable Week 3 smart clarification (default True)
        """
        # Initialize LLM client and prompt manager if not provided
        from src.llm.gemini_client import GeminiClient
        from src.llm.prompt_manager import PromptManager
        
        llm_client = GeminiClient()
        prompt_manager = PromptManager()
        
        # Week 3 agents (new architecture)
        self.prompt_engineer_agent = prompt_engineer_agent or PromptEngineerAgent(llm_client, prompt_manager)
        self.planning_agent = planning_agent or PlanningAgent(llm_client=llm_client, prompt_manager=prompt_manager)
        self.clarification_agent = clarification_agent or ClarificationAgent()
        
        # Original agents
        self.coding_agent = coding_agent or CodingAgent()
        self.readability_reviewer = readability_reviewer or ReadabilityReviewer()
        self.logic_flow_reviewer = logic_flow_reviewer or LogicFlowReviewer()
        self.code_connectivity_reviewer = code_connectivity_reviewer or CodeConnectivityReviewer()
        self.project_connectivity_reviewer = project_connectivity_reviewer or ProjectConnectivityReviewer()
        self.performance_reviewer = performance_reviewer or PerformanceReviewer()
        self.security_reviewer = security_reviewer or SecurityReviewer()
        self.scoring_system = scoring_system or ScoringSystem()
        
        self.quality_threshold = quality_threshold
        self.max_iterations = max_iterations
        self.min_improvement = min_improvement
        
        # Week 3 feature flags
        self.enable_prompt_engineering = enable_prompt_engineering
        self.enable_dynamic_planning = enable_dynamic_planning
        self.enable_smart_clarification = enable_smart_clarification
        
        # Test execution tools (Phase 2.5)
        self.enable_test_execution = enable_test_execution
        self.test_directory = test_directory
        if enable_test_execution:
            self.test_failure_handler = TestFailureHandler()
            self.task_runner = TaskRunner()
            logger.info("Test execution enabled with TestFailureHandler and TaskRunner")
        else:
            self.test_failure_handler = None
            self.task_runner = None
        
        # Store enhanced specification (Week 3)
        self._enhanced_specification = None
        self._dynamic_architecture = None
        self._clarification_answers = {}
        
        logger.info(
            f"AutoIterationWorkflow initialized (Week 3 Architecture): "
            f"threshold={quality_threshold}, "
            f"max_iterations={max_iterations}, "
            f"min_improvement={min_improvement}, "
            f"test_execution={'enabled' if enable_test_execution else 'disabled'}, "
            f"prompt_engineering={'enabled' if enable_prompt_engineering else 'disabled'}, "
            f"dynamic_planning={'enabled' if enable_dynamic_planning else 'disabled'}, "
            f"smart_clarification={'enabled' if enable_smart_clarification else 'disabled'}"
        )
    
    async def _enhance_user_input(self, user_input: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Phase 1a (Week 3): Enhance vague user input into detailed specification.
        
        Args:
            user_input: Original vague input (e.g., "digit classifier")
            context: Additional context
            
        Returns:
            Enhanced specification with requirements, technical specs, ambiguities
        """
        if not self.enable_prompt_engineering:
            logger.info("Prompt engineering disabled, skipping enhancement")
            return {
                "original_input": user_input,
                "analyzed_task": {"description": user_input},
                "requirements": [],
                "technical_specs": {},
                "ambiguities": [],
                "suggested_architecture": {},
                "confidence": 100
            }
        
        logger.info("=== PHASE 1a: PROMPT ENGINEERING ===")
        logger.info(f"Enhancing user input: {user_input[:100]}...")
        
        try:
            specification = await self.prompt_engineer_agent.enhance_user_input(user_input, context)
            self._enhanced_specification = specification
            
            logger.info(f"✅ Prompt engineering complete:")
            logger.info(f"   - Confidence: {specification.get('confidence', 0)}%")
            logger.info(f"   - Requirements: {len(specification.get('requirements', []))} identified")
            logger.info(f"   - Ambiguities: {len(specification.get('ambiguities', []))} detected")
            
            return specification
            
        except Exception as e:
            logger.error(f"Prompt engineering failed: {e}")
            # Return fallback specification
            return {
                "original_input": user_input,
                "analyzed_task": {"description": user_input},
                "requirements": [],
                "technical_specs": {},
                "ambiguities": [],
                "suggested_architecture": {},
                "confidence": 50
            }
    
    async def _design_dynamic_architecture(self, specification: Dict[str, Any]) -> Dict[str, Any]:
        """
        Phase 1b (Week 3): Design dynamic project architecture (not hardcoded).
        
        Args:
            specification: Enhanced specification from prompt engineering
            
        Returns:
            Dynamic architecture with file structure, classes, modules
        """
        if not self.enable_dynamic_planning:
            logger.info("Dynamic planning disabled, skipping architecture design")
            return {
                "structure_type": "modular",
                "files": [],
                "config_files": [],
                "total_estimated_files": 3,
                "total_estimated_lines": 500
            }
        
        logger.info("=== PHASE 1b: DYNAMIC PLANNING ===")
        logger.info("Designing dynamic architecture (LLM-driven)...")
        
        try:
            architecture = await self.planning_agent.design_dynamic_architecture(specification)
            self._dynamic_architecture = architecture
            
            logger.info(f"✅ Dynamic architecture designed:")
            logger.info(f"   - Structure: {architecture.get('structure_type', 'unknown')}")
            logger.info(f"   - Files: {len(architecture.get('files', []))} planned")
            logger.info(f"   - Total lines: ~{architecture.get('total_estimated_lines', 0)}")
            
            return architecture
            
        except Exception as e:
            logger.error(f"Dynamic planning failed: {e}")
            # Return fallback architecture
            return self.planning_agent._get_fallback_architecture(specification)
    
    async def _generate_smart_questions(
        self, 
        specification: Dict[str, Any], 
        architecture: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Phase 1c (Week 3): Generate smart, context-aware clarification questions.
        
        Args:
            specification: Enhanced specification
            architecture: Dynamic architecture
            
        Returns:
            List of smart questions with options and priorities
        """
        if not self.enable_smart_clarification:
            logger.info("Smart clarification disabled, skipping questions")
            return []
        
        logger.info("=== PHASE 1c: SMART CLARIFICATION ===")
        logger.info("Generating context-aware questions...")
        
        try:
            ambiguities = specification.get('ambiguities', [])
            if not ambiguities:
                logger.info("No ambiguities detected, skipping clarification")
                return []
            
            questions = await self.clarification_agent.generate_smart_questions(
                specification=specification,
                ambiguities=ambiguities
            )
            
            # Prioritize questions
            prioritized = await self.clarification_agent.prioritize_questions_intelligently(
                questions=questions,
                user_context={"architecture": architecture}
            )
            
            logger.info(f"✅ Smart clarification complete:")
            logger.info(f"   - Questions generated: {len(prioritized)}")
            if prioritized:
                critical_count = sum(1 for q in prioritized if q.get('priority') == 'critical')
                logger.info(f"   - Critical questions: {critical_count}")
            
            return prioritized
            
        except Exception as e:
            logger.error(f"Smart clarification failed: {e}")
            return []
    
    async def _ask_questions_interactive(self, questions: List[Dict[str, Any]]) -> Dict[str, str]:
        """
        Phase 1d (Week 3): Ask user clarification questions interactively.
        
        Args:
            questions: List of smart questions
            
        Returns:
            Dict mapping question IDs to answers
        """
        if not questions:
            logger.info("No questions to ask")
            return {}
        
        logger.info("=== PHASE 1d: USER INTERACTION ===")
        logger.info(f"Asking {len(questions)} clarification questions...")
        
        answers = {}
        for i, question in enumerate(questions, 1):
            question_text = question.get('question', '')
            question_id = question.get('id', f'q{i}')
            options = question.get('options', [])
            priority = question.get('priority', 'medium')
            
            print(f"\n{'🔴' if priority == 'critical' else '🟡' if priority == 'high' else '🟢'} Question {i}/{len(questions)} [{priority.upper()}]:")
            print(f"   {question_text}")
            
            if options:
                print("   Options:")
                for j, option in enumerate(options, 1):
                    print(f"      {j}. {option}")
                print("   (Enter option number or custom answer)")
            
            # For now, auto-select first option (in production, get user input)
            # TODO: Replace with actual user input mechanism
            if options:
                answer = options[0]
                logger.info(f"Auto-selected: {answer}")
            else:
                answer = "default"
                logger.info(f"Using default answer")
            
            answers[question_id] = answer
        
        self._clarification_answers = answers
        logger.info(f"✅ Collected {len(answers)} answers")
        
        return answers
    
    def _merge_specification_with_answers(
        self,
        specification: Dict[str, Any],
        architecture: Dict[str, Any],
        answers: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Phase 1e (Week 3): Merge enhanced specification with user answers.
        
        Args:
            specification: Enhanced specification
            architecture: Dynamic architecture
            answers: User answers
            
        Returns:
            Final merged specification for code generation
        """
        logger.info("=== PHASE 1e: SPECIFICATION FINALIZATION ===")
        logger.info("Merging specification with user answers...")
        
        merged = {
            "original_input": specification.get('original_input', ''),
            "analyzed_task": specification.get('analyzed_task', {}),
            "requirements": specification.get('requirements', []),
            "technical_specs": specification.get('technical_specs', {}),
            "architecture": architecture,
            "clarifications": answers,
            "confidence": specification.get('confidence', 0)
        }
        
        logger.info("✅ Final specification ready for code generation")
        logger.info(f"   - Requirements: {len(merged['requirements'])}")
        logger.info(f"   - Files to generate: {len(architecture.get('files', []))}")
        logger.info(f"   - Clarifications: {len(answers)}")
        
        return merged
    
    def execute(
        self,
        task_description: str,
        project_files: Optional[Dict[str, str]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> WorkflowResult:
        """
        Execute auto-iteration workflow (Week 3 Architecture).
        
        NEW WORKFLOW:
        Phase 1a: Prompt Engineering (enhance vague input)
        Phase 1b: Dynamic Planning (LLM decides structure)
        Phase 1c: Smart Clarification (context-aware questions)
        Phase 1d: User Interaction (collect answers)
        Phase 1e: Specification Finalization (merge everything)
        Phase 2+: Code Generation & Iteration (existing workflow)
        
        Args:
            task_description: What code to generate (can be vague)
            project_files: Existing project files for context
            context: Additional context (requirements, constraints, etc.)
            
        Returns:
            WorkflowResult with best code and improvement history
        """
        logger.info(f"Starting auto-iteration workflow (Week 3): {task_description[:100]}...")
        start_time = time.time()
        
        iterations: List[IterationResult] = []
        best_code = ""
        best_score = None
        improvement_trajectory = []
        
        try:
            # ============================================
            # WEEK 3 PHASES (NEW) - Run ONCE before iteration
            # ============================================
            
            # Helper to run async code (handles both sync and async contexts)
            import asyncio
            from concurrent.futures import ThreadPoolExecutor
            
            def run_async(coro, timeout=180):
                """Run async coroutine, handling both sync and async contexts."""
                try:
                    loop = asyncio.get_running_loop()
                    # We're in an event loop - run in thread pool
                    with ThreadPoolExecutor() as executor:
                        future = executor.submit(asyncio.run, coro)
                        return future.result(timeout=timeout)
                except RuntimeError:
                    # No running loop - use asyncio.run()
                    return asyncio.run(coro)
            
            # Phase 1a: Prompt Engineering (can take 1-2 minutes)
            specification = run_async(self._enhance_user_input(task_description, context), timeout=180)
            
            # Phase 1b: Dynamic Planning (usually < 1 minute)
            architecture = run_async(self._design_dynamic_architecture(specification), timeout=120)
            
            # Phase 1c: Smart Clarification (usually < 1 minute)
            questions = run_async(self._generate_smart_questions(specification, architecture), timeout=120)
            
            # Phase 1d: User Interaction (5 minutes for user to answer)
            answers = run_async(self._ask_questions_interactive(questions), timeout=300)
            
            # Phase 1e: Merge specification
            final_specification = self._merge_specification_with_answers(
                specification, architecture, answers
            )
            
            # Update context with final specification for code generation
            enhanced_context = context or {}
            enhanced_context['specification'] = final_specification
            enhanced_context['architecture'] = architecture
            enhanced_context['requirements'] = specification.get('requirements', [])
            
            logger.info("=" * 60)
            logger.info("Week 3 phases complete, starting code generation iterations...")
            logger.info("=" * 60)
            
            # ============================================
            # ITERATION LOOP (EXISTING) - Enhanced with Week 3 context
            # ============================================
            
            for iteration_num in range(1, self.max_iterations + 1):
                logger.info(f"=== Iteration {iteration_num}/{self.max_iterations} ===")
                iteration_start = time.time()
                
                # Step 1: Generate code (with Week 3 enhanced context)
                code = self._generate_code(
                    task_description,
                    iteration_num,
                    iterations,
                    enhanced_context  # Use enhanced context from Week 3
                )
                
                if not code:
                    logger.error(f"Code generation failed in iteration {iteration_num}")
                    break
                
                # Step 2: Review code
                aggregated_score = self._review_code(code, project_files)
                
                if not aggregated_score:
                    logger.error(f"Code review failed in iteration {iteration_num}")
                    break
                
                # Step 3: Track iteration
                iteration_duration = time.time() - iteration_start
                improvements = self._calculate_improvements(iterations, aggregated_score)
                
                iteration_result = IterationResult(
                    iteration_number=iteration_num,
                    code=code,
                    aggregated_score=aggregated_score,
                    status=IterationStatus.COMPLETED,
                    duration=iteration_duration,
                    improvements=improvements
                )
                iterations.append(iteration_result)
                improvement_trajectory.append(aggregated_score.overall_score)
                
                logger.info(
                    f"Iteration {iteration_num} complete: "
                    f"Score={aggregated_score.overall_score:.1f}, "
                    f"Quality={aggregated_score.quality_level.value}, "
                    f"Issues={aggregated_score.total_issues}"
                )
                
                # Step 4: Update best code
                if best_score is None or aggregated_score.overall_score > best_score.overall_score:
                    best_code = code
                    best_score = aggregated_score
                    logger.info(f"New best score: {best_score.overall_score:.1f}")
                
                # Step 5: Decide whether to continue
                should_continue, stop_reason = self._should_continue(
                    iteration_num,
                    aggregated_score,
                    iterations
                )
                
                if not should_continue:
                    logger.info(f"Stopping workflow: {stop_reason}")
                    total_duration = time.time() - start_time
                    
                    return WorkflowResult(
                        success=True,
                        final_code=best_code,
                        best_score=best_score,
                        iterations=iterations,
                        total_duration=total_duration,
                        stop_reason=stop_reason,
                        improvement_trajectory=improvement_trajectory
                    )
            
            # Max iterations reached
            total_duration = time.time() - start_time
            stop_reason = f"Max iterations ({self.max_iterations}) reached"
            
            logger.info(f"Workflow complete: {stop_reason}")
            
            return WorkflowResult(
                success=best_score is not None,
                final_code=best_code,
                best_score=best_score,
                iterations=iterations,
                total_duration=total_duration,
                stop_reason=stop_reason,
                improvement_trajectory=improvement_trajectory
            )
            
        except Exception as e:
            logger.exception(f"Workflow failed: {e}")
            total_duration = time.time() - start_time
            
            return WorkflowResult(
                success=False,
                final_code=best_code if best_code else "",
                best_score=best_score if best_score else None,
                iterations=iterations,
                total_duration=total_duration,
                stop_reason=f"Error: {str(e)}",
                improvement_trajectory=improvement_trajectory
            )
    
    def _generate_code(
        self,
        task_description: str,
        iteration_num: int,
        previous_iterations: List[IterationResult],
        context: Optional[Dict[str, Any]]
    ) -> str:
        """
        Generate code, incorporating feedback from previous iterations.
        
        Args:
            task_description: Original task description
            iteration_num: Current iteration number
            previous_iterations: Results from previous iterations
            context: Additional context
            
        Returns:
            Generated code string
        """
        logger.info(f"Generating code (iteration {iteration_num})...")
        
        # Build enhanced task description with feedback
        enhanced_description = task_description
        
        if previous_iterations:
            last_iteration = previous_iterations[-1]
            score = last_iteration.aggregated_score
            
            # Add feedback from previous iteration
            feedback_parts = [
                f"\n\n=== ITERATION {iteration_num} - IMPROVEMENT FEEDBACK ===",
                f"Previous Score: {score.overall_score:.1f}/100 ({score.quality_level.value})",
                f"Issues Found: {score.total_issues}",
                ""
            ]
            
            if score.critical_issues:
                feedback_parts.append("CRITICAL ISSUES TO FIX:")
                for i, issue in enumerate(score.critical_issues, 1):
                    feedback_parts.append(f"  {i}. {issue}")
                feedback_parts.append("")
            
            if score.recommendations:
                feedback_parts.append("RECOMMENDATIONS:")
                for i, rec in enumerate(score.recommendations[:5], 1):
                    feedback_parts.append(f"  {i}. {rec}")
                feedback_parts.append("")
            
            # Add dimension-specific feedback
            feedback_parts.append("DIMENSION SCORES:")
            for dimension, dim_score in score.dimension_scores.items():
                status = "✅" if dim_score >= 75 else "⚠️" if dim_score >= 60 else "❌"
                feedback_parts.append(f"  {status} {dimension}: {dim_score:.1f}/100")
            
            enhanced_description += "\n".join(feedback_parts)
        
        # Generate code (Week 3: Use enhanced specification if available)
        try:
            # Check if Week 3 specification is available
            specification = context.get('specification', {}) if context else {}
            architecture = context.get('architecture', {}) if context else {}
            
            # Build comprehensive task params with Week 3 enhancements
            task_params = {
                "task_type": "generate_module",
                "data": {
                    "module_name": "generated_code",
                    "purpose": enhanced_description,
                    "requirements": context.get("requirements", "") if context else "",
                    "language": context.get("language", "python") if context else "python"
                }
            }
            
            # Add Week 3 specification if available
            if specification:
                task_params["data"]["specification"] = specification
                task_params["data"]["architecture"] = architecture
                logger.info("Using Week 3 enhanced specification for code generation")
            else:
                logger.info("Using legacy code generation (no Week 3 specification)")
            
            # Check if we're in an event loop
            import asyncio
            try:
                loop = asyncio.get_running_loop()
                # We're in an event loop, run in thread
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        lambda: asyncio.run(self.coding_agent.execute_task(task_params))
                    )
                    result = future.result()
            except RuntimeError:
                # No running loop, safe to use asyncio.run()
                result = asyncio.run(self.coding_agent.execute_task(task_params))
            
            if result.get("status") == "success":
                code = result.get("code", "")
                # Debug: Log first 500 chars of generated code to check format
                logger.debug(f"Generated code preview (first 500 chars):\n{code[:500]}")
                return code
            else:
                logger.error(f"Code generation failed: {result.get('message', 'Unknown error')}")
                return ""
                
        except Exception as e:
            logger.exception(f"Error generating code: {e}")
            return ""
    
    async def _review_code_async(
        self,
        code: str,
        project_files: Optional[Dict[str, str]]
    ) -> Optional[AggregatedScore]:
        """
        Review code with all 6 reviewers and aggregate scores (async version).
        
        Args:
            code: Code to review
            project_files: Project context files
            
        Returns:
            AggregatedScore or None if review failed
        """
        logger.info("Reviewing code with all 6 reviewers...")
        
        try:
            # Run all 6 reviewers (all are async)
            logger.debug("Calling ReadabilityReviewer...")
            readability_result = await self.readability_reviewer.execute_task({
                "task_type": "analyze_readability",
                "data": {"code": code}
            })
            logger.debug(f"ReadabilityReviewer result: success={readability_result.get('success')}, score={readability_result.get('score')}")
            
            logger.debug("Calling LogicFlowReviewer...")
            logic_flow_result = await self.logic_flow_reviewer.execute_task({
                "task_type": "analyze_logic_flow",
                "data": {"code": code}
            })
            logger.debug(f"LogicFlowReviewer result: success={logic_flow_result.get('success')}, score={logic_flow_result.get('score')}")
            
            logger.debug("Calling CodeConnectivityReviewer...")
            code_conn_result = await self.code_connectivity_reviewer.execute_task({
                "task_type": "analyze_connectivity",
                "data": {"code": code}
            })
            logger.debug(f"CodeConnectivityReviewer result: success={code_conn_result.get('success')}, score={code_conn_result.get('score')}")
            
            # Performance reviewer (Phase 2.5) - async
            logger.debug("Calling PerformanceReviewer...")
            performance_result = await self.performance_reviewer.execute_task({
                "task_type": "review_performance",
                "data": {"code": code}
            })
            logger.debug(f"PerformanceReviewer result: success={performance_result.get('success')}, score={performance_result.get('score')}")
            
            # Security reviewer (Phase 2.5) - async
            logger.debug("Calling SecurityReviewer...")
            security_result = await self.security_reviewer.execute_task({
                "task_type": "review_security",
                "data": {"code": code}
            })
            logger.debug(f"SecurityReviewer result: success={security_result.get('success')}, score={security_result.get('score')}")
            
            # Project connectivity needs file context
            logger.debug("Calling ProjectConnectivityReviewer...")
            if project_files:
                project_conn_result = await self.project_connectivity_reviewer.execute_task({
                    "task_type": "analyze_project",
                    "data": {
                        "code": code,
                        "project_context": project_files
                    }
                })
                logger.debug(f"ProjectConnectivityReviewer result: success={project_conn_result.get('success')}, score={project_conn_result.get('score')}")
            else:
                project_conn_result = None
                logger.debug("ProjectConnectivityReviewer skipped (no project files)")
            
            # Aggregate scores from all 6 reviewers
            aggregated = self.scoring_system.aggregate_scores(
                readability_score=readability_result if readability_result.get("success", False) else None,
                logic_flow_score=logic_flow_result if logic_flow_result.get("success", False) else None,
                code_connectivity_score=code_conn_result if code_conn_result.get("success", False) else None,
                project_connectivity_score=project_conn_result if project_conn_result and project_conn_result.get("success", False) else None,
                performance_score=performance_result if performance_result.get("success", False) else None,
                security_score=security_result if security_result.get("success", False) else None
            )
            
            return aggregated
            
        except Exception as e:
            logger.exception(f"Error reviewing code: {e}")
            return None
    
    def _review_code(
        self,
        code: str,
        project_files: Optional[Dict[str, str]]
    ) -> Optional[AggregatedScore]:
        """
        Review code with all 6 reviewers and aggregate scores (synchronous wrapper).
        
        Args:
            code: Code to review
            project_files: Project context files
            
        Returns:
            AggregatedScore or None if review failed
        """
        import asyncio
        try:
            # Check if we're already in an event loop
            try:
                loop = asyncio.get_running_loop()
                # We're in an event loop, use new loop in thread
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        lambda: asyncio.run(self._review_code_async(code, project_files))
                    )
                    return future.result()
            except RuntimeError:
                # No running loop, safe to use asyncio.run()
                return asyncio.run(self._review_code_async(code, project_files))
        except Exception as e:
            logger.exception(f"Error in _review_code: {e}")
            return None
    
    def _calculate_improvements(
        self,
        previous_iterations: List[IterationResult],
        current_score: AggregatedScore
    ) -> Dict[str, Any]:
        """
        Calculate improvements from previous iteration.
        
        Args:
            previous_iterations: Previous iteration results
            current_score: Current aggregated score
            
        Returns:
            Dictionary with improvement metrics
        """
        if not previous_iterations:
            return {
                "is_first_iteration": True,
                "overall_delta": 0.0,
                "dimension_deltas": {},
                "issues_delta": 0
            }
        
        previous = previous_iterations[-1]
        prev_score = previous.aggregated_score
        
        # Use ScoringSystem's compare_scores
        comparison = self.scoring_system.compare_scores(prev_score, current_score)
        
        return {
            "is_first_iteration": False,
            "overall_delta": comparison["overall_delta"],
            "improved": comparison["improved"],
            "dimension_deltas": {
                dim: changes["delta"]
                for dim, changes in comparison["dimension_changes"].items()
            },
            "issues_delta": comparison["issues_delta"],
            "quality_level_change": f"{comparison['quality_level_before']} → {comparison['quality_level_after']}"
        }
    
    def _should_continue(
        self,
        iteration_num: int,
        current_score: AggregatedScore,
        iterations: List[IterationResult]
    ) -> tuple[bool, str]:
        """
        Decide whether to continue iterating.
        
        Args:
            iteration_num: Current iteration number
            current_score: Current aggregated score
            iterations: All iterations so far
            
        Returns:
            (should_continue, reason)
        """
        # Check if quality threshold achieved
        if self.scoring_system.is_acceptable_quality(current_score, self.quality_threshold):
            return (
                False,
                f"Quality threshold achieved: {current_score.overall_score:.1f} >= {self.quality_threshold}"
            )
        
        # Check if max iterations reached
        if iteration_num >= self.max_iterations:
            return (
                False,
                f"Max iterations ({self.max_iterations}) reached"
            )
        
        # Check for diminishing returns (if not first iteration)
        if len(iterations) >= 2:
            last_improvement = iterations[-1].improvements
            if last_improvement and not last_improvement.get("is_first_iteration"):
                overall_delta = last_improvement.get("overall_delta", 0)
                
                # If improvement is too small, stop
                if 0 < overall_delta < self.min_improvement:
                    return (
                        False,
                        f"Diminishing returns: improvement {overall_delta:.1f} < {self.min_improvement}"
                    )
                
                # If score is regressing, stop
                if overall_delta < 0:
                    return (
                        False,
                        f"Score regression detected: {overall_delta:.1f}"
                    )
        
        # Continue iterating
        return (
            True,
            f"Continue: score {current_score.overall_score:.1f} < {self.quality_threshold}, improvement possible"
        )
    
    def get_summary(self, result: WorkflowResult) -> str:
        """
        Generate human-readable workflow summary.
        
        Args:
            result: Workflow result
            
        Returns:
            Formatted summary string
        """
        lines = [
            "=" * 70,
            "AUTO-ITERATION WORKFLOW SUMMARY",
            "=" * 70,
            f"Success: {'✅ YES' if result.success else '❌ NO'}",
            f"Total Iterations: {len(result.iterations)}",
            f"Total Duration: {result.total_duration:.2f}s",
            f"Stop Reason: {result.stop_reason}",
            "",
            f"BEST RESULT:",
            f"  Overall Score: {result.best_score.overall_score:.1f}/100",
            f"  Quality Level: {result.best_score.quality_level.value.upper()}",
            f"  Total Issues: {result.best_score.total_issues}",
            f"  Critical Issues: {len(result.best_score.critical_issues)}",
            "",
            "IMPROVEMENT TRAJECTORY:",
        ]
        
        for i, score in enumerate(result.improvement_trajectory, 1):
            delta = ""
            if i > 1:
                prev_score = result.improvement_trajectory[i - 2]
                change = score - prev_score
                delta = f" ({change:+.1f})"
            lines.append(f"  Iteration {i}: {score:.1f}/100{delta}")
        
        if result.best_score.critical_issues:
            lines.append("")
            lines.append("REMAINING CRITICAL ISSUES:")
            for i, issue in enumerate(result.best_score.critical_issues, 1):
                lines.append(f"  {i}. {issue}")
        
        if result.best_score.recommendations:
            lines.append("")
            lines.append("RECOMMENDATIONS:")
            for i, rec in enumerate(result.best_score.recommendations[:5], 1):
                lines.append(f"  {i}. {rec}")
        
        lines.append("=" * 70)
        
        return "\n".join(lines)
    
    def run_tests_with_analysis(
        self,
        test_files: Optional[List[str]] = None,
        timeout: int = 300
    ) -> Dict[str, Any]:
        """
        Run tests and analyze failures using TestFailureHandler.
        
        Args:
            test_files: Specific test files to run (None = all tests in test_directory)
            timeout: Timeout for test execution in seconds
            
        Returns:
            Dictionary with:
                - success: Whether tests passed
                - output: Raw test output
                - failures: Parsed failure details
                - suggestions: Fix suggestions for failures
                - summary: Test result summary
        """
        if not self.enable_test_execution:
            logger.warning("Test execution not enabled")
            return {
                "success": False,
                "error": "Test execution not enabled. Set enable_test_execution=True"
            }
        
        if not self.test_directory or not self.test_directory.exists():
            logger.warning(f"Test directory not found: {self.test_directory}")
            return {
                "success": False,
                "error": f"Test directory not found: {self.test_directory}"
            }
        
        logger.info(f"Running tests in: {self.test_directory}")
        
        try:
            # Build pytest command
            if test_files:
                test_paths = [str(self.test_directory / f) for f in test_files]
            else:
                test_paths = [str(self.test_directory)]
            
            # Run tests using TaskRunner
            test_result = self.task_runner.run_pytest(
                test_paths=test_paths,
                markers=[],
                options=["-v", "--tb=short"],
                timeout=timeout
            )
            
            output = test_result.get("output", "")
            exit_code = test_result.get("exit_code", 1)
            
            # Parse failures using TestFailureHandler
            parsed_result = self.test_failure_handler.parse_pytest_output(output)
            
            # Generate failure report
            if parsed_result.has_failures:
                failure_report = self.test_failure_handler.generate_failure_report(parsed_result)
                logger.info(f"Test failures detected:\n{failure_report}")
                
                # Extract critical failures
                critical_failures = [
                    f for f in parsed_result.failures 
                    if f.severity in [FailureSeverity.CRITICAL, FailureSeverity.HIGH]
                ]
                
                # Collect fix suggestions
                suggestions = []
                for failure in parsed_result.failures:
                    if failure.suggested_fixes:
                        suggestions.extend([
                            {
                                "test": failure.test_name,
                                "error": failure.error_message,
                                "fix": fix
                            }
                            for fix in failure.suggested_fixes
                        ])
                
                return {
                    "success": False,
                    "output": output,
                    "exit_code": exit_code,
                    "passed": parsed_result.passed,
                    "failed": parsed_result.failed,
                    "skipped": parsed_result.skipped,
                    "failures": [
                        {
                            "test": f.test_name,
                            "location": f.location,
                            "error": f.error_message,
                            "type": f.failure_type.value,
                            "severity": f.severity.value,
                            "priority": f.priority
                        }
                        for f in parsed_result.failures
                    ],
                    "critical_failures": len(critical_failures),
                    "suggestions": suggestions,
                    "report": failure_report
                }
            else:
                logger.info(f"✅ All tests passed: {parsed_result.passed} passed")
                return {
                    "success": True,
                    "output": output,
                    "exit_code": exit_code,
                    "passed": parsed_result.passed,
                    "failed": 0,
                    "skipped": parsed_result.skipped,
                    "failures": [],
                    "suggestions": []
                }
        
        except Exception as e:
            logger.exception(f"Test execution failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "output": ""
            }
