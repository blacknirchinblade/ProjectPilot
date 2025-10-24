"""
Post-Generation Interactive Workflow

This orchestrates the complete post-generation cycle:
1. User runs generated code locally
2. Gets error/issue
3. ErrorFixingAgent analyzes and fixes
4. Triggers ReviewOrchestrator
5. Auto-iteration workflow runs
6. Repeat until code works

Flow:
    Generated Code
         ↓
    [User runs locally]
         ↓
    [Error occurs] ──→ ErrorFixingAgent
         ↓                    ↓
    [Fixes applied]    [Intelligent analysis]
         ↓                    ↓
    ReviewOrchestrator ← [Trigger review?]
         ↓
    Auto-iteration (5 cycles)
         ↓
    High-quality code ✅

Author: GaneshS 271123
gmail:ganeshnaik214@gmail.com
"""

import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional
from loguru import logger

from src.agents.interactive.error_fixing_agent import ErrorFixingAgent
from src.agents.interactive.contextual_change_agent import ContextualChangeAgent
from src.agents.review.review_orchestrator import ReviewOrchestrator
from src.agents.review.readability_reviewer import ReadabilityReviewer
from src.agents.review.logic_flow_reviewer import LogicFlowReviewer
from src.agents.review.code_connectivity_reviewer import CodeConnectivityReviewer
from src.agents.review.project_connectivity_reviewer import ProjectConnectivityReviewer
from src.agents.review.performance_reviewer import PerformanceReviewer
from src.agents.review.security_reviewer import SecurityReviewer


class PostGenerationWorkflow:
    """
    Complete post-generation interactive workflow with error fixing and review cycles
    """
    
    def __init__(
        self,
        project_root: str,
        max_fix_iterations: int = 3,
        max_review_iterations: int = 5
    ):
        """
        Initialize post-generation workflow
        
        Args:
            project_root: Project directory
            max_fix_iterations: Maximum error-fix cycles before giving up
            max_review_iterations: Maximum review iterations per fix
        """
        self.project_root = Path(project_root)
        self.max_fix_iterations = max_fix_iterations
        self.max_review_iterations = max_review_iterations
        
        # Initialize agents
        self.error_fixer = ErrorFixingAgent(project_root=str(project_root))
        self.contextual_changer = ContextualChangeAgent(project_root=str(project_root))
        
        # Initialize all 6 reviewers
        self.readability_reviewer = ReadabilityReviewer()
        self.logic_flow_reviewer = LogicFlowReviewer()
        self.code_connectivity_reviewer = CodeConnectivityReviewer()
        self.project_connectivity_reviewer = ProjectConnectivityReviewer()
        self.performance_reviewer = PerformanceReviewer()
        self.security_reviewer = SecurityReviewer()
        
        # Initialize ReviewOrchestrator with all reviewers
        self.review_orchestrator = ReviewOrchestrator(
            readability_reviewer=self.readability_reviewer,
            logic_flow_reviewer=self.logic_flow_reviewer,
            code_connectivity_reviewer=self.code_connectivity_reviewer,
            project_connectivity_reviewer=self.project_connectivity_reviewer,
            performance_reviewer=self.performance_reviewer,
            security_reviewer=self.security_reviewer
        )
        
        # Workflow state
        self.workflow_history = []
        self.current_iteration = 0
        
        logger.info("PostGenerationWorkflow initialized")
        logger.info(f"  - Project: {self.project_root}")
        logger.info(f"  - Max fix iterations: {max_fix_iterations}")
        logger.info(f"  - Max review iterations: {max_review_iterations}")
    
    async def handle_error(
        self,
        error_message: str,
        error_context: Optional[Dict[str, Any]] = None,
        auto_apply: bool = True,
        trigger_review: bool = True
    ) -> Dict[str, Any]:
        """
        Handle runtime error from user
        
        This is the main entry point when user reports an error
        
        Args:
            error_message: Error traceback/message from running code
            error_context: Context (command run, environment, etc)
            auto_apply: Automatically apply fixes
            trigger_review: Run review cycle after fixing
            
        Returns:
            {
                "status": "success" | "error" | "max_iterations",
                "fixes_applied": int,
                "review_cycles": int,
                "final_quality_score": float,
                "suggested_next_command": str,
                "workflow_summary": Dict
            }
        """
        logger.info("="*80)
        logger.info("POST-GENERATION ERROR HANDLING WORKFLOW")
        logger.info("="*80)
        logger.info(f"Iteration: {self.current_iteration + 1}/{self.max_fix_iterations}")
        
        try:
            # Step 1: Analyze and fix error
            logger.info("\n[1/4] Analyzing error...")
            fix_result = await self.error_fixer.fix_error(
                error_message=error_message,
                error_context=error_context,
                auto_apply=auto_apply
            )
            
            if fix_result["status"] != "success":
                return {
                    "status": "error",
                    "error": "Failed to analyze error",
                    "details": fix_result
                }
            
            fixes_applied = len(fix_result.get("applied_fixes", []))
            logger.info(f"✓ Applied {fixes_applied} fixes")
            
            # Step 2: Trigger review cycle if needed
            review_result = None
            if trigger_review and fix_result.get("needs_review", False):
                logger.info("\n[2/4] Triggering review cycle...")
                review_result = await self._run_review_cycle()
                logger.info(f"✓ Review complete - Quality: {review_result.get('final_score', 0):.1f}%")
            else:
                logger.info("\n[2/4] Skipping review (not needed for simple fixes)")
            
            # Step 3: Track iteration
            self.current_iteration += 1
            self.workflow_history.append({
                "iteration": self.current_iteration,
                "error_type": fix_result["error_analysis"].get("error_type"),
                "fixes_applied": fixes_applied,
                "review_triggered": review_result is not None,
                "quality_score": review_result.get("final_score") if review_result else None
            })
            
            # Step 4: Check if should continue
            if self.current_iteration >= self.max_fix_iterations:
                logger.warning(f"⚠️  Reached max iterations ({self.max_fix_iterations})")
                status = "max_iterations"
            else:
                status = "success"
            
            # Generate summary
            summary = self._generate_summary()
            
            logger.info("\n[4/4] Workflow complete!")
            logger.info(f"Status: {status}")
            logger.info(f"Total fixes: {sum(h['fixes_applied'] for h in self.workflow_history)}")
            logger.info(f"Review cycles: {sum(1 for h in self.workflow_history if h['review_triggered'])}")
            
            return {
                "status": status,
                "fixes_applied": fixes_applied,
                "review_cycles": 1 if review_result else 0,
                "final_quality_score": review_result.get("final_score") if review_result else None,
                "suggested_next_command": fix_result.get("suggested_command"),
                "workflow_summary": summary,
                "fix_details": fix_result,
                "review_details": review_result
            }
            
        except Exception as e:
            logger.error(f"Error in workflow: {e}")
            import traceback
            traceback.print_exc()
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def handle_user_request(
        self,
        user_request: str,
        scope: Optional[str] = None,
        auto_apply: bool = False,
        trigger_review: bool = True
    ) -> Dict[str, Any]:
        """
        Handle natural language change request from user
        
        This is for intentional changes (not errors)
        e.g., "change learning rate to 0.001"
        
        Args:
            user_request: Natural language request
            scope: Optional scope for search
            auto_apply: Apply changes automatically
            trigger_review: Run review cycle after changes
            
        Returns:
            Similar to handle_error()
        """
        logger.info("="*80)
        logger.info("POST-GENERATION CHANGE REQUEST WORKFLOW")
        logger.info("="*80)
        logger.info(f"Request: {user_request}")
        
        try:
            # Step 1: Understand and plan changes
            logger.info("\n[1/4] Understanding request...")
            change_result = await self.contextual_changer.understand_and_plan_changes(
                user_request=user_request,
                scope=scope,
                auto_apply=auto_apply
            )
            
            if change_result["status"] != "success":
                return {
                    "status": "error",
                    "error": "Failed to understand request",
                    "details": change_result
                }
            
            changes_applied = len(change_result.get("applied_changes", []))
            logger.info(f"✓ Applied {changes_applied} changes")
            
            # Step 2: Trigger review cycle
            review_result = None
            if trigger_review:
                logger.info("\n[2/4] Triggering review cycle...")
                review_result = await self._run_review_cycle()
                logger.info(f"✓ Review complete - Quality: {review_result.get('final_score', 0):.1f}%")
            else:
                logger.info("\n[2/4] Skipping review")
            
            # Step 3: Track iteration
            self.current_iteration += 1
            self.workflow_history.append({
                "iteration": self.current_iteration,
                "type": "user_request",
                "request": user_request,
                "changes_applied": changes_applied,
                "review_triggered": review_result is not None,
                "quality_score": review_result.get("final_score") if review_result else None
            })
            
            summary = self._generate_summary()
            
            logger.info("\n[4/4] Workflow complete!")
            logger.info(f"Total changes: {changes_applied}")
            
            return {
                "status": "success",
                "changes_applied": changes_applied,
                "review_cycles": 1 if review_result else 0,
                "final_quality_score": review_result.get("final_score") if review_result else None,
                "workflow_summary": summary,
                "change_details": change_result,
                "review_details": review_result
            }
            
        except Exception as e:
            logger.error(f"Error in workflow: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def _run_review_cycle(self) -> Dict[str, Any]:
        """
        Run complete review cycle with auto-iteration
        
        Returns:
            {
                "final_score": float,
                "iterations": int,
                "improvements": List[str]
            }
        """
        # Get list of Python files to review
        python_files = list(self.project_root.glob("**/*.py"))
        
        # Limit to 10 files for speed, skip __pycache__ and test files
        selected_files = [
            f for f in python_files 
            if "__pycache__" not in str(f) and "test_" not in f.name
        ][:10]
        
        all_reviews = []
        total_score = 0
        
        # Review each file
        for file_path in selected_files:
            try:
                # Read file content
                with open(file_path, 'r', encoding='utf-8') as f:
                    code = f.read()
                
                # Call _review_all() async method directly
                review_result = await self.review_orchestrator._review_all(
                    code=code,
                    parallel=True,
                    project_context={
                        "file_path": str(file_path.relative_to(self.project_root)),
                        "project_root": str(self.project_root)
                    }
                )
                
                all_reviews.append({
                    "file": str(file_path.relative_to(self.project_root)),
                    "score": review_result.get("overall_score", 0),
                    "issues": review_result.get("issues", [])
                })
                
                total_score += review_result.get("overall_score", 0)
                
            except Exception as e:
                logger.error(f"Error reviewing {file_path}: {e}")
                continue
        
        # Calculate average score
        avg_score = total_score / len(selected_files) if selected_files else 0
        
        return {
            "final_score": avg_score,
            "iterations": 1,  # Single iteration for now
            "improvements": [f"Reviewed {len(selected_files)} files"],
            "all_reviews": all_reviews
        }
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate workflow summary"""
        total_fixes = sum(h.get('fixes_applied', 0) for h in self.workflow_history)
        total_changes = sum(h.get('changes_applied', 0) for h in self.workflow_history)
        review_cycles = sum(1 for h in self.workflow_history if h.get('review_triggered'))
        
        quality_scores = [
            h['quality_score'] 
            for h in self.workflow_history 
            if h.get('quality_score') is not None
        ]
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else None
        
        return {
            "total_iterations": self.current_iteration,
            "total_fixes": total_fixes,
            "total_changes": total_changes,
            "review_cycles": review_cycles,
            "average_quality": avg_quality,
            "history": self.workflow_history
        }
    
    async def interactive_loop(self):
        """
        Interactive loop for continuous development
        
        User can:
        1. Report errors
        2. Request changes
        3. Exit
        """
        logger.info("="*80)
        logger.info("INTERACTIVE POST-GENERATION DEVELOPMENT")
        logger.info("="*80)
        logger.info("\nCommands:")
        logger.info("  error <traceback>  - Fix a runtime error")
        logger.info("  change <request>   - Make a code change")
        logger.info("  status             - Show workflow status")
        logger.info("  exit               - Exit interactive mode")
        logger.info("="*80)
        
        while True:
            try:
                command = input("\n> ").strip()
                
                if command == "exit":
                    logger.info("Exiting interactive mode")
                    break
                
                elif command == "status":
                    summary = self._generate_summary()
                    print(f"\nIterations: {summary['total_iterations']}")
                    print(f"Fixes applied: {summary['total_fixes']}")
                    print(f"Changes applied: {summary['total_changes']}")
                    print(f"Review cycles: {summary['review_cycles']}")
                    if summary['average_quality']:
                        print(f"Average quality: {summary['average_quality']:.1f}%")
                
                elif command.startswith("error "):
                    error_msg = command[6:]
                    result = await self.handle_error(
                        error_message=error_msg,
                        auto_apply=True,
                        trigger_review=True
                    )
                    print(f"\nStatus: {result['status']}")
                    print(f"Fixes: {result['fixes_applied']}")
                    if result.get('final_quality_score'):
                        print(f"Quality: {result['final_quality_score']:.1f}%")
                
                elif command.startswith("change "):
                    request = command[7:]
                    result = await self.handle_user_request(
                        user_request=request,
                        auto_apply=True,
                        trigger_review=True
                    )
                    print(f"\nStatus: {result['status']}")
                    print(f"Changes: {result['changes_applied']}")
                    if result.get('final_quality_score'):
                        print(f"Quality: {result['final_quality_score']:.1f}%")
                
                else:
                    print("Unknown command. Use: error, change, status, or exit")
            
            except KeyboardInterrupt:
                logger.info("\nExiting interactive mode")
                break
            except Exception as e:
                logger.error(f"Error: {e}")


# Example usage
async def main():
    """Example workflow"""
    
    # Initialize workflow
    workflow = PostGenerationWorkflow(
        project_root="./output/cifar10_project",
        max_fix_iterations=3,
        max_review_iterations=5
    )
    
    # Example 1: Handle error
    error_traceback = """
Traceback (most recent call last):
  File "train.py", line 5, in <module>
    import torch
ModuleNotFoundError: No module named 'torch'
"""
    
    result = await workflow.handle_error(
        error_message=error_traceback,
        error_context={"command": "python train.py"},
        auto_apply=True,
        trigger_review=True
    )
    
    print(f"Error handling result: {result['status']}")
    print(f"Suggested command: {result['suggested_next_command']}")
    
    # Example 2: Handle user change request
    result2 = await workflow.handle_user_request(
        user_request="change learning rate to 0.001",
        auto_apply=True,
        trigger_review=True
    )
    
    print(f"Change request result: {result2['status']}")
    
    # Example 3: Interactive loop
    # await workflow.interactive_loop()


if __name__ == "__main__":
    asyncio.run(main())
