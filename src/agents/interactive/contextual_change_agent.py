"""
Contextual Change Agent

An intelligent agent that understands natural language requests, analyzes code context,
and identifies all related changes needed when making a modification.

Key Features:
- Natural language understanding (e.g., "change the optimizer to Adam")
- Code search and context analysis
- Dependency tracking (what else needs to change)
- Impact analysis (ripple effects across files)
- Intelligent suggestions for related changes
- Multi-file coordination

Example:
    User: "Change the learning rate to 0.001"
    Agent:
    1. Finds all files with learning rate references
    2. Identifies: config files, training scripts, documentation
    3. Suggests: Update config.py, train.py, README.md
    4. Checks: Are there hyperparameter validation functions?
    5. Proposes: Coordinated changes across all affected files

Authors: GaneshS 271123
gmail:ganeshnaik214@gmail.com
"""

import ast
import os
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Set, Tuple
from loguru import logger

from ..base_agent import BaseAgent
from .modification_agent import InteractiveModificationAgent
from src.llm.gemini_client import GeminiClient
from src.llm.prompt_manager import PromptManager

class ContextualChangeAgent(BaseAgent):
    """
    Intelligent agent that understands natural language and makes contextual code changes
    
    This agent goes beyond simple modifications by:
    1. Understanding user intent from natural language
    2. Finding all related code locations
    3. Identifying dependencies and ripple effects
    4. Proposing coordinated multi-file changes
    """
    
    def __init__(
        self,
        llm_client: "GeminiClient",
        prompt_manager: "PromptManager",
        project_root: str,
        modification_agent: "InteractiveModificationAgent"
    ):
        """
        Initialize ContextualChangeAgent
        
        Args:
            llm_client: LLM client for understanding natural language
            prompt_manager: Prompt template manager
            project_root: Root directory of the project
            modification_agent: Agent for applying changes
        """
        super().__init__(
            agent_type="contextual_change",
            name="contextual_change_agent",
            role="Expert Contextual Code Change Specialist with Natural Language Understanding",
            llm_client=llm_client,
            prompt_manager=prompt_manager
        )
        
        self.project_root = Path(project_root)
        self.modification_agent = modification_agent # Use the injected agent
        
        # Cache for code analysis
        self.code_index: Dict[str, Any] = {}
        self.dependency_graph: Dict[str, Set[str]] = {}
        
        logger.info(f"{self.name} initialized")
        logger.info(f"   - Project root: {self.project_root}")
    
    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a contextual change task
        
        Args:
            task: Task with:
                - user_request: Natural language request (e.g., "change optimizer to Adam")
                - scope: Optional file/directory scope
                - auto_apply: Whether to apply changes automatically
        
        Returns:
            Result with analysis and proposed changes
        """
        user_request = task.get("user_request")
        scope = task.get("scope")
        auto_apply = task.get("auto_apply", False)
        
        return await self.understand_and_plan_changes(
            user_request=user_request,
            scope=scope,
            auto_apply=auto_apply
        )
    
    async def understand_and_plan_changes(
        self,
        user_request: str,
        scope: Optional[str] = None,
        auto_apply: bool = False
    ) -> Dict[str, Any]:
        """
        Understand natural language request and plan all necessary changes
        
        Args:
            user_request: Natural language description (e.g., "change learning rate to 0.001")
            scope: Optional file/directory to limit search
            auto_apply: Whether to automatically apply changes
        
        Returns:
            {
                "status": "success" | "error",
                "understood_intent": Dict with parsed intent,
                "affected_locations": List of code locations,
                "proposed_changes": List of specific changes,
                "impact_analysis": Dict with ripple effects,
                "applied_changes": Optional[List] if auto_apply=True
            }
        """
        try:
            logger.info(f"Understanding request: '{user_request}'")
            
            # Step 1: Parse natural language intent using LLM
            intent = await self._parse_user_intent(user_request)
            logger.info(f"Parsed intent: {intent.get('action')} on {intent.get('target')}")
            
            # Step 2: Search codebase for relevant locations
            affected_locations = await self._find_affected_locations(intent, scope)
            logger.info(f"Found {len(affected_locations)} affected locations")
            
            # Step 3: Analyze dependencies and ripple effects
            impact_analysis = await self._analyze_impact(affected_locations, intent)
            logger.info(f"Impact: {len(impact_analysis.get('related_changes', []))} related changes")
            
            # Step 4: Generate specific change proposals
            proposed_changes = await self._generate_change_proposals(
                intent,
                affected_locations,
                impact_analysis
            )
            logger.info(f"Proposed {len(proposed_changes)} changes")
            
            # Step 5: Apply changes if requested
            applied_changes = []
            if auto_apply and proposed_changes:
                logger.info("Auto-applying changes...")
                applied_changes = await self._apply_changes(proposed_changes)
            
            result = {
                "status": "success",
                "user_request": user_request,
                "understood_intent": intent,
                "affected_locations": affected_locations,
                "proposed_changes": proposed_changes,
                "impact_analysis": impact_analysis,
                "applied_changes": applied_changes if auto_apply else None,
                "total_files_affected": len(affected_locations),
                "total_changes_proposed": len(proposed_changes)
            }
            
            logger.info(f"✅ Contextual analysis complete")
            return result
            
        except Exception as e:
            logger.error(f"Error in contextual change: {e}")
            return {
                "status": "error",
                "message": str(e),
                "user_request": user_request
            }
    
    async def _parse_user_intent(self, user_request: str) -> Dict[str, Any]:
        """
        Parse natural language request using LLM
        
        Args:
            user_request: Natural language request
        
        Returns:
            {
                "action": "change" | "add" | "remove" | "refactor",
                "target": "learning_rate" | "optimizer" | "batch_size" | etc.,
                "value": Optional new value,
                "constraints": Optional constraints,
                "scope": Optional scope hints
            }
        """
        prompt = f"""Analyze this code change request and extract the intent:

REQUEST: "{user_request}"

Identify:
1. Action: change, add, remove, refactor, update
2. Target: What code element (variable, function, class, parameter, etc.)
3. Value: New value if specified
4. Constraints: Any conditions mentioned
5. Scope: Files/modules mentioned

Return as JSON:
{{
    "action": "...",
    "target": "...",
    "value": "...",
    "constraints": [...],
    "scope": "...",
    "keywords": [...]  // Keywords to search for in code
}}

Example:
Request: "change the learning rate to 0.001"
Response: {{
    "action": "change",
    "target": "learning_rate",
    "value": "0.001",
    "keywords": ["learning_rate", "lr", "learning-rate", "learningRate"]
}}

Request: "use Adam optimizer instead of SGD"
Response: {{
    "action": "change",
    "target": "optimizer",
    "value": "Adam",
    "keywords": ["optimizer", "optim", "SGD", "Adam"]
}}

Now analyze: "{user_request}"
"""
        
        response = await self.generate_response(prompt, temperature=0.2)
        
        try:
            # Try to parse JSON from response
            import json
            # Extract JSON from response (handle markdown code blocks)
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', response, re.DOTALL)
            if json_match:
                intent = json.loads(json_match.group(1))
            else:
                # Try direct parsing
                intent = json.loads(response)
            
            logger.debug(f"Parsed intent: {intent}")
            return intent
            
        except Exception as e:
            logger.warning(f"Failed to parse LLM intent response: {e}")
            # Fallback: basic keyword extraction
            return self._fallback_intent_parsing(user_request)
    
    def _fallback_intent_parsing(self, user_request: str) -> Dict[str, Any]:
        """Fallback intent parsing using keywords"""
        user_request_lower = user_request.lower()
        
        # Detect action
        action = "change"
        if any(word in user_request_lower for word in ["add", "insert", "create"]):
            action = "add"
        elif any(word in user_request_lower for word in ["remove", "delete"]):
            action = "remove"
        elif any(word in user_request_lower for word in ["refactor", "rename"]):
            action = "refactor"
        
        # Extract common ML terms
        ml_terms = {
            "learning rate": ["learning_rate", "lr", "learning-rate"],
            "optimizer": ["optimizer", "optim", "Adam", "SGD"],
            "batch size": ["batch_size", "batch-size"],
            "epochs": ["epochs", "num_epochs"],
            "dropout": ["dropout", "drop_rate"],
        }
        
        keywords = []
        target = "unknown"
        for term, variants in ml_terms.items():
            if term in user_request_lower:
                target = term.replace(" ", "_")
                keywords.extend(variants)
                break
        
        # Extract value (numbers, variable names)
        value_match = re.search(r'to\s+([0-9.]+|[A-Za-z_]+)', user_request)
        value = value_match.group(1) if value_match else None
        
        return {
            "action": action,
            "target": target,
            "value": value,
            "keywords": keywords or [target],
            "confidence": "low"  # Fallback has low confidence
        }
    
    async def _find_affected_locations(
        self,
        intent: Dict[str, Any],
        scope: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Find all code locations affected by the intended change
        
        Args:
            intent: Parsed intent
            scope: Optional scope restriction
        
        Returns:
            List of locations: [
                {
                    "filepath": str,
                    "line_number": int,
                    "context": str,
                    "type": "variable" | "function" | "class" | "config",
                    "confidence": float
                }
            ]
        """
        keywords = intent.get("keywords", [])
        target = intent.get("target", "")
        
        affected_locations = []
        
        # Search Python files
        python_files = self._find_python_files(scope)
        
        for filepath in python_files:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                    lines = content.split('\n')
                
                # Search for keywords in file
                for i, line in enumerate(lines, 1):
                    for keyword in keywords:
                        if keyword.lower() in line.lower():
                            # Found a match
                            location = {
                                "filepath": str(filepath.relative_to(self.project_root)),
                                "line_number": i,
                                "context": line.strip(),
                                "type": self._classify_location_type(line),
                                "confidence": self._calculate_relevance(line, intent),
                                "keyword_matched": keyword
                            }
                            affected_locations.append(location)
                            
            except Exception as e:
                logger.debug(f"Error reading {filepath}: {e}")
                continue
        
        # Sort by confidence (most relevant first)
        affected_locations.sort(key=lambda x: x["confidence"], reverse=True)
        
        return affected_locations
    
    def _find_python_files(self, scope: Optional[str] = None) -> List[Path]:
        """Find all Python files in project (or scope)"""
        if scope:
            search_path = self.project_root / scope
        else:
            search_path = self.project_root
        
        if not search_path.exists():
            search_path = self.project_root
        
        python_files = []
        for root, dirs, files in os.walk(search_path):
            # Skip common non-code directories
            dirs[:] = [d for d in dirs if d not in [
                '__pycache__', '.git', 'venv', 'env', 'node_modules',
                '.vscode', '.idea', 'dist', 'build'
            ]]
            
            for file in files:
                if file.endswith('.py'):
                    python_files.append(Path(root) / file)
        
        return python_files
    
    def _classify_location_type(self, line: str) -> str:
        """Classify what type of code location this is"""
        line_stripped = line.strip()
        
        if line_stripped.startswith('def '):
            return "function"
        elif line_stripped.startswith('class '):
            return "class"
        elif '=' in line_stripped:
            return "variable_assignment"
        elif line_stripped.startswith('#'):
            return "comment"
        else:
            return "other"
    
    def _calculate_relevance(self, line: str, intent: Dict[str, Any]) -> float:
        """Calculate how relevant this location is (0-1)"""
        confidence = 0.5  # Base confidence
        
        target = intent.get("target", "").lower()
        line_lower = line.lower()
        
        # Boost if target exactly matches
        if f"{target} =" in line_lower or f"{target}=" in line_lower:
            confidence += 0.3
        
        # Boost if it's a variable assignment
        if '=' in line:
            confidence += 0.1
        
        # Reduce if it's a comment
        if line.strip().startswith('#'):
            confidence -= 0.2
        
        # Boost if in config file
        if 'config' in line_lower or 'hyperparameter' in line_lower:
            confidence += 0.2
        
        return min(max(confidence, 0.0), 1.0)
    
    async def _analyze_impact(
        self,
        affected_locations: List[Dict[str, Any]],
        intent: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze ripple effects and dependencies
        
        Returns:
            {
                "related_changes": List of additional changes needed,
                "dependency_files": Files that import affected files,
                "documentation_updates": Docs that need updating,
                "test_files": Tests that might need updating
            }
        """
        impact = {
            "related_changes": [],
            "dependency_files": [],
            "documentation_updates": [],
            "test_files": []
        }
        
        # Get unique files affected
        affected_files = set(loc["filepath"] for loc in affected_locations)
        
        # Find dependency files (files that import these)
        for filepath in affected_files:
            module_name = filepath.replace('.py', '').replace('/', '.')
            
            # Search for imports of this module
            all_files = self._find_python_files()
            for file in all_files:
                try:
                    with open(file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if f"import {module_name}" in content or f"from {module_name}" in content:
                            impact["dependency_files"].append(str(file.relative_to(self.project_root)))
                except:
                    pass
        
        # Find documentation files
        doc_files = [
            'README.md', 'CHANGELOG.md', 'docs/README.md',
            'docs/configuration.md', 'docs/parameters.md'
        ]
        for doc in doc_files:
            doc_path = self.project_root / doc
            if doc_path.exists():
                impact["documentation_updates"].append(doc)
        
        # Find test files
        for filepath in affected_files:
            test_file = filepath.replace('.py', '_test.py')
            test_path = self.project_root / test_file
            if test_path.exists():
                impact["test_files"].append(test_file)
            
            # Also check tests/ directory
            test_file2 = Path('tests') / Path(filepath).name
            test_path2 = self.project_root / test_file2
            if test_path2.exists():
                impact["test_files"].append(str(test_file2))
        
        return impact
    
    async def _generate_change_proposals(
        self,
        intent: Dict[str, Any],
        affected_locations: List[Dict[str, Any]],
        impact_analysis: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Generate specific, actionable change proposals
        
        Returns:
            List of changes: [
                {
                    "filepath": str,
                    "change_type": "modify" | "add" | "remove",
                    "description": str,
                    "old_code": Optional[str],
                    "new_code": Optional[str],
                    "priority": "high" | "medium" | "low",
                    "reason": str
                }
            ]
        """
        proposals = []
        
        action = intent.get("action")
        target = intent.get("target")
        new_value = intent.get("value")
        
        # Generate proposals for each affected location
        for location in affected_locations[:10]:  # Limit to top 10 most relevant
            if location["confidence"] < 0.3:
                continue  # Skip low-confidence matches
            
            filepath = location["filepath"]
            line_num = location["line_number"]
            context = location["context"]
            
            # Read the file to get more context
            full_path = self.project_root / filepath
            try:
                with open(full_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                if line_num > 0 and line_num <= len(lines):
                    old_code = lines[line_num - 1].rstrip()
                    
                    # Generate new code based on intent
                    if action == "change" and new_value:
                        # Replace the value
                        new_code = self._generate_replacement(old_code, target, new_value)
                        
                        if new_code != old_code:
                            proposal = {
                                "filepath": filepath,
                                "change_type": "modify",
                                "description": f"Change {target} to {new_value}",
                                "old_code": old_code,
                                "new_code": new_code,
                                "priority": "high" if location["confidence"] > 0.7 else "medium",
                                "reason": f"Matches user request at line {line_num}",
                                "line_number": line_num
                            }
                            proposals.append(proposal)
                            
            except Exception as e:
                logger.debug(f"Error generating proposal for {filepath}: {e}")
                continue
        
        return proposals
    
    def _generate_replacement(self, old_code: str, target: str, new_value: str) -> str:
        """Generate replacement code"""
        # Simple value replacement
        # Pattern: target = old_value  ->  target = new_value
        
        # Try different patterns
        patterns = [
            (rf'({target}\s*=\s*)([^,\n\r]+)', rf'\g<1>{new_value}'),
            (r'("' + target + r'"\s*:\s*)([^,}\n\r]+)', rf'\g<1>{new_value}'),
            (r'(\["' + target + r'"\]\s*=\s*)([^,\n\r]+)', rf'\g<1>{new_value}'),
        ]
        
        for pattern, replacement in patterns:
            new_code = re.sub(pattern, replacement, old_code)
            if new_code != old_code:
                return new_code
        
        return old_code
    
    async def _apply_changes(
        self,
        proposed_changes: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Apply proposed changes using ModificationAgent"""
        applied = []
        
        for change in proposed_changes:
            if change["change_type"] == "modify":
                result = await self.modification_agent.modify_file(
                    filepath=change["filepath"],
                    changes={
                        "old_code": change["old_code"],
                        "new_code": change["new_code"],
                        "validate": True
                    }
                )
                
                applied.append({
                    "change": change,
                    "result": result
                })
        
        return applied
    
    async def preview_changes(
        self,
        user_request: str,
        scope: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Preview what changes would be made without applying them
        
        Args:
            user_request: Natural language request
            scope: Optional scope
        
        Returns:
            Full analysis with proposals (but no changes applied)
        """
        return await self.understand_and_plan_changes(
            user_request=user_request,
            scope=scope,
            auto_apply=False
        )
