"""
Data Structures for Clarification Agent

Defines the core data structures used for clarification workflow.
Authors: GaneshS 271123
gmail:ganeshnaik214@gmail.com
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)



class QuestionCategory(Enum):
    """Question categories for clarification"""
    FRAMEWORK = "framework"
    HARDWARE = "hardware"
    DATA = "data"
    ARCHITECTURE = "architecture"
    DEPLOYMENT = "deployment"
    TESTING = "testing"
    DOCUMENTATION = "documentation"
    OTHER = "other"


class Priority(Enum):
    """Question priority levels"""
    BLOCKING = 1  # Must answer to proceed
    HIGH = 2      # Should answer for quality
    MEDIUM = 3    # Nice to have
    LOW = 4       # Optional refinement


@dataclass
class Question:
    """Represents a clarification question"""
    
    id: str
    category: QuestionCategory
    text: str
    priority: Priority
    context_key: str
    options: Optional[List[str]] = None
    default: Optional[str] = None
    validation_rule: Optional[str] = None
    required: bool = False
    follow_up_questions: Dict[str, List[str]] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate question structure"""
        if not self.id or not self.text:
            raise ValueError("Question must have id and text")
        
        # Convert string category/priority to enum if needed
        if isinstance(self.category, str):
            self.category = QuestionCategory(self.category)
        if isinstance(self.priority, str):
            try:
                self.priority = Priority[self.priority.upper()]
            except KeyError:
                # Try converting from int
                self.priority = Priority(int(self.priority))
    
    def has_options(self) -> bool:
        """Check if question has predefined options"""
        return self.options is not None and len(self.options) > 0
    
    def format_for_display(self) -> str:
        """Format question for user display"""
        output = [f"\n{self.text}"]
        
        if self.has_options():
            for i, option in enumerate(self.options, 1):
                marker = " (default)" if option == self.default else ""
                output.append(f"  {i}. {option}{marker}")
        
        if self.default and not self.has_options():
            output.append(f"  (Press Enter for default: {self.default})")
        
        return "\n".join(output)


@dataclass
class Answer:
    """Represents a user answer to a question"""
    
    question_id: str
    raw_text: str
    parsed_value: Any
    confidence: float
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_valid(self) -> bool:
        """Check if answer is valid"""
        return (
            self.confidence >= 0.7 
            and self.parsed_value is not None 
            and self.raw_text.strip() != ""
        )
    
    def __repr__(self) -> str:
        return (
            f"Answer(question_id='{self.question_id}', "
            f"value={self.parsed_value}, confidence={self.confidence:.2f})"
        )


@dataclass
class ProjectContext:
    """Enhanced project context with clarified requirements"""
    
    # Original user input
    original_request: str
    
    # Framework & Hardware
    framework: str = "pytorch"
    hardware: str = "cpu"
    gpu_support: bool = False
    batch_size: Optional[int] = None
    
    # Data handling
    data_augmentation: bool = True
    data_validation: bool = False
    preprocessing_steps: List[str] = field(default_factory=list)
    
    # Model architecture
    architecture_type: str = "custom"
    pretrained_model: Optional[str] = None
    num_classes: Optional[int] = None
    
    # Deployment
    deployment_target: str = "local"
    containerize: bool = False
    api_required: bool = False
    
    # Testing
    test_coverage_target: float = 0.8
    include_integration_tests: bool = True
    include_e2e_tests: bool = False
    
    # Documentation
    documentation_level: str = "basic"
    include_examples: bool = True
    
    # Metadata
    clarification_complete: bool = False
    confidence_score: float = 0.0
    unanswered_questions: List[str] = field(default_factory=list)
    clarification_timestamp: datetime = field(default_factory=datetime.now)
    
    def validate(self) -> Tuple[bool, List[str]]:
        """
        Validate context completeness
        
        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = []
        
        # Check required fields
        if not self.framework:
            errors.append("Framework not specified")
        
        if self.architecture_type == "pretrained" and not self.pretrained_model:
            errors.append("Pretrained model not specified but architecture type is 'pretrained'")
        
        if self.gpu_support and self.hardware == "cpu":
            errors.append("GPU support enabled but hardware set to CPU")
        
        if self.test_coverage_target < 0 or self.test_coverage_target > 1:
            errors.append(f"Invalid test coverage target: {self.test_coverage_target}")
        
        # Log validation results
        if errors:
            logger.warning(f"Context validation found {len(errors)} errors: {errors}")
        else:
            logger.info("Context validation passed")
        
        return len(errors) == 0, errors
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "original_request": self.original_request,
            "framework": self.framework,
            "hardware": self.hardware,
            "gpu_support": self.gpu_support,
            "batch_size": self.batch_size,
            "data_augmentation": self.data_augmentation,
            "data_validation": self.data_validation,
            "preprocessing_steps": self.preprocessing_steps,
            "architecture_type": self.architecture_type,
            "pretrained_model": self.pretrained_model,
            "num_classes": self.num_classes,
            "deployment_target": self.deployment_target,
            "containerize": self.containerize,
            "api_required": self.api_required,
            "test_coverage_target": self.test_coverage_target,
            "include_integration_tests": self.include_integration_tests,
            "include_e2e_tests": self.include_e2e_tests,
            "documentation_level": self.documentation_level,
            "include_examples": self.include_examples,
            "clarification_complete": self.clarification_complete,
            "confidence_score": self.confidence_score,
            "unanswered_questions": self.unanswered_questions,
        }
    
    def get_summary(self) -> str:
        """Get human-readable summary of context"""
        lines = [
            "=== Project Context Summary ===",
            f"Original Request: {self.original_request[:100]}...",
            f"Framework: {self.framework}",
            f"Hardware: {self.hardware} (GPU: {self.gpu_support})",
            f"Architecture: {self.architecture_type}",
            f"Deployment: {self.deployment_target}",
            f"Test Coverage: {self.test_coverage_target*100:.0f}%",
            f"Documentation: {self.documentation_level}",
            f"Completeness: {self.confidence_score*100:.0f}%",
        ]
        
        if self.unanswered_questions:
            lines.append(f"Unanswered: {len(self.unanswered_questions)} questions")
        
        return "\n".join(lines)
