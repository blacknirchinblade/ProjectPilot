"""
Question Template Loader

Loads question templates from YAML files and converts them to Question objects.
Authors: GaneshS 271123
gmail:ganeshnaik214@gmail.com
"""

import yaml
import logging
from pathlib import Path
from typing import Dict, List, Optional

from src.agents.interactive.data_structures import Question, QuestionCategory, Priority

logger = logging.getLogger(__name__)


class QuestionTemplateLoader:
    """Loads and manages question templates"""
    
    def __init__(self, template_file: Optional[str] = None):
        """
        Initialize template loader
        
        Args:
            template_file: Path to YAML template file (default: config/question_templates.yaml)
        """
        if template_file is None:
            # Default path - go up from src/agents/interactive to project root
            template_file = Path(__file__).parent.parent.parent.parent / "config" / "question_templates.yaml"
        
        self.template_file = Path(template_file)
        self.templates: Dict[str, Question] = {}
        self.templates_by_category: Dict[QuestionCategory, List[Question]] = {}
        
        self._load_templates()
    
    def _load_templates(self):
        """Load templates from YAML file"""
        if not self.template_file.exists():
            logger.warning(f"Template file not found: {self.template_file}")
            return
        
        try:
            with open(self.template_file, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            
            if not data:
                logger.warning("Template file is empty")
                return
            
            # Convert YAML to Question objects
            for template_key, template_data in data.items():
                try:
                    question = self._create_question_from_template(template_key, template_data)
                    self.templates[template_key] = question
                    
                    # Group by category
                    category = question.category
                    if category not in self.templates_by_category:
                        self.templates_by_category[category] = []
                    self.templates_by_category[category].append(question)
                    
                except Exception as e:
                    logger.error(f"Failed to load template '{template_key}': {e}")
            
            logger.info(f"Loaded {len(self.templates)} question templates from {self.template_file}")
            
        except Exception as e:
            logger.error(f"Failed to load templates: {e}")
    
    def _create_question_from_template(self, key: str, data: Dict) -> Question:
        """
        Create Question object from template data
        
        Args:
            key: Template key (e.g., 'framework_selection')
            data: Template data dictionary
            
        Returns:
            Question object
        """
        # Parse priority
        priority_str = data.get('priority', 'MEDIUM').upper()
        try:
            priority = Priority[priority_str]
        except KeyError:
            priority = Priority.MEDIUM
            logger.warning(f"Invalid priority '{priority_str}' for template '{key}', using MEDIUM")
        
        # Parse category
        category_str = data.get('category', 'other')
        try:
            category = QuestionCategory(category_str)
        except ValueError:
            category = QuestionCategory.OTHER
            logger.warning(f"Invalid category '{category_str}' for template '{key}', using OTHER")
        
        # Create Question object
        question = Question(
            id=data.get('id', f"Q_{key.upper()}"),
            category=category,
            text=data.get('text', 'No question text provided'),
            priority=priority,
            context_key=data.get('context_key', key),
            options=data.get('options'),
            default=data.get('default'),
            validation_rule=data.get('validation_rule'),
            required=data.get('required', False),
            follow_up_questions=data.get('follow_up_questions', {}),
            metadata={
                'help_text': data.get('help_text'),
                'template_key': key
            }
        )
        
        return question
    
    def get_template(self, key: str) -> Optional[Question]:
        """
        Get template by key
        
        Args:
            key: Template key (e.g., 'framework_selection')
            
        Returns:
            Question object or None if not found
        """
        return self.templates.get(key)
    
    def get_templates_by_category(self, category: QuestionCategory) -> List[Question]:
        """
        Get all templates for a category
        
        Args:
            category: Question category
            
        Returns:
            List of Question objects
        """
        return self.templates_by_category.get(category, [])
    
    def get_essential_templates(self) -> List[Question]:
        """
        Get essential/required templates (BLOCKING priority)
        
        Returns:
            List of essential Question objects
        """
        return [
            q for q in self.templates.values()
            if q.priority == Priority.BLOCKING
        ]
    
    def get_recommended_templates(self, context: Optional[Dict] = None) -> List[Question]:
        """
        Get recommended templates based on context
        
        Args:
            context: Optional context dictionary (e.g., user request, previous answers)
            
        Returns:
            List of recommended Question objects
        """
        # Start with essential questions
        questions = self.get_essential_templates()
        
        # Add high-priority questions
        high_priority = [
            q for q in self.templates.values()
            if q.priority == Priority.HIGH
        ]
        questions.extend(high_priority)
        
        # TODO: Use context to filter/prioritize questions
        # For now, return essential + high priority
        
        return questions
    
    def search_templates(self, keyword: str) -> List[Question]:
        """
        Search templates by keyword in text or metadata
        
        Args:
            keyword: Search keyword
            
        Returns:
            List of matching Question objects
        """
        keyword_lower = keyword.lower()
        matches = []
        
        for q in self.templates.values():
            if (keyword_lower in q.text.lower() or
                keyword_lower in q.category.value.lower() or
                (q.metadata.get('help_text') and keyword_lower in q.metadata['help_text'].lower())):
                matches.append(q)
        
        return matches
    
    def get_follow_up_template(self, parent_key: str, answer_key: str) -> Optional[Question]:
        """
        Get follow-up template based on parent question and answer
        
        Args:
            parent_key: Parent template key
            answer_key: Answer key that triggers follow-up
            
        Returns:
            Follow-up Question or None
        """
        parent = self.get_template(parent_key)
        if not parent or not parent.follow_up_questions:
            return None
        
        # Get follow-up template keys for this answer
        follow_up_keys = parent.follow_up_questions.get(answer_key, [])
        
        # Return first follow-up template (could be extended to return all)
        if follow_up_keys:
            follow_up_key = follow_up_keys[0] if isinstance(follow_up_keys, list) else follow_up_keys
            return self.get_template(follow_up_key)
        
        return None
    
    def reload(self):
        """Reload templates from file"""
        self.templates.clear()
        self.templates_by_category.clear()
        self._load_templates()
    
    def get_statistics(self) -> Dict[str, int]:
        """
        Get template statistics
        
        Returns:
            Dictionary with counts by category and priority
        """
        stats = {
            'total': len(self.templates),
            'by_category': {},
            'by_priority': {}
        }
        
        for q in self.templates.values():
            # Count by category
            cat_key = q.category.value
            stats['by_category'][cat_key] = stats['by_category'].get(cat_key, 0) + 1
            
            # Count by priority
            pri_key = q.priority.name
            stats['by_priority'][pri_key] = stats['by_priority'].get(pri_key, 0) + 1
        
        return stats


# Singleton instance
_template_loader = None


def get_template_loader(reload: bool = False) -> QuestionTemplateLoader:
    """
    Get singleton template loader instance
    
    Args:
        reload: If True, reload templates from file
        
    Returns:
        QuestionTemplateLoader instance
    """
    global _template_loader
    
    if _template_loader is None or reload:
        _template_loader = QuestionTemplateLoader()
    elif reload:
        _template_loader.reload()
    
    return _template_loader
