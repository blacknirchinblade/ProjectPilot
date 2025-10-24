"""
Feedback Handler - Process User Corrections and Learn Preferences

This handler:
- Parses user feedback into structured data
- Applies corrections to code
- Learns from user preferences
- Tracks feedback patterns
- Suggests improvements based on history

Authors: GaneshS 271123
gmail:ganeshnaik214@gmail.com
"""

from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from loguru import logger
import json
import re

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage


@dataclass
class FeedbackData:
    """Structured feedback data."""
    feedback_id: str
    file_path: Path
    original_code: str
    feedback_text: str
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Parsed components
    corrections: List[str] = field(default_factory=list)
    preferences: Dict[str, str] = field(default_factory=dict)
    sentiment: str = "neutral"  # positive, neutral, negative
    rating: Optional[int] = None


@dataclass
class UserPreference:
    """User preference learned from feedback."""
    preference_type: str  # style, pattern, library, etc.
    preference_value: str
    confidence: float  # 0.0 to 1.0
    occurrences: int = 1
    last_seen: datetime = field(default_factory=datetime.now)


class FeedbackHandler:
    """
    Processes user feedback and learns preferences.
    """
    
    def __init__(
        self,
        model_name: str = "gemini-2.0-flash-exp",
        preferences_path: Optional[Path] = None
    ):
        """
        Initialize the feedback handler.
        
        Args:
            model_name: LLM model to use
            preferences_path: Path to save learned preferences
        """
        self.llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.3)
        self.preferences_path = preferences_path or Path("data/preferences")
        self.preferences_path.mkdir(parents=True, exist_ok=True)
        
        self.user_preferences: Dict[str, UserPreference] = {}
        self._load_preferences()
        
        logger.info("Initialized FeedbackHandler")
    
    def parse_feedback(
        self,
        feedback_text: str,
        file_path: Path,
        original_code: str
    ) -> FeedbackData:
        """
        Parse free-form feedback into structured data.
        
        Args:
            feedback_text: User's feedback text
            file_path: File the feedback is about
            original_code: Original code
        
        Returns:
            FeedbackData object with parsed components
        """
        logger.info(f"Parsing feedback for {file_path.name}")
        
        try:
            # Use LLM to extract structured information
            prompt = f"""
Parse this user feedback about Python code into structured components.

FEEDBACK:
{feedback_text}

Extract:
1. Specific corrections (what needs to be changed)
2. User preferences (coding style, patterns, libraries)
3. Sentiment (positive/neutral/negative)
4. Rating (1-5 if mentioned)

Return JSON format:
{{
    "corrections": ["correction 1", "correction 2"],
    "preferences": {{"style": "value", "pattern": "value"}},
    "sentiment": "positive|neutral|negative",
    "rating": 1-5 or null
}}
"""
            
            messages = [
                SystemMessage(content="You are an expert at parsing user feedback."),
                HumanMessage(content=prompt)
            ]
            
            response = self.llm.invoke(messages)
            
            # Parse JSON response
            parsed = self._extract_json(response.content)
            
            # Create FeedbackData
            feedback_id = f"fb_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            feedback_data = FeedbackData(
                feedback_id=feedback_id,
                file_path=file_path,
                original_code=original_code,
                feedback_text=feedback_text,
                corrections=parsed.get("corrections", []),
                preferences=parsed.get("preferences", {}),
                sentiment=parsed.get("sentiment", "neutral"),
                rating=parsed.get("rating")
            )
            
            # Learn from preferences
            self._learn_preferences(feedback_data.preferences)
            
            logger.success(f"✓ Parsed feedback: {len(feedback_data.corrections)} corrections")
            return feedback_data
            
        except Exception as e:
            logger.error(f"Failed to parse feedback: {e}")
            
            # Return basic feedback data
            return FeedbackData(
                feedback_id=f"fb_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                file_path=file_path,
                original_code=original_code,
                feedback_text=feedback_text
            )
    
    def apply_corrections(
        self,
        code: str,
        corrections: List[str],
        file_path: Optional[Path] = None
    ) -> Tuple[bool, str]:
        """
        Apply user corrections to code.
        
        Args:
            code: Original code
            corrections: List of corrections to apply
            file_path: Optional file path for context
        
        Returns:
            Tuple of (success: bool, corrected_code: str)
        """
        logger.info(f"Applying {len(corrections)} corrections")
        
        try:
            # Build correction prompt
            corrections_text = "\n".join([f"{i+1}. {c}" for i, c in enumerate(corrections)])
            
            prompt = f"""
Apply these user corrections to the Python code.

ORIGINAL CODE:
```python
{code}
```

CORRECTIONS TO APPLY:
{corrections_text}

Apply each correction carefully:
- Make the exact changes requested
- Preserve working functionality
- Maintain code style
- Fix any syntax issues

Return ONLY the corrected code, nothing else.
"""
            
            messages = [
                SystemMessage(content="You are an expert Python developer."),
                HumanMessage(content=prompt)
            ]
            
            response = self.llm.invoke(messages)
            corrected_code = self._extract_code(response.content)
            
            logger.success("✓ Corrections applied successfully")
            return True, corrected_code
            
        except Exception as e:
            logger.error(f"Failed to apply corrections: {e}")
            return False, code
    
    def learn_preferences(self, preferences: Dict[str, str]):
        """
        Learn user preferences from feedback.
        
        Args:
            preferences: Dictionary of preferences
        """
        self._learn_preferences(preferences)
    
    def get_preferences(self, preference_type: Optional[str] = None) -> Dict[str, UserPreference]:
        """
        Get learned preferences.
        
        Args:
            preference_type: Filter by type (None = all)
        
        Returns:
            Dictionary of preferences
        """
        if preference_type:
            return {
                k: v for k, v in self.user_preferences.items()
                if v.preference_type == preference_type
            }
        return self.user_preferences.copy()
    
    def suggest_improvements(
        self,
        code: str,
        file_path: Path
    ) -> List[str]:
        """
        Suggest improvements based on learned preferences.
        
        Args:
            code: Code to analyze
            file_path: File path
        
        Returns:
            List of improvement suggestions
        """
        logger.info(f"Generating preference-based suggestions for {file_path.name}")
        
        try:
            # Build preferences context
            pref_text = "\n".join([
                f"- {pref.preference_type}: {pref.preference_value} (confidence: {pref.confidence:.1%})"
                for pref in self.user_preferences.values()
                if pref.confidence > 0.5
            ])
            
            if not pref_text:
                logger.info("No strong preferences learned yet")
                return []
            
            prompt = f"""
Analyze this Python code and suggest improvements based on user preferences.

CODE:
```python
{code}
```

USER PREFERENCES:
{pref_text}

Suggest 2-3 specific improvements that align with these preferences.
Be concrete and actionable.
"""
            
            messages = [
                SystemMessage(content="You are a code quality expert."),
                HumanMessage(content=prompt)
            ]
            
            response = self.llm.invoke(messages)
            
            # Extract suggestions
            suggestions = []
            for line in response.content.split('\n'):
                line = line.strip()
                if line.startswith('-') or line.startswith('*') or line.startswith('•'):
                    suggestions.append(line.lstrip('-*• '))
            
            logger.info(f"Generated {len(suggestions)} preference-based suggestions")
            return suggestions[:3]
            
        except Exception as e:
            logger.error(f"Failed to generate suggestions: {e}")
            return []
    
    def get_feedback_summary(self) -> Dict[str, any]:
        """
        Get summary of collected feedback.
        
        Returns:
            Summary dictionary
        """
        summary = {
            "total_preferences": len(self.user_preferences),
            "high_confidence": sum(
                1 for p in self.user_preferences.values()
                if p.confidence > 0.7
            ),
            "preference_types": {},
            "most_common": []
        }
        
        # Count by type
        for pref in self.user_preferences.values():
            summary["preference_types"][pref.preference_type] = \
                summary["preference_types"].get(pref.preference_type, 0) + 1
        
        # Get most common preferences
        sorted_prefs = sorted(
            self.user_preferences.values(),
            key=lambda p: (p.confidence, p.occurrences),
            reverse=True
        )
        summary["most_common"] = [
            {
                "type": p.preference_type,
                "value": p.preference_value,
                "confidence": p.confidence
            }
            for p in sorted_prefs[:5]
        ]
        
        return summary
    
    def reset_preferences(self):
        """Reset all learned preferences."""
        self.user_preferences.clear()
        self._save_preferences()
        logger.info("Reset all user preferences")
    
    # Helper methods
    
    def _learn_preferences(self, preferences: Dict[str, str]):
        """Learn from new preferences."""
        for pref_type, pref_value in preferences.items():
            key = f"{pref_type}:{pref_value}"
            
            if key in self.user_preferences:
                # Increase occurrences and confidence
                pref = self.user_preferences[key]
                pref.occurrences += 1
                pref.confidence = min(1.0, pref.confidence + 0.1)
                pref.last_seen = datetime.now()
            else:
                # New preference
                self.user_preferences[key] = UserPreference(
                    preference_type=pref_type,
                    preference_value=pref_value,
                    confidence=0.6  # Initial confidence
                )
            
            logger.debug(f"Learned preference: {pref_type} = {pref_value}")
        
        # Save preferences
        self._save_preferences()
    
    def _extract_json(self, response: str) -> Dict:
        """Extract JSON from LLM response."""
        try:
            # Try to find JSON block
            json_pattern = r'```json\n(.*?)\n```'
            matches = re.findall(json_pattern, response, re.DOTALL)
            
            if matches:
                return json.loads(matches[0])
            
            # Try to parse entire response
            return json.loads(response)
        except:
            logger.warning("Failed to parse JSON from response")
            return {}
    
    def _extract_code(self, response: str) -> str:
        """Extract code from LLM response."""
        # Try to find code blocks
        code_pattern = r'```python\n(.*?)\n```'
        matches = re.findall(code_pattern, response, re.DOTALL)
        
        if matches:
            return matches[0]
        
        # Try without language specifier
        code_pattern = r'```\n(.*?)\n```'
        matches = re.findall(code_pattern, response, re.DOTALL)
        
        if matches:
            return matches[0]
        
        # Return as-is
        return response.strip()
    
    def _load_preferences(self):
        """Load saved preferences from disk."""
        pref_file = self.preferences_path / "user_preferences.json"
        
        if not pref_file.exists():
            return
        
        try:
            with open(pref_file) as f:
                data = json.load(f)
            
            for key, pref_data in data.items():
                self.user_preferences[key] = UserPreference(
                    preference_type=pref_data["preference_type"],
                    preference_value=pref_data["preference_value"],
                    confidence=pref_data["confidence"],
                    occurrences=pref_data["occurrences"],
                    last_seen=datetime.fromisoformat(pref_data["last_seen"])
                )
            
            logger.info(f"Loaded {len(self.user_preferences)} preferences")
            
        except Exception as e:
            logger.error(f"Failed to load preferences: {e}")
    
    def _save_preferences(self):
        """Save preferences to disk."""
        pref_file = self.preferences_path / "user_preferences.json"
        
        try:
            data = {
                key: {
                    "preference_type": pref.preference_type,
                    "preference_value": pref.preference_value,
                    "confidence": pref.confidence,
                    "occurrences": pref.occurrences,
                    "last_seen": pref.last_seen.isoformat()
                }
                for key, pref in self.user_preferences.items()
            }
            
            with open(pref_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.debug("Saved preferences")
            
        except Exception as e:
            logger.error(f"Failed to save preferences: {e}")


# CODE_GENERATION_COMPLETE
