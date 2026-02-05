"""
Analysis Service for conversation feedback and scoring.

This service analyzes completed conversations and provides:
- Grammar score (0-100)
- Final overall score (0-100)
- List of grammar mistakes with corrections
- Strengths and suggestions for improvement
"""

import asyncio
import json
import re
from dataclasses import dataclass, field
from typing import List, Optional

from loguru import logger

from app.config import settings
from app.prompts import get_analysis_prompt


@dataclass
class GrammarMistake:
    """A single grammar mistake found in the conversation."""

    original: str
    correction: str
    explanation: str


@dataclass
class ConversationFeedback:
    """Complete feedback for a conversation."""

    grammar_score: int
    final_score: int
    mistakes: List[GrammarMistake] = field(default_factory=list)
    strengths: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert feedback to dictionary for JSON serialization."""
        return {
            "grammar_score": self.grammar_score,
            "final_score": self.final_score,
            "mistakes": [
                {
                    "original": m.original,
                    "correction": m.correction,
                    "explanation": m.explanation,
                }
                for m in self.mistakes
            ],
            "strengths": self.strengths,
            "suggestions": self.suggestions,
        }


class AnalysisService:
    """
    Service for analyzing conversations and providing feedback.

    This service uses the LLM to evaluate:
    - Grammar accuracy
    - Vocabulary usage
    - Overall fluency

    It provides actionable feedback to help learners improve.
    """

    def __init__(self):
        """Initialize the analysis service."""
        self._llm_service = None
        self._lock = asyncio.Lock()

    def _get_llm_service(self):
        """Lazy load LLM service to avoid circular imports."""
        if self._llm_service is None:
            from app.services.llm_service import get_llm_service

            self._llm_service = get_llm_service()
        return self._llm_service

    async def analyze_conversation(
        self, user_utterances: List[str]
    ) -> ConversationFeedback:
        """
        Analyze a completed conversation and provide feedback.

        Args:
            user_utterances: List of all user messages from the conversation

        Returns:
            ConversationFeedback with scores, mistakes, and suggestions
        """
        if not user_utterances:
            return self._empty_feedback()

        # Filter out empty utterances
        utterances = [u.strip() for u in user_utterances if u and u.strip()]

        if not utterances:
            return self._empty_feedback()

        try:
            # Generate analysis using LLM
            analysis_json = await self._generate_analysis(utterances)

            # Parse and validate the response
            feedback = self._parse_analysis(analysis_json)

            return feedback

        except Exception as e:
            logger.error(f"Conversation analysis failed: {e}")
            return self._fallback_feedback(utterances)

    async def _generate_analysis(self, utterances: List[str]) -> str:
        """Generate analysis using the LLM."""
        llm_service = self._get_llm_service()

        if not llm_service.is_loaded:
            await llm_service.load_model()

        # Create analysis prompt
        prompt = get_analysis_prompt(utterances)

        # Use a lower temperature for more consistent analysis
        try:
            from app.services.llm_service import ConversationContext

            # Create a simple context for analysis
            context = ConversationContext()
            context.system_prompt = "You are an English language assessment expert. Respond only with valid JSON."
            context.add_user_message(prompt)

            response = await llm_service.generate(
                context,
                max_tokens=1024,
                temperature=0.3,
                top_p=0.9,
            )

            return response

        except Exception as e:
            logger.error(f"LLM analysis generation failed: {e}")
            raise

    def _parse_analysis(self, analysis_json: str) -> ConversationFeedback:
        """Parse the LLM's JSON response into ConversationFeedback."""
        try:
            # Try to extract JSON from the response
            json_str = self._extract_json(analysis_json)
            data = json.loads(json_str)

            # Extract scores
            grammar_score = self._validate_score(data.get("grammar_score", 70))
            final_score = self._validate_score(data.get("final_score", 70))

            # Extract mistakes
            mistakes = []
            for mistake_data in data.get("mistakes", []):
                if isinstance(mistake_data, dict):
                    mistakes.append(
                        GrammarMistake(
                            original=mistake_data.get("original", ""),
                            correction=mistake_data.get("correction", ""),
                            explanation=mistake_data.get("explanation", ""),
                        )
                    )

            # Extract strengths and suggestions
            strengths = data.get("strengths", [])
            if not isinstance(strengths, list):
                strengths = [strengths] if strengths else []

            suggestions = data.get("suggestions", [])
            if not isinstance(suggestions, list):
                suggestions = [suggestions] if suggestions else []

            return ConversationFeedback(
                grammar_score=grammar_score,
                final_score=final_score,
                mistakes=mistakes,
                strengths=strengths[:5],  # Limit to 5
                suggestions=suggestions[:5],  # Limit to 5
            )

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse analysis JSON: {e}")
            logger.debug(f"Raw response: {analysis_json}")
            raise ValueError(f"Invalid JSON in analysis response: {e}")

    def _extract_json(self, text: str) -> str:
        """Extract JSON from text that might contain other content."""
        # Try to find JSON object in the text
        text = text.strip()

        # If it starts with {, assume it's JSON
        if text.startswith("{"):
            # Find the matching closing brace
            brace_count = 0
            for i, char in enumerate(text):
                if char == "{":
                    brace_count += 1
                elif char == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        return text[: i + 1]
            return text  # Return as-is if no matching brace found

        # Try to find JSON block in markdown code block
        code_block_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if code_block_match:
            return code_block_match.group(1)

        # Try to find any JSON object
        json_match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
        if json_match:
            return json_match.group(0)

        # Return original text and let JSON parser handle the error
        return text

    def _validate_score(self, score) -> int:
        """Validate and normalize a score to 0-100."""
        try:
            score = int(score)
            return max(0, min(100, score))
        except (ValueError, TypeError):
            return 70  # Default score

    def _empty_feedback(self) -> ConversationFeedback:
        """Return feedback for an empty conversation."""
        return ConversationFeedback(
            grammar_score=0,
            final_score=0,
            mistakes=[],
            strengths=[],
            suggestions=["Try to speak more during the conversation!"],
        )

    def _fallback_feedback(self, utterances: List[str]) -> ConversationFeedback:
        """
        Generate basic feedback when LLM analysis fails.

        Uses simple heuristics to provide some feedback.
        """
        # Simple heuristic-based analysis
        total_words = sum(len(u.split()) for u in utterances)
        avg_words_per_utterance = total_words / len(utterances) if utterances else 0

        # Basic scoring heuristics
        if avg_words_per_utterance < 3:
            grammar_score = 60
            fluency_note = "Try to use longer, more complete sentences."
        elif avg_words_per_utterance < 7:
            grammar_score = 70
            fluency_note = "Good sentence length. Try to vary your sentence structures."
        else:
            grammar_score = 80
            fluency_note = "Great job using longer sentences!"

        final_score = grammar_score

        return ConversationFeedback(
            grammar_score=grammar_score,
            final_score=final_score,
            mistakes=[],
            strengths=["You participated in the conversation!"],
            suggestions=[
                fluency_note,
                "Keep practicing to improve your fluency.",
            ],
        )

    async def quick_grammar_check(self, text: str) -> Optional[str]:
        """
        Perform a quick grammar check on a single utterance.

        Returns the corrected text if there are errors, or None if correct.
        This can be used for real-time feedback during conversation.

        Args:
            text: The user's utterance to check

        Returns:
            Corrected text if errors found, None if text is correct
        """
        if not text or not text.strip():
            return None

        try:
            llm_service = self._get_llm_service()

            if not llm_service.is_loaded:
                return None  # Don't block for quick checks

            from app.prompts.tutor_prompts import get_grammar_check_prompt
            from app.services.llm_service import ConversationContext

            prompt = get_grammar_check_prompt(text)

            context = ConversationContext()
            context.system_prompt = "You are a grammar checker. Be concise."
            context.add_user_message(prompt)

            response = await llm_service.generate(
                context,
                max_tokens=100,
                temperature=0.1,
            )

            response = response.strip()

            if "CORRECT" in response.upper():
                return None

            # Extract correction
            if "CORRECTION:" in response.upper():
                correction = response.split(":", 1)[1].strip()
                return correction

            return None

        except Exception as e:
            logger.debug(f"Quick grammar check failed: {e}")
            return None


# Global service instance (singleton pattern)
_analysis_service: Optional[AnalysisService] = None


def get_analysis_service() -> AnalysisService:
    """Get or create the global analysis service instance."""
    global _analysis_service
    if _analysis_service is None:
        _analysis_service = AnalysisService()
    return _analysis_service
