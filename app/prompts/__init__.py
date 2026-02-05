"""
Prompts module for AI-Talk English tutor.

Contains system prompts for:
- Conversation mode (English tutoring)
- Analysis mode (grammar and fluency evaluation)
"""

from .tutor_prompts import (
    UserLevel,
    get_analysis_prompt,
    get_tutor_system_prompt,
)

__all__ = [
    "get_tutor_system_prompt",
    "get_analysis_prompt",
    "UserLevel",
]
