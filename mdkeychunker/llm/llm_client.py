"""LLM client for various providers."""

import json
import logging
from typing import Dict, List, Optional, Any

from ..utils.config import Config
from .prompt_templates import PromptTemplates


class LLMClient:
    """
    Unified LLM client supporting multiple providers.
    
    Supports: OpenAI, Anthropic, and custom endpoints.
    """
    
    def __init__(self, config: Config):
        """Initialize LLM client."""
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.provider = config.llm_provider.lower()
        self.templates = PromptTemplates()
        
        # Initialize provider-specific client
        self.client = None
        if self.provider == "openai":
            self._init_openai()
        elif self.provider == "anthropic":
            self._init_anthropic()
        else:
            self.logger.warning(f"Unknown LLM provider: {self.provider}")
    
    def _init_openai(self) -> None:
        """Initialize OpenAI client."""
        if not self.config.llm_api_key:
            self.logger.warning("No OpenAI API key provided. LLM features disabled.")
            return
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=self.config.llm_api_key)
            self.logger.info("OpenAI client initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize OpenAI client: {e}")
    
    def _init_anthropic(self) -> None:
        """Initialize Anthropic client."""
        if not self.config.llm_api_key:
            self.logger.warning("No Anthropic API key provided. LLM features disabled.")
            return
        try:
            from anthropic import Anthropic
            self.client = Anthropic(api_key=self.config.llm_api_key)
            self.logger.info("Anthropic client initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize Anthropic client: {e}")
    
    def generate_summary(self, chunk_text: str, section_title: str) -> str:
        """
        Generate a concise summary for a chunk.
        
        Args:
            chunk_text: The text to summarize
            section_title: Section context
            
        Returns:
            Summary string (1-2 sentences)
        """
        if not self.client:
            return self._fallback_summary(chunk_text)
        
        prompt = self.templates.summary_generation(chunk_text, section_title)
        
        try:
            response = self._call_llm(prompt, max_tokens=150)
            summary = response.strip()
            
            # Validate: should be 1-2 sentences
            if summary and len(summary.split('.')) <= 3:
                return summary
            else:
                return self._fallback_summary(chunk_text)
        
        except Exception as e:
            self.logger.error(f"Summary generation failed: {e}")
            return self._fallback_summary(chunk_text)
    
    def generate_questions(
        self,
        chunk_text: str,
        section_title: str,
        num_questions: int = 3
    ) -> List[str]:
        """
        Generate questions that the chunk can answer.
        
        Args:
            chunk_text: The text
            section_title: Section context
            num_questions: Number of questions to generate
            
        Returns:
            List of question strings
        """
        if not self.client:
            return []
        
        prompt = self.templates.question_generation(chunk_text, section_title, num_questions)
        
        try:
            response = self._call_llm(prompt, max_tokens=200)
            
            # Parse JSON response
            data = self._parse_json_response(response)
            
            if data and 'questions' in data:
                return data['questions'][:num_questions]
        
        except Exception as e:
            self.logger.error(f"Question generation failed: {e}")
        
        return []
    
    def resolve_key_matches(
        self,
        chunk_text: str,
        section_title: str,
        candidate_keys: List[str],
        scores: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Resolve ambiguous key matches using LLM.
        
        Args:
            chunk_text: Current chunk text
            section_title: Section context
            candidate_keys: List of candidate key names
            scores: Pre-computed scores for each key
            
        Returns:
            Dictionary with 'matched', 'rejected', 'reasons', 'confidence'
        """
        if not self.client:
            return {'matched': [], 'rejected': candidate_keys}
        
        prompt = self.templates.key_matching(chunk_text, section_title, candidate_keys, scores)
        
        try:
            response = self._call_llm(prompt, max_tokens=300)
            
            # Parse JSON response
            data = self._parse_json_response(response)
            
            if data and 'matched' in data:
                return data
            else:
                self.logger.warning("Invalid key matching response from LLM")
        
        except Exception as e:
            self.logger.error(f"Key matching resolution failed: {e}")
        
        return {'matched': [], 'rejected': candidate_keys}
    
    def refine_keywords(self, chunk_text: str, extracted_keywords: List[str]) -> List[str]:
        """
        Refine extracted keywords using LLM.
        
        Args:
            chunk_text: The text
            extracted_keywords: Initially extracted keywords
            
        Returns:
            Refined list of keywords
        """
        if not self.client or not extracted_keywords:
            return extracted_keywords
        
        prompt = self.templates.keyword_refinement(chunk_text, extracted_keywords)
        
        try:
            response = self._call_llm(prompt, max_tokens=150)
            
            # Parse JSON response
            data = self._parse_json_response(response)
            
            if data and 'keywords' in data:
                return data['keywords']
        
        except Exception as e:
            self.logger.error(f"Keyword refinement failed: {e}")
        
        return extracted_keywords
    
    def _call_llm(self, prompt: str, max_tokens: int = 500) -> str:
        """
        Call LLM with prompt.
        
        Args:
            prompt: The prompt text
            max_tokens: Maximum tokens in response
            
        Returns:
            Response text
        """
        if self.provider == "openai":
            return self._call_openai(prompt, max_tokens)
        elif self.provider == "anthropic":
            return self._call_anthropic(prompt, max_tokens)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    def _call_openai(self, prompt: str, max_tokens: int) -> str:
        """Call OpenAI API."""
        response = self.client.chat.completions.create(
            model=self.config.llm_model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that provides concise, accurate responses."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.3,
        )
        return response.choices[0].message.content
    
    def _call_anthropic(self, prompt: str, max_tokens: int) -> str:
        """Call Anthropic API."""
        response = self.client.messages.create(
            model=self.config.llm_model,
            max_tokens=max_tokens,
            temperature=0.3,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response.content[0].text
    
    def _parse_json_response(self, response: str) -> Optional[Dict]:
        """
        Parse JSON from LLM response.
        
        Handles cases where LLM adds markdown code blocks or extra text.
        """
        # Try direct parse
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass
        
        # Try extracting JSON from markdown code block
        import re
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # Try finding JSON object in text
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass
        
        self.logger.warning(f"Failed to parse JSON from response: {response[:200]}")
        return None
    
    def _fallback_summary(self, text: str, max_length: int = 200) -> str:
        """Generate fallback summary when LLM is unavailable."""
        # Simple heuristic: take first sentence or two
        sentences = text.split('.')
        summary = sentences[0].strip()
        
        if len(summary) < 50 and len(sentences) > 1:
            summary += '. ' + sentences[1].strip()
        
        if len(summary) > max_length:
            summary = summary[:max_length].rsplit(' ', 1)[0] + '...'
        
        return summary + '.' if not summary.endswith('.') else summary
