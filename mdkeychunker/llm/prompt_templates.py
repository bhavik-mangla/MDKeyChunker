"""LLM prompt templates."""


class PromptTemplates:
    """Centralized prompt templates for LLM interactions."""
    
    @staticmethod
    def summary_generation(chunk_text: str, section_title: str, max_sentences: int = 2) -> str:
        """
        Generate prompt for chunk summarization.
        
        Args:
            chunk_text: The chunk text to summarize
            section_title: Section/header context
            max_sentences: Maximum sentences in summary
            
        Returns:
            Formatted prompt string
        """
        return f"""Summarize the following text in {max_sentences} concise sentence(s). Focus on key information and main points.

Section: {section_title}

Text:
{chunk_text}

Provide only the summary, no additional commentary."""
    
    @staticmethod
    def question_generation(chunk_text: str, section_title: str, num_questions: int = 3) -> str:
        """
        Generate prompt for question creation.
        
        Args:
            chunk_text: The chunk text
            section_title: Section context
            num_questions: Number of questions to generate
            
        Returns:
            Formatted prompt string
        """
        return f"""Generate {num_questions} clear, specific questions that this text chunk can answer. Questions should be useful for retrieval and search.

Section: {section_title}

Text:
{chunk_text}

Output format (JSON only):
{{
  "questions": ["question 1", "question 2", "question 3"]
}}"""
    
    @staticmethod
    def key_matching(
        chunk_text: str,
        section_title: str,
        candidate_keys: list,
        scores: dict
    ) -> str:
        """
        Generate prompt for ambiguous key matching resolution.
        
        Args:
            chunk_text: Current chunk text
            section_title: Section context
            candidate_keys: List of candidate key names
            scores: Dictionary of key -> score
            
        Returns:
            Formatted prompt string
        """
        keys_info = "\n".join([
            f"- {key} (score: {scores.get(key, 0.0):.3f})"
            for key in candidate_keys
        ])
        
        return f"""You are analyzing relationships between text chunks in a document. Determine which of the following candidate keys/concepts are meaningfully related to the current chunk.

Section: {section_title}

Current Chunk:
{chunk_text}

Candidate Keys:
{keys_info}

Consider:
1. Does the chunk reference or continue discussion of the key?
2. Is there thematic or semantic connection?
3. Are there shared entities, concepts, or topics?

Output ONLY valid JSON in this exact format:
{{
  "matched": ["key1", "key2"],
  "rejected": ["key3"],
  "reasons": ["brief reason for key1", "brief reason for key2"],
  "confidence": {{"key1": 0.85, "key2": 0.75}}
}}

Be conservative - only match if there's clear connection."""
    
    @staticmethod
    def keyword_refinement(chunk_text: str, extracted_keywords: list) -> str:
        """
        Generate prompt for keyword refinement.
        
        Args:
            chunk_text: The chunk text
            extracted_keywords: Initially extracted keywords
            
        Returns:
            Formatted prompt string
        """
        keywords_str = ", ".join(extracted_keywords)
        
        return f"""Refine the following extracted keywords for this text. Keep the most relevant, remove generic terms, and optionally add 1-2 missing important concepts.

Text:
{chunk_text}

Extracted Keywords:
{keywords_str}

Output format (JSON only):
{{
  "keywords": ["keyword1", "keyword2", "keyword3"]
}}

Maximum 8 keywords."""


# Example expected responses for documentation and testing
EXAMPLE_RESPONSES = {
    "summary": {
        "input": "The transformer architecture uses self-attention mechanisms to process sequences in parallel, achieving state-of-the-art results on various NLP tasks.",
        "output": "The transformer architecture leverages self-attention for parallel sequence processing. It achieves state-of-the-art performance across multiple NLP tasks."
    },
    
    "questions": {
        "input": "The transformer architecture uses self-attention mechanisms...",
        "output": {
            "questions": [
                "What mechanism does the transformer architecture use for sequence processing?",
                "What advantage does the transformer have over sequential models?",
                "What types of tasks does the transformer excel at?"
            ]
        }
    },
    
    "key_matching": {
        "input": "Candidate keys: transformer, attention, neural networks",
        "output": {
            "matched": ["transformer", "attention"],
            "rejected": ["neural networks"],
            "reasons": [
                "Directly discusses transformer architecture",
                "Self-attention is core mechanism"
            ],
            "confidence": {
                "transformer": 0.95,
                "attention": 0.90
            }
        }
    },
    
    "keyword_refinement": {
        "input": ["transformer", "architecture", "attention", "mechanism", "processing"],
        "output": {
            "keywords": ["transformer architecture", "self-attention", "parallel processing", "NLP"]
        }
    }
}
