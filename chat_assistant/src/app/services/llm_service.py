"""
Real LLM service for Medical Quiz Assistant.
Implements Ollama integration per ADR-001.
"""

import json
import logging
from typing import Any, Optional

import httpx

from app.core.config import Settings
from app.domain.models import Question
from app.domain.schemas import RAGExplainOutput

logger = logging.getLogger(__name__)


class OllamaLLMService:
    """Service for LLM inference using Ollama."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.base_url = f"http://{settings.ollama_host}:{settings.ollama_port}"
        self.model_name = settings.llm_model
        self.temperature = settings.llm_temperature
        self.max_tokens = settings.llm_max_tokens
        self.timeout = 30.0

        # HTTP client for async requests
        self.client = httpx.AsyncClient(timeout=self.timeout)

    async def initialize(self) -> None:
        """Initialize the LLM service."""
        try:
            # Check if Ollama is running and model is available
            await self._check_ollama_health()
            await self._ensure_model_available()

            logger.info(f"LLM service initialized with model {self.model_name}")

        except Exception as e:
            logger.error(f"Error initializing LLM service: {e}")
            raise

    async def _check_ollama_health(self) -> None:
        """Check if Ollama service is running."""
        try:
            response = await self.client.get(f"{self.base_url}/api/tags")
            response.raise_for_status()

            logger.info("Ollama service is running")

        except Exception as e:
            logger.error(f"Ollama service not available: {e}")
            raise RuntimeError(f"Cannot connect to Ollama at {self.base_url}")

    async def _ensure_model_available(self) -> None:
        """Ensure the required model is available."""
        try:
            response = await self.client.get(f"{self.base_url}/api/tags")
            response.raise_for_status()

            models = response.json().get("models", [])
            model_names = [model["name"] for model in models]

            if self.model_name not in model_names:
                logger.warning(
                    f"Model {self.model_name} not found. Available models: {model_names}"
                )
                # Try to pull the model
                await self._pull_model()

        except Exception as e:
            logger.error(f"Error checking model availability: {e}")
            raise

    async def _pull_model(self) -> None:
        """Pull the required model from Ollama."""
        try:
            logger.info(f"Pulling model {self.model_name}...")

            response = await self.client.post(
                f"{self.base_url}/api/pull", json={"name": self.model_name}
            )
            response.raise_for_status()

            logger.info(f"Model {self.model_name} pulled successfully")

        except Exception as e:
            logger.error(f"Error pulling model {self.model_name}: {e}")
            raise

    async def generate_explanation(
        self, question: Question, context: list[dict[str, Any]]
    ) -> RAGExplainOutput:
        """
        Generate RAG-enhanced explanation.

        Args:
            question: Question to explain
            context: Retrieved context documents

        Returns:
            RAG explanation output
        """
        try:
            # Build prompt
            prompt = self._build_explanation_prompt(question, context)

            # Generate response
            response = await self._generate_response(prompt)

            # Parse response
            explanation = self._parse_explanation_response(response, question)

            return explanation

        except Exception as e:
            logger.error(f"Error generating explanation: {e}")
            # Return fallback explanation
            return self._create_fallback_explanation(question)

    def _build_explanation_prompt(self, question: Question, context: list[dict[str, Any]]) -> str:
        """Build prompt for explanation generation with correct option handling."""

        # Build context string
        context_text = self._format_context(context)

        # Get correct option label
        correct_label = chr(65 + question.correct_idx)  # 0->A, 1->B, 2->C, 3->D

        # Build incorrect option labels
        incorrect_labels = []
        for i in range(len(question.options)):
            if i != question.correct_idx:
                incorrect_labels.append(chr(65 + i))

        prompt = f"""You are a medical education expert. Provide a detailed explanation for the following multiple-choice question.

    Question: {question.stem}

    Options:
    A) {question.options[0]}
    B) {question.options[1]}
    C) {question.options[2]}
    D) {question.options[3]}

    Correct Answer: {correct_label} ({question.options[question.correct_idx]})

    Context from medical literature:
    {context_text}

    IMPORTANT: Option {correct_label} is the CORRECT answer. Do NOT include option {correct_label} in the reason_incorrect list.

    Please provide a structured explanation in JSON format with the following fields:
    - reason_correct: Detailed explanation of why option {correct_label} is correct
    - reason_incorrect: List of objects explaining why the OTHER options ({', '.join(incorrect_labels)}) are incorrect
    - key_points: List of key learning points
    - citations: List of source document IDs or references
    - confidence: Confidence score between 0.0 and 1.0

    Response format (JSON only):
    {{
        "reason_correct": "Detailed explanation of why {correct_label} is correct...",
        "reason_incorrect": [
            {{"option_idx": "{incorrect_labels[0]}", "short_reason": "Why option {incorrect_labels[0]} is wrong"}},
            {{"option_idx": "{incorrect_labels[1]}", "short_reason": "Why option {incorrect_labels[1]} is wrong"}},
            {{"option_idx": "{incorrect_labels[2]}", "short_reason": "Why option {incorrect_labels[2]} is wrong"}}
        ],
        "key_points": ["Key point 1", "Key point 2", "Key point 3"],
        "citations": ["doc1", "doc2"],
        "confidence": 0.85
    }}"""

        return prompt

    def _format_context(self, context: list[dict[str, Any]]) -> str:
        """Format context documents for the prompt."""
        if not context:
            return "No additional context available."

        context_parts = []
        for i, doc in enumerate(context[:5]):  # Limit to top 5 documents
            content = doc.get("content", "")
            metadata = doc.get("metadata", {})

            context_part = f"Source {i+1}: {content}"
            if metadata.get("question_id"):
                context_part += f" (from question {metadata['question_id']})"

            context_parts.append(context_part)

        return "\n\n".join(context_parts)

    async def _generate_response(self, prompt: str) -> str:
        """Generate response from Ollama."""
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens,
                    "top_p": 0.9,
                    "top_k": 40,
                },
            }

            response = await self.client.post(f"{self.base_url}/api/generate", json=payload)
            response.raise_for_status()

            result = response.json()
            return result.get("response", "")

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise

    def _parse_explanation_response(self, response: str, question: Question) -> RAGExplainOutput:
        """Parse LLM response into RAGExplainOutput - FIXED schema validation."""
        try:
            # Extract JSON from response
            json_start = response.find("{")
            json_end = response.rfind("}") + 1

            if json_start == -1 or json_end == 0:
                raise ValueError("No JSON found in response")

            json_str = response[json_start:json_end]
            data = json.loads(json_str)

            # Parse and fix reason_incorrect format
            reason_incorrect = []
            raw_incorrect = data.get("reason_incorrect", [])
            labels = ["A", "B", "C", "D"]

            for item in raw_incorrect:
                if isinstance(item, dict):
                    option_idx = item.get("option_idx")
                    short_reason = item.get("short_reason", "This option is incorrect.")

                    # Convert integer indices to string labels
                    if isinstance(option_idx, int):
                        if 0 <= option_idx < 4:
                            option_idx = labels[option_idx]  # Convert 0,1,2,3 to A,B,C,D
                        else:
                            option_idx = str(option_idx)

                    reason_incorrect.append(
                        {
                            "option_idx": str(option_idx),  # Ensure it's a string
                            "short_reason": str(short_reason),
                        }
                    )

            # Validate and create RAGExplainOutput
            return RAGExplainOutput(
                reason_correct=data.get("reason_correct", "No explanation provided."),
                reason_incorrect=reason_incorrect,  # Now properly formatted
                key_points=data.get("key_points", []),
                citations=data.get("citations", []),
                confidence=min(max(data.get("confidence", 0.5), 0.0), 1.0),
            )

        except Exception as e:
            logger.warning(f"Error parsing response: {e}")
            return self._create_fallback_explanation(question)

    def _create_fallback_explanation(self, question: Question) -> RAGExplainOutput:
        """Create fallback explanation when LLM fails - FIXED schema validation."""
        labels = ["A", "B", "C", "D"]

        return RAGExplainOutput(
            reason_correct=question.expl_correct or "No explanation available.",
            reason_incorrect=[
                {
                    "option_idx": labels[i],
                    "short_reason": "This option is incorrect.",
                }  # STRING labels
                for i in range(4)
                if i != question.correct_idx
            ],
            key_points=["Key learning point 1", "Key learning point 2"],
            citations=[],
            confidence=0.5,
        )

    async def health_check(self) -> dict[str, Any]:
        """Check if the LLM service is healthy."""
        try:
            # Test with a simple prompt
            test_prompt = "Hello, are you working?"
            response = await self._generate_response(test_prompt)

            return {
                "status": "healthy",
                "model": self.model_name,
                "base_url": self.base_url,
                "response_length": len(response),
            }

        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "model": self.model_name,
                "base_url": self.base_url,
            }

    async def get_model_info(self) -> dict[str, Any]:
        """Get information about the current model."""
        try:
            response = await self.client.get(f"{self.base_url}/api/tags")
            response.raise_for_status()

            models = response.json().get("models", [])
            current_model = next(
                (model for model in models if model["name"] == self.model_name), None
            )

            return {
                "model_name": self.model_name,
                "base_url": self.base_url,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "model_info": current_model,
                "available_models": [model["name"] for model in models],
            }

        except Exception as e:
            return {"model_name": self.model_name, "base_url": self.base_url, "error": str(e)}

    async def close(self) -> None:
        """Close the HTTP client."""
        await self.client.aclose()


class LLMCache:
    """Cache for LLM responses to avoid recomputation."""

    def __init__(self, max_size: int = 1000):
        self.cache: dict[str, RAGExplainOutput] = {}
        self.max_size = max_size
        self.access_count: dict[str, int] = {}

    def get(self, prompt_hash: str) -> Optional[RAGExplainOutput]:
        """Get cached response."""
        if prompt_hash in self.cache:
            self.access_count[prompt_hash] = self.access_count.get(prompt_hash, 0) + 1
            return self.cache[prompt_hash]
        return None

    def put(self, prompt_hash: str, response: RAGExplainOutput) -> None:
        """Cache response."""
        if len(self.cache) >= self.max_size:
            # Remove least recently used item
            lru_key = min(self.access_count.keys(), key=lambda k: self.access_count[k])
            del self.cache[lru_key]
            del self.access_count[lru_key]

        self.cache[prompt_hash] = response
        self.access_count[prompt_hash] = 1

    def clear(self) -> None:
        """Clear cache."""
        self.cache.clear()
        self.access_count.clear()

    def stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hit_rate": sum(self.access_count.values()) / max(len(self.cache), 1),
        }
