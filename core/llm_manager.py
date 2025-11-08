"""
Language Model Manager for Ollama API interactions.

This module handles all communication with the Ollama API for running
local language models.
"""

import requests
from typing import Tuple


class OllamaManager:
    """
    Manages interactions with Ollama API.
    
    Ollama provides a local API for running language models. This class
    handles API calls, token estimation, and error handling.
    """
    
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "deepseek-r1:1.5b"):
        """
        Initialize the Ollama manager.
        
        Args:
            base_url: Base URL for the Ollama API
            model: Model name to use (e.g., "deepseek-r1:1.5b")
        """
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.api_url = f"{self.base_url}/api/generate"

    def call_api(self, prompt: str, temperature: float = 0.0, max_tokens: int = 2000) -> Tuple[str, int]:
        """
        Call Ollama API and return response text and estimated token count.
        
        Args:
            prompt: Input prompt for the model
            temperature: Sampling temperature (0.0 = greedy, higher = more random)
            max_tokens: Maximum tokens to generate
            
        Returns:
            Tuple of (response_text, estimated_token_count)
            
        Raises:
            RuntimeError: If API call fails
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            }
        }
        
        try:
            response = requests.post(self.api_url, json=payload, timeout=120)
            response.raise_for_status()
            
            result = response.json()
            text = result.get("response", "").strip()
            
            # Rough token estimation (4 chars per token)
            estimated_tokens = len(prompt) // 4 + len(text) // 4
            
            return text, estimated_tokens
            
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Ollama API call failed: {e}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error during Ollama API call: {e}")

    def get_model(self) -> str:
        """Return the current model name."""
        return self.model
    
    def set_model(self, model: str):
        """Change the model being used."""
        self.model = model