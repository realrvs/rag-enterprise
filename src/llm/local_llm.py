"""
Local LLM integration using Ollama or HuggingFace transformers.
Supports: Ollama (recommended), Transformers (HuggingFace).
"""

import logging
import time
from typing import Optional, List, Dict, Any

import requests
import json

from src.config import settings

logger = logging.getLogger(__name__)


class LocalLLM:
    """
    Local LLM client supporting Ollama and HuggingFace models.
    """

    def __init__(
        self,
        model_name: str = "qwen2.5-coder:7b",
        ollama_url: str = "http://localhost:11434",
        use_ollama: bool = True,
    ):
        """
        Initialize Local LLM.

        Args:
            model_name: Name of the model to use.
            ollama_url: Ollama API URL (if use_ollama=True).
            use_ollama: If True, use Ollama. Otherwise, use HuggingFace.
        """
        self.model_name = model_name
        self.ollama_url = ollama_url.rstrip("/")
        self.use_ollama = use_ollama
        self.logger = logging.getLogger(f"{__name__}.LocalLLM")
        self._ollama_available = False
        self._hf_available = False

        if use_ollama:
            self._init_ollama()
        else:
            self._init_huggingface()

    def _fix_encoding(self, text: str) -> str:
        """Fix encoding issues for Russian text."""
        if not text:
            return text
        
        try:
            # Try to decode from latin-1 to utf-8
            return text.encode('latin-1').decode('utf-8')
        except (UnicodeEncodeError, UnicodeDecodeError):
            try:
                # Fallback: try windows-1251
                return text.encode('windows-1251').decode('utf-8')
            except (UnicodeEncodeError, UnicodeDecodeError):
                try:
                    # Another fallback
                    return text.encode('cp1251').decode('utf-8')
                except (UnicodeEncodeError, UnicodeDecodeError):
                    return text

    def _init_ollama(self):
        """Initialize Ollama client."""
        self.logger.info(f"🔧 Initializing Ollama LLM: {self.model_name}")

        # Check if Ollama is running
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m.get("name", "") for m in models]

                if self.model_name not in model_names:
                    self.logger.warning(
                        f"⚠️ Model '{self.model_name}' not found in Ollama. "
                        f"Available: {model_names[:5]}..."
                    )
                    self.logger.info(f"📥 Pulling model: {self.model_name}...")
                    self._pull_model()
                else:
                    self.logger.info(f"✅ Ollama model '{self.model_name}' found")
                    self._ollama_available = True
            else:
                self.logger.error(f"❌ Ollama API error: {response.status_code}")
                self.logger.info("💡 Start Ollama with: ollama serve")

        except requests.exceptions.ConnectionError:
            self.logger.error(
                "❌ Could not connect to Ollama. "
                "Make sure Ollama is running: ollama serve"
            )
            self.logger.info(
                "💡 Install Ollama from: https://ollama.ai/download"
            )
            self._ollama_available = False
            return

        self.logger.info(f"✅ Ollama LLM initialized: {self.model_name}")

    def _pull_model(self):
        """Pull model from Ollama."""
        try:
            response = requests.post(
                f"{self.ollama_url}/api/pull",
                json={"name": self.model_name},
                timeout=300,
            )
            if response.status_code == 200:
                self.logger.info(f"✅ Model '{self.model_name}' pulled successfully")
                self._ollama_available = True
            else:
                self.logger.error(f"❌ Failed to pull model: {response.text}")
                self._ollama_available = False
        except Exception as e:
            self.logger.error(f"❌ Error pulling model: {e}")
            self._ollama_available = False

    def _init_huggingface(self):
        """Initialize HuggingFace transformers."""
        self.logger.info(f"🔧 Initializing HuggingFace LLM: {self.model_name}")

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch

            self.logger.info("📥 Loading model and tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                low_cpu_mem_usage=True,
            )
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.logger.info(f"✅ HuggingFace model loaded on {self.device}")
            self._hf_available = True

        except ImportError:
            self.logger.error(
                "❌ transformers library not installed. "
                "Install: pip install transformers torch"
            )
            self._hf_available = False
        except Exception as e:
            self.logger.error(f"❌ Failed to load HuggingFace model: {e}")
            self._hf_available = False

    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 512,
        top_p: float = 0.95,
        stop: Optional[List[str]] = None,
    ) -> str:
        """
        Generate text using LLM.

        Args:
            prompt: Input prompt.
            temperature: Sampling temperature (0.0-1.0).
            max_tokens: Maximum tokens to generate.
            top_p: Nucleus sampling parameter.
            stop: Stop sequences.

        Returns:
            Generated text.
        """
        if self.use_ollama:
            return self._generate_ollama(prompt, temperature, max_tokens, top_p, stop)
        else:
            return self._generate_huggingface(prompt, temperature, max_tokens, top_p, stop)

    def _generate_ollama(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 512,
        top_p: float = 0.95,
        stop: Optional[List[str]] = None,
    ) -> str:
        """Generate using Ollama API."""
        if not self._ollama_available:
            return "⚠️ Ollama is not available. Please start Ollama: ollama serve"

        try:
            start_time = time.time()

            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                    "top_p": top_p,
                },
            }

            if stop:
                payload["options"]["stop"] = stop

            self.logger.info(f"🔄 Generating with Ollama (max_tokens={max_tokens})...")
            
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
                timeout=300,
            )

            if response.status_code == 200:
                result = response.json()
                generated_text = result.get("response", "")
                
                # Fix encoding for Russian text
                generated_text = self._fix_encoding(generated_text)
                
                elapsed_ms = (time.time() - start_time) * 1000

                self.logger.info(
                    f"✅ Generated {len(generated_text)} chars in {elapsed_ms:.0f}ms"
                )
                return generated_text.strip()
            else:
                self.logger.error(f"❌ Ollama API error: {response.status_code}")
                return f"Error: {response.text}"

        except requests.exceptions.Timeout:
            self.logger.error("❌ Ollama request timed out")
            return "Error: Request timed out"
        except Exception as e:
            self.logger.error(f"❌ Ollama generation error: {e}")
            return f"Error: {str(e)}"

    def _generate_huggingface(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 512,
        top_p: float = 0.95,
        stop: Optional[List[str]] = None,
    ) -> str:
        """Generate using HuggingFace transformers."""
        if not self._hf_available:
            return "⚠️ HuggingFace model is not available."

        try:
            import torch

            start_time = time.time()

            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            generated_text = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )

            # Fix encoding for Russian text
            generated_text = self._fix_encoding(generated_text)

            elapsed_ms = (time.time() - start_time) * 1000
            self.logger.info(
                f"✅ Generated {len(generated_text)} chars in {elapsed_ms:.0f}ms"
            )

            return generated_text.strip()

        except Exception as e:
            self.logger.error(f"❌ HuggingFace generation error: {e}")
            return f"Error: {str(e)}"

    def is_available(self) -> bool:
        """Check if LLM is available."""
        if self.use_ollama:
            return self._ollama_available
        else:
            return self._hf_available

    def list_models(self) -> List[str]:
        """List available Ollama models."""
        if not self.use_ollama:
            return []

        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                return [m.get("name", "") for m in models]
            return []
        except Exception:
            return []

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model."""
        info = {
            "model_name": self.model_name,
            "use_ollama": self.use_ollama,
            "available": self.is_available(),
        }
        
        if self.use_ollama:
            info["ollama_url"] = self.ollama_url
            info["available_models"] = self.list_models()
        else:
            info["hf_available"] = self._hf_available
        
        return info