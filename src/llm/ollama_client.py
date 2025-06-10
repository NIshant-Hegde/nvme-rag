import requests
import json
import logging
from typing import List, Dict, Any, Optional, Iterator, Union
from dataclasses import dataclass, asdict
from enum import Enum
import time

logger = logging.getLogger(__name__)

class OllamaModel(Enum):
    """Available Ollama models"""
    LLAMA2_7B = "llama2:7b"
    LLAMA2_13B = "llama2:13b"
    MISTRAL_7B = "mistral:7b"
    CODELLAMA_7B = "codellama:7b"
    NEURAL_CHAT_7B = "neural-chat:7b"
    ZEPHYR_7B = "zephyr:7b"
    VICUNA_7B = "vicuna:7b"
    GEMMA3_12B_IT_QAT = "gemma3:12b-it-qat"

@dataclass
class OllamaConfig:
    """Configuration for Ollama LLM client"""
    base_url: str = "http://localhost:11434"
    model: str = OllamaModel.MISTRAL_7B.value
    temperature: float = 0.1
    max_tokens: int = 2048
    top_p: float = 0.9
    timeout: int = 120
    keep_alive: str = "5m"
    stream: bool = False

@dataclass
class ChatMessage:
    """Chat message structure"""
    role: str  # "system", "user", "assistant"
    content: str
    
    def to_dict(self) -> Dict[str, str]:
        return {"role": self.role, "content": self.content}

@dataclass
class LLMResponse:
    """LLM response structure"""
    content: str
    model: str
    total_duration: Optional[int] = None
    load_duration: Optional[int] = None
    prompt_eval_count: Optional[int] = None
    eval_count: Optional[int] = None
    eval_duration: Optional[int] = None
    done: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class OllamaClient:
    """
    Professional Ollama LLM client for RAG applications
    Provides both streaming and non-streaming chat capabilities
    """
    
    def __init__(self, config: OllamaConfig = None):
        """
        Initialize Ollama client
        
        Args:
            config: Ollama configuration
        """
        self.config = config or OllamaConfig()
        self.session = requests.Session()
        self.session.timeout = self.config.timeout
        
        # Verify Ollama server connection
        self._verify_connection()
        
        # Verify model availability
        self._verify_model()
        
        logger.info(f"Ollama client initialized with model: {self.config.model}")
    
    def _verify_connection(self):
        """Verify connection to Ollama server"""
        try:
            response = self.session.get(f"{self.config.base_url}/api/tags")
            response.raise_for_status()
            logger.info("Successfully connected to Ollama server")
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to connect to Ollama server at {self.config.base_url}: {e}")
            raise ConnectionError(f"Cannot connect to Ollama server: {e}")
    
    def _verify_model(self):
        """Verify that the specified model is available"""
        try:
            response = self.session.get(f"{self.config.base_url}/api/tags")
            response.raise_for_status()
            
            available_models = response.json().get("models", [])
            model_names = [model["name"] for model in available_models]
            
            if self.config.model not in model_names:
                logger.warning(f"Model {self.config.model} not found. Available models: {model_names}")
                logger.info("Attempting to pull model...")
                self._pull_model()
            else:
                logger.info(f"Model {self.config.model} is available")
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to verify model availability: {e}")
            raise
    
    def _pull_model(self):
        """Pull the specified model from Ollama"""
        try:
            payload = {"name": self.config.model}
            response = self.session.post(
                f"{self.config.base_url}/api/pull",
                json=payload,
                stream=True
            )
            response.raise_for_status()
            
            logger.info(f"Pulling model {self.config.model}...")
            
            # Stream the pull progress
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        if "status" in data:
                            logger.info(f"Pull status: {data['status']}")
                        if data.get("status") == "success":
                            logger.info(f"Successfully pulled model {self.config.model}")
                            return
                    except json.JSONDecodeError:
                        continue
                        
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to pull model {self.config.model}: {e}")
            raise
    
    def chat(self, messages: List[ChatMessage], stream: bool = None) -> Union[LLMResponse, Iterator[LLMResponse]]:
        """
        Send chat messages to Ollama
        
        Args:
            messages: List of chat messages
            stream: Whether to stream response (overrides config)
            
        Returns:
            LLMResponse or Iterator of LLMResponse if streaming
        """
        stream = self.config.stream if stream is None else stream
        
        try:
            payload = {
                "model": self.config.model,
                "messages": [msg.to_dict() for msg in messages],
                "stream": stream,
                "options": {
                    "temperature": self.config.temperature,
                    "top_p": self.config.top_p,
                    "num_predict": self.config.max_tokens
                },
                "keep_alive": self.config.keep_alive
            }
            
            response = self.session.post(
                f"{self.config.base_url}/api/chat",
                json=payload,
                stream=stream
            )
            response.raise_for_status()
            
            if stream:
                return self._handle_streaming_response(response)
            else:
                return self._handle_single_response(response)
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Chat request failed: {e}")
            raise
    
    def _handle_single_response(self, response: requests.Response) -> LLMResponse:
        """Handle non-streaming response"""
        try:
            data = response.json()
            
            message_content = data.get("message", {}).get("content", "")
            
            return LLMResponse(
                content=message_content,
                model=data.get("model", self.config.model),
                total_duration=data.get("total_duration"),
                load_duration=data.get("load_duration"),
                prompt_eval_count=data.get("prompt_eval_count"),
                eval_count=data.get("eval_count"),
                eval_duration=data.get("eval_duration"),
                done=data.get("done", True)
            )
            
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to parse response: {e}")
            raise
    
    def _handle_streaming_response(self, response: requests.Response) -> Iterator[LLMResponse]:
        """Handle streaming response"""
        try:
            content_buffer = ""
            
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line)
                        
                        message_data = data.get("message", {})
                        content_chunk = message_data.get("content", "")
                        content_buffer += content_chunk
                        
                        yield LLMResponse(
                            content=content_chunk,
                            model=data.get("model", self.config.model),
                            total_duration=data.get("total_duration"),
                            load_duration=data.get("load_duration"),
                            prompt_eval_count=data.get("prompt_eval_count"),
                            eval_count=data.get("eval_count"),
                            eval_duration=data.get("eval_duration"),
                            done=data.get("done", False)
                        )
                        
                        if data.get("done", False):
                            break
                            
                    except json.JSONDecodeError:
                        continue
                        
        except Exception as e:
            logger.error(f"Streaming response failed: {e}")
            raise
    
    def generate_completion(self, prompt: str, stream: bool = None) -> Union[LLMResponse, Iterator[LLMResponse]]:
        """
        Generate completion for a single prompt
        
        Args:
            prompt: Input prompt
            stream: Whether to stream response
            
        Returns:
            LLMResponse or Iterator of LLMResponse if streaming
        """
        messages = [ChatMessage(role="user", content=prompt)]
        return self.chat(messages, stream=stream)
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """
        Get list of available models
        
        Returns:
            List of model information dictionaries
        """
        try:
            response = self.session.get(f"{self.config.base_url}/api/tags")
            response.raise_for_status()
            return response.json().get("models", [])
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get available models: {e}")
            return []
    
    def get_model_info(self, model_name: str = None) -> Dict[str, Any]:
        """
        Get information about a specific model
        
        Args:
            model_name: Model name (defaults to current model)
            
        Returns:
            Model information dictionary
        """
        model_name = self.config.model if model_name is None else model_name
        
        try:
            payload = {"name": model_name}
            response = self.session.post(
                f"{self.config.base_url}/api/show",
                json=payload
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get model info for {model_name}: {e}")
            return {}
    
    def delete_model(self, model_name: str) -> bool:
        """
        Delete a model from Ollama
        
        Args:
            model_name: Name of model to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            payload = {"name": model_name}
            response = self.session.delete(
                f"{self.config.base_url}/api/delete",
                json=payload
            )
            response.raise_for_status()
            logger.info(f"Successfully deleted model: {model_name}")
            return True
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to delete model {model_name}: {e}")
            return False
    
    def update_config(self, **kwargs):
        """Update configuration parameters"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                logger.info(f"Updated config: {key} = {value}")
            else:
                logger.warning(f"Unknown config parameter: {key}")
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on Ollama server
        
        Returns:
            Health status information
        """
        try:
            start_time = time.time()
            
            # Test basic connectivity
            response = self.session.get(f"{self.config.base_url}/api/tags")
            response.raise_for_status()
            
            response_time = time.time() - start_time
            
            # Test model availability
            models = response.json().get("models", [])
            model_available = any(model["name"] == self.config.model for model in models)
            
            return {
                "status": "healthy",
                "response_time_seconds": round(response_time, 3),
                "server_url": self.config.base_url,
                "model": self.config.model,
                "model_available": model_available,
                "total_models": len(models),
                "timestamp": time.time()
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "server_url": self.config.base_url,
                "model": self.config.model,
                "timestamp": time.time()
            }
    
    def cleanup(self):
        """Cleanup resources"""
        if hasattr(self, 'session'):
            self.session.close()
        logger.info("Ollama client cleanup completed")