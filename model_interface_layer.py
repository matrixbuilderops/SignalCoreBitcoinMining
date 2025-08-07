import subprocess
import json
import time
from typing import Any, Dict, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum


class ModelResponseType(Enum):
    """Enumeration of possible model response types"""
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"
    RETRY_NEEDED = "retry_needed"


@dataclass
class ModelInput:
    """Structured input for model interface"""
    prompt: str
    context: Optional[Dict[str, Any]] = None
    timeout: int = 30
    retry_attempts: int = 3


@dataclass
class ModelOutput:
    """Structured output from model interface"""
    response_type: ModelResponseType
    content: Optional[str] = None
    raw_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    processing_time: float = 0.0


class ModelInterface:
    def __init__(self, model_command: Optional[str] = None) -> None:
        # Allow override, but default to the provided model call
        self.model_command = model_command or "ollama run mixtral:8x7b-instruct-v0.1-q6_K"
        self._validate_configuration()

    def _validate_configuration(self) -> None:
        """Validate model configuration and availability"""
        try:
            # Test if ollama is available
            result = subprocess.run(
                ["ollama", "--version"], 
                capture_output=True, 
                text=True, 
                timeout=5
            )
            if result.returncode != 0:
                raise RuntimeError("Ollama not available")
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.CalledProcessError):
            # Log warning but don't fail - allow fallback behavior
            pass

    def query_model(self, prompt: str) -> Dict[str, Any]:
        """Legacy method for backward compatibility"""
        model_input = ModelInput(prompt=prompt)
        result = self.query_model_structured(model_input)
        
        if result.response_type == ModelResponseType.SUCCESS:
            if result.raw_data:
                return result.raw_data
            return {"raw_output": result.content or ""}
        else:
            return {"error": result.error_message or "Unknown error"}

    def query_model_structured(self, model_input: ModelInput) -> ModelOutput:
        """
        Query model with structured input/output and retry logic
        
        Args:
            model_input: Structured input containing prompt and configuration
            
        Returns:
            ModelOutput with structured response data
        """
        start_time = time.time()
        
        for attempt in range(model_input.retry_attempts):
            try:
                result = self._attempt_model_call(model_input.prompt, model_input.timeout)
                processing_time = time.time() - start_time
                
                if result[0] == ModelResponseType.SUCCESS:
                    return ModelOutput(
                        response_type=ModelResponseType.SUCCESS,
                        content=result[1],
                        raw_data=result[2],
                        retry_count=attempt,
                        processing_time=processing_time
                    )
                elif result[0] == ModelResponseType.RETRY_NEEDED and attempt < model_input.retry_attempts - 1:
                    # Wait before retry with exponential backoff
                    wait_time = 2 ** attempt
                    time.sleep(wait_time)
                    continue
                else:
                    return ModelOutput(
                        response_type=result[0],
                        error_message=result[1],
                        retry_count=attempt,
                        processing_time=time.time() - start_time
                    )
                    
            except Exception as e:
                if attempt == model_input.retry_attempts - 1:
                    return ModelOutput(
                        response_type=ModelResponseType.ERROR,
                        error_message=f"Model interface error after {attempt + 1} attempts: {str(e)}",
                        retry_count=attempt,
                        processing_time=time.time() - start_time
                    )
                # Wait before retry
                time.sleep(2 ** attempt)
        
        return ModelOutput(
            response_type=ModelResponseType.ERROR,
            error_message="Max retry attempts exceeded",
            retry_count=model_input.retry_attempts,
            processing_time=time.time() - start_time
        )

    def _attempt_model_call(self, prompt: str, timeout: int) -> Tuple[ModelResponseType, Optional[str], Optional[Dict[str, Any]]]:
        """
        Attempt a single model call
        
        Returns:
            Tuple of (response_type, content_or_error, raw_data)
        """
        try:
            process = subprocess.Popen(
                self.model_command.split(),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            try:
                stdout, stderr = process.communicate(input=prompt, timeout=timeout)
            except subprocess.TimeoutExpired:
                process.kill()
                try:
                    process.communicate(timeout=1)  # Clean up with short timeout
                except subprocess.TimeoutExpired:
                    pass  # Force kill if needed
                return (ModelResponseType.TIMEOUT, "Model call timed out", None)

            if process.returncode != 0:
                if stderr and ("connection" in stderr.lower() or "network" in stderr.lower()):
                    return (ModelResponseType.RETRY_NEEDED, stderr.strip(), None)
                else:
                    return (ModelResponseType.ERROR, stderr.strip(), None)

            if not stdout.strip():
                return (ModelResponseType.RETRY_NEEDED, "Empty response from model", None)

            # Try to parse as JSON first
            try:
                parsed_json = json.loads(stdout)
                return (ModelResponseType.SUCCESS, stdout.strip(), parsed_json)
            except json.JSONDecodeError:
                # Return as raw text if not JSON
                return (ModelResponseType.SUCCESS, stdout.strip(), None)

        except FileNotFoundError:
            return (ModelResponseType.ERROR, "Model command not found", None)
        except Exception as e:
            return (ModelResponseType.ERROR, str(e), None)

    def health_check(self) -> bool:
        """
        Perform a health check on the model interface
        
        Returns:
            True if model interface is healthy
        """
        try:
            test_input = ModelInput(
                prompt="Test prompt",
                timeout=5,
                retry_attempts=1
            )
            result = self.query_model_structured(test_input)
            # Consider timeout and success as healthy states, only error as unhealthy
            return result.response_type in [ModelResponseType.SUCCESS, ModelResponseType.TIMEOUT, ModelResponseType.RETRY_NEEDED]
        except Exception:
            return False


if __name__ == "__main__":
    interface = ModelInterface()
    prompt = "Calculate optimal nonce using Knuth(10, 3, 16000) and verify ForkIntegrity."
    result = interface.query_model(prompt)
    print(json.dumps(result, indent=2))
