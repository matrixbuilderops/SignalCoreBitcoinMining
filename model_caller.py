"""
model_caller.py — Interfaces with LLaMA/other local AI

AI model interface for Bitcoin mining analysis using Ollama and Level 16000 math context.
Integrates with existing ai_interface.py functionality.
"""

import subprocess  # nosec B404
import json
from typing import Dict, Any, Optional
from ai_interface import call_ai_model, extract_recommendation


class ModelCaller:
    """
    Interface for calling local AI models with Bitcoin mining context.
    """

    def __init__(self, model_name: str = "mixtral:8x7b-instruct-v0.1-q6_K", verbose: bool = True):
        """
        Initialize the model caller.

        Args:
            model_name: Ollama model name to use
            verbose: Enable verbose logging
        """
        self.model_name = model_name
        self.verbose = verbose
        self.calls_made = 0
        self.successful_calls = 0

    def log(self, message: str) -> None:
        """Log message if verbose mode is enabled."""
        if self.verbose:
            print(f"[ModelCaller] {message}")

    def inject_math_context(self, validation_data: Dict[str, Any]) -> str:
        """
        Inject Level 16000 math context into model prompt.

        Args:
            validation_data: Validation results from math processing

        Returns:
            Enhanced prompt with math context
        """
        # Extract math values from validation data
        level = validation_data.get("level", 16000)
        sorrell = validation_data.get("sorrell", 0)
        fork_cluster = validation_data.get("fork_cluster", 0)
        over_recursion = validation_data.get("over_recursion", 0)
        bit_load = validation_data.get("bit_load", 1600000)
        cycles = validation_data.get("cycles", 161)

        math_context = f"""
LEVEL {level} MATH CONTEXT:
Pre-Safeguards:
 - DriftCheck: {"PASS" if validation_data.get("pre_drift", False) else "FAIL"}
 - ForkIntegrity: {"PASS" if validation_data.get("fork_integrity", False) else "FAIL"}
 - RecursionSync: {"PASS" if validation_data.get("recursion_sync", False) else "FAIL"}
 - EntropyParity: {"PASS" if validation_data.get("entropy_parity", False) else "FAIL"}
 - SHA512 Stabilizer (Pre): {validation_data.get("pre_stabilizer", "N/A")[:64]}...

Main Equation:
 - Sorrell: Knuth(10, 3, {level}) = {sorrell}
 - ForkCluster: Knuth(10, 3, {level}) = {fork_cluster}
 - OverRecursion: Knuth(10, 3, {level}) = {over_recursion}
 - BitLoad: {bit_load}
 - Sandboxes: 1
 - Cycles: {cycles}

Post-Safeguards:
 - SHA512 Stabilizer (Post): {validation_data.get("post_stabilizer", "N/A")[:64]}...
 - DriftCheck: {"PASS" if validation_data.get("post_drift", False) else "FAIL"}
 - RecursionSync: {"PASS" if validation_data.get("post_recursion_sync", False) else "FAIL"}
 - ForkSync: {"PASS" if validation_data.get("fork_sync", False) else "FAIL"}
"""
        return math_context

    def call_model_with_math(self, validation_data: Dict[str, Any], block_hash: str = "") -> str:
        """
        Call the AI model with Level 16000 math context injected.

        Args:
            validation_data: Validation results from math processing
            block_hash: Block hash for context

        Returns:
            AI model response
        """
        self.calls_made += 1
        self.log(f"Calling model (attempt {self.calls_made})...")

        # Inject math context into validation data
        math_context = self.inject_math_context(validation_data)
        
        # Enhance validation data with math context
        enhanced_data = {
            **validation_data,
            "math_context": math_context,
            "model_prompt_type": "level_16000_mining"
        }

        try:
            # Use existing ai_interface functionality
            response = call_ai_model(enhanced_data, block_hash)
            
            if not response.startswith("AI_"):
                self.successful_calls += 1
                self.log("Model call successful")
            else:
                self.log(f"Model call failed: {response}")
            
            return response

        except Exception as e:
            self.log(f"Model call error: {str(e)}")
            return f"MODEL_ERROR: {str(e)}"

    def direct_ollama_call(self, prompt: str, timeout: int = 30) -> str:
        """
        Direct call to Ollama with custom prompt.

        Args:
            prompt: Custom prompt text
            timeout: Timeout in seconds

        Returns:
            Model response
        """
        try:
            self.log("Making direct Ollama call...")
            
            result = subprocess.run(  # nosec B603 B607
                ["ollama", "run", self.model_name],
                input=prompt,
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            if result.returncode == 0:
                self.log("Direct Ollama call successful")
                return result.stdout.strip()
            else:
                return f"OLLAMA_ERROR: {result.stderr.strip()}"

        except subprocess.TimeoutExpired:
            return "OLLAMA_TIMEOUT: Model response took too long"
        except subprocess.CalledProcessError as e:
            return f"OLLAMA_CALL_ERROR: {str(e)}"
        except Exception as e:
            return f"OLLAMA_UNKNOWN_ERROR: {str(e)}"

    def analyze_mining_opportunity(self, validation_data: Dict[str, Any], block_hash: str = "") -> Dict[str, Any]:
        """
        Comprehensive mining opportunity analysis.

        Args:
            validation_data: Math validation results
            block_hash: Current block hash

        Returns:
            Analysis results with recommendation
        """
        self.log("Analyzing mining opportunity...")

        # Get AI response
        ai_response = self.call_model_with_math(validation_data, block_hash)
        
        # Extract recommendation
        recommendation = extract_recommendation(ai_response)
        
        # Analyze validation results
        critical_checks = [
            validation_data.get("fork_integrity", False),
            validation_data.get("entropy_parity", False),
            validation_data.get("fork_sync", False),
            validation_data.get("pre_drift", False),
            validation_data.get("post_drift", False)
        ]
        
        validation_score = sum(critical_checks) / len(critical_checks) * 100
        
        analysis = {
            "recommendation": recommendation,
            "ai_response": ai_response,
            "validation_score": validation_score,
            "critical_checks_passed": sum(critical_checks),
            "total_checks": len(critical_checks),
            "should_proceed": recommendation == "PROCEED" and validation_score >= 80,
            "block_hash": block_hash,
            "level": validation_data.get("level", 16000)
        }
        
        self.log(f"Analysis complete - Recommendation: {recommendation}, Score: {validation_score:.1f}%")
        return analysis

    def get_model_stats(self) -> Dict[str, Any]:
        """
        Get model calling statistics.

        Returns:
            Dictionary with call statistics
        """
        success_rate = (self.successful_calls / self.calls_made * 100) if self.calls_made > 0 else 0
        
        return {
            "total_calls": self.calls_made,
            "successful_calls": self.successful_calls,
            "success_rate_percent": round(success_rate, 2),
            "model_name": self.model_name
        }


def main():
    """
    Main entry point for testing the model caller.
    """
    # Test data
    test_validation_data = {
        "level": 16000,
        "pre_drift": True,
        "fork_integrity": True,
        "recursion_sync": True,
        "entropy_parity": True,
        "sorrell": 123456,
        "fork_cluster": 234567,
        "over_recursion": 345678,
        "bit_load": 1600000,
        "cycles": 161,
        "post_drift": True,
        "post_recursion_sync": True,
        "fork_sync": True,
        "pre_stabilizer": "941d793ce78e45983a4d98d6e4ed0529d923f06f8ecefcabe45c5448c65333fca",
        "post_stabilizer": "74402f56dc3f9154da10ab8d5dbe518db9aa2a332b223bc7bdca9871d0b1a55c"
    }
    
    test_block_hash = "000000000019d6689c085ae165831e934ff763ae46a2a6c172b3f1b60a8ce26f"

    # Create model caller
    caller = ModelCaller(verbose=True)
    
    # Test analysis
    print("Testing mining opportunity analysis...")
    analysis = caller.analyze_mining_opportunity(test_validation_data, test_block_hash)
    
    print(f"\nAnalysis Results:")
    print(json.dumps(analysis, indent=2))
    
    # Show stats
    stats = caller.get_model_stats()
    print(f"\nModel Stats: {stats}")


if __name__ == "__main__":
    main()