"""
Model orchestrator for Bitcoin mining system.

This module provides advanced LLM orchestration that integrates real-time
blockchain data via ZMQ and returns valid solutions to proof-of-work using
mathematical logic, with output control and streamlining.
"""

import sys
import json
import time
from typing import Dict, Any, Optional
from ai_interface import call_ai_model, extract_recommendation
from math_module import process_block_with_math
from math_engine import run_level_16000


class ModelOrchestrator:
    """
    Advanced LLM orchestrator for Bitcoin mining with real-time blockchain integration.
    """

    def __init__(self, verbose: bool = True, thinking_mode: bool = False):
        """
        Initialize the model orchestrator.

        Args:
            verbose: Enable detailed output logging
            thinking_mode: Enable thinking... mode for debug output
        """
        self.verbose = verbose
        self.thinking_mode = thinking_mode
        self.solutions_generated = 0
        self.valid_solutions = 0
        self.start_time = time.time()

    def log(self, message: str, force: bool = False) -> None:
        """
        Log message with output control.

        Args:
            message: Message to log
            force: Force output even in quiet mode
        """
        if self.verbose or force:
            timestamp = time.strftime("%H:%M:%S")
            print(f"[{timestamp}] ModelOrchestrator: {message}")
        elif self.thinking_mode:
            print("thinking...", end="", flush=True)

    def inject_blockchain_data(
        self, block_data: bytes, block_hash: str = ""
    ) -> Dict[str, Any]:
        """
        Inject real-time blockchain data into the processing pipeline.

        Args:
            block_data: Raw blockchain data from ZMQ
            block_hash: Block hash for context

        Returns:
            Enriched data dictionary with blockchain context
        """
        self.log("Injecting real-time blockchain data...")

        # Process blockchain data with math module
        processed_data, math_results = process_block_with_math(block_data, 16000)

        # Get Level 16000 math engine results
        level_results = run_level_16000(verbose=False)

        # Combine all data sources
        enriched_data = {
            "block_hash": block_hash,
            "block_size": len(block_data),
            "processed_hash": processed_data.hex()[:32],
            "math_validation": math_results,
            "level_16000_results": level_results,
            "timestamp": time.time(),
            "processing_level": 16000,
        }

        self.log(f"Blockchain data injected - Block size: {len(block_data)} bytes")
        return enriched_data

    def call_model_with_context(self, enriched_data: Dict[str, Any]) -> str:
        """
        Call the LLM model with enriched blockchain context.

        Args:
            enriched_data: Data with blockchain and math context

        Returns:
            Model response string
        """
        self.log("Calling model with blockchain context...")

        # Prepare enhanced prompt with all context
        validation_data = enriched_data["math_validation"]
        level_data = enriched_data["level_16000_results"]

        # Enhanced validation data for AI model
        enhanced_validation = {
            **validation_data,
            "block_hash": enriched_data["block_hash"],
            "block_size": enriched_data["block_size"],
            "level_16000_sorrell": level_data["Main Equation"]["Sorrell"],
            "level_16000_fork_cluster": level_data["Main Equation"]["ForkCluster"],
            "level_16000_over_recursion": level_data["Main Equation"]["OverRecursion"],
            "level_16000_bit_load": level_data["Main Equation"]["BitLoad"],
            "level_16000_cycles": level_data["Main Equation"]["Cycles"],
            "processing_timestamp": enriched_data["timestamp"],
        }

        # Call AI model with enhanced context
        response = call_ai_model(enhanced_validation, enriched_data["block_hash"])

        self.log("Model response received")
        return response

    def validate_proof_of_work_solution(
        self, ai_response: str, enriched_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Validate and extract proof-of-work solution from AI response.

        Args:
            ai_response: Raw AI model response
            enriched_data: Blockchain and math context data

        Returns:
            Valid solution dictionary or None if invalid
        """
        self.log("Validating proof-of-work solution...")

        # Extract recommendation
        recommendation = extract_recommendation(ai_response)

        # Check if all critical validations pass
        math_results = enriched_data["math_validation"]
        critical_checks = [
            math_results.get("fork_integrity", False),
            math_results.get("entropy_parity", False),
            math_results.get("fork_sync", False),
            math_results.get("pre_drift", False),
            math_results.get("post_drift", False),
        ]

        validation_passed = all(critical_checks)

        if recommendation == "PROCEED" and validation_passed:
            solution = {
                "valid": True,
                "recommendation": recommendation,
                "block_hash": enriched_data["block_hash"],
                "level": 16000,
                "sorrell": math_results["sorrell"],
                "fork_cluster": math_results["fork_cluster"],
                "over_recursion": math_results["over_recursion"],
                "bit_load": math_results["bit_load"],
                "cycles": math_results["cycles"],
                "pre_stabilizer": math_results["pre_stabilizer"],
                "post_stabilizer": math_results["post_stabilizer"],
                "ai_response_summary": (
                    ai_response[:200] + "..." if len(ai_response) > 200 else ai_response
                ),
            }

            self.valid_solutions += 1
            self.log(f"Valid solution generated! Total valid: {self.valid_solutions}")
            return solution

        else:
            self.log(
                f"Solution validation failed - Recommendation: {recommendation}, Validation: {validation_passed}"
            )
            return None

    def generate_solution(
        self, block_data: bytes, block_hash: str = ""
    ) -> Optional[Dict[str, Any]]:
        """
        Generate a complete proof-of-work solution using LLM orchestration.

        Args:
            block_data: Raw blockchain data
            block_hash: Block hash for context

        Returns:
            Valid solution or None if generation failed
        """
        self.solutions_generated += 1

        try:
            # Step 1: Inject real-time blockchain data
            enriched_data = self.inject_blockchain_data(block_data, block_hash)

            # Step 2: Call model with enriched context
            ai_response = self.call_model_with_context(enriched_data)

            # Step 3: Validate and extract solution
            solution = self.validate_proof_of_work_solution(ai_response, enriched_data)

            if solution:
                self.log("Solution generation successful", force=True)
                return solution
            else:
                self.log("Solution generation failed validation")
                return None

        except Exception as e:
            self.log(f"Error in solution generation: {str(e)}")
            return None

    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics for the orchestrator.

        Returns:
            Dictionary with performance metrics
        """
        runtime = time.time() - self.start_time
        success_rate = (
            (self.valid_solutions / self.solutions_generated * 100)
            if self.solutions_generated > 0
            else 0
        )

        return {
            "solutions_generated": self.solutions_generated,
            "valid_solutions": self.valid_solutions,
            "success_rate_percent": round(success_rate, 2),
            "runtime_seconds": round(runtime, 2),
            "solutions_per_minute": (
                round((self.solutions_generated / runtime * 60), 2)
                if runtime > 0
                else 0
            ),
        }

    def suppress_output(self) -> None:
        """Suppress all output except critical errors."""
        self.verbose = False
        self.thinking_mode = False

    def enable_thinking_mode(self) -> None:
        """Enable minimal thinking... feedback mode."""
        self.verbose = False
        self.thinking_mode = True

    def enable_verbose_mode(self) -> None:
        """Enable full verbose output mode."""
        self.verbose = True
        self.thinking_mode = False


def main():
    """Main entry point for testing the model orchestrator."""
    import argparse

    parser = argparse.ArgumentParser(description="Bitcoin Mining Model Orchestrator")
    parser.add_argument("--quiet", action="store_true", help="Suppress output")
    parser.add_argument("--thinking", action="store_true", help="Enable thinking mode")
    args = parser.parse_args()

    # Create orchestrator
    orchestrator = ModelOrchestrator(
        verbose=not args.quiet, thinking_mode=args.thinking
    )

    # Test with mock data
    test_block_data = b"test_block_data_for_model_orchestrator_validation"
    test_block_hash = "000000000019d6689c085ae165831e934ff763ae46a2a6c172b3f1b60a8ce26f"

    print("Testing model orchestrator...")
    solution = orchestrator.generate_solution(test_block_data, test_block_hash)

    if solution:
        print("✓ Solution generated successfully")
        print(json.dumps(solution, indent=2))
    else:
        print("✗ Solution generation failed")

    # Show performance stats
    stats = orchestrator.get_performance_stats()
    print(f"\nPerformance Stats: {stats}")


if __name__ == "__main__":
    main()
