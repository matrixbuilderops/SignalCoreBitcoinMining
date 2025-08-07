"""
AI interface module for Bitcoin mining system.

This module provides AI-powered decision making capabilities using Ollama
to analyze mathematical validation results and provide mining recommendations.
"""

import subprocess  # nosec B404
from typing import Dict, Any


def call_ai_model(validation_data: Dict[str, Any], block_hash: str = "") -> str:
    """
    Call the Ollama AI model with validation data and get response

    Args:
        validation_data: Dictionary containing validation results from math processing
        block_hash: Optional block hash for context

    Returns:
        AI model response string
    """
    # Prepare structured prompt for the AI model
    prompt = f"""You are a Bitcoin mining validation AI. Analyze the following data:

Block Hash: {block_hash}
Level: {validation_data.get('level', 'unknown')}
Validation Results:
- Pre-drift check: {validation_data.get('pre_drift', False)}
- Fork integrity: {validation_data.get('fork_integrity', False)}
- Recursion sync: {validation_data.get('recursion_sync', False)}
- Entropy parity: {validation_data.get('entropy_parity', False)}
- Sorrell value: {validation_data.get('sorrell', 0)}
- Fork cluster: {validation_data.get('fork_cluster', 0)}
- Over recursion: {validation_data.get('over_recursion', 0)}
- Bit load: {validation_data.get('bit_load', 0)}
- Sandboxes: {validation_data.get('sandboxes', 0)}
- Cycles: {validation_data.get('cycles', 0)}
- Post-drift check: {validation_data.get('post_drift', False)}
- Post recursion sync: {validation_data.get('post_recursion_sync', False)}
- Fork sync: {validation_data.get('fork_sync', False)}

Provide a brief mining recommendation (PROCEED, HOLD, or RETRY) and reasoning."""

    try:
        # Call Ollama with the structured prompt
        result = subprocess.run(  # nosec B603 B607
            ["ollama", "run", "mixtral:8x7b-instruct-v0.1-q6_K"],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode == 0:
            return result.stdout.strip()
        else:
            # Fallback response if AI call fails
            return f"AI_ERROR: {result.stderr.strip()}"

    except subprocess.TimeoutExpired:
        return "AI_TIMEOUT: Model response took too long"
    except subprocess.CalledProcessError as e:
        return f"AI_CALL_ERROR: {str(e)}"
    except Exception as e:
        return f"AI_UNKNOWN_ERROR: {str(e)}"


def fallback_math_decision(validation_data: Dict[str, Any]) -> str:
    """
    Make mining decision based purely on mathematical validation when AI is unavailable.
    
    Args:
        validation_data: Dictionary containing validation results from math processing
        
    Returns:
        Decision based on mathematical criteria (PROCEED, HOLD, RETRY)
    """
    # Count passed validation checks
    critical_checks = [
        validation_data.get('pre_drift', False),
        validation_data.get('fork_integrity', False),
        validation_data.get('recursion_sync', False),
        validation_data.get('entropy_parity', False),
        validation_data.get('post_drift', False),
        validation_data.get('fork_sync', False)
    ]
    
    passed_checks = sum(critical_checks)
    total_checks = len(critical_checks)
    
    # Level 16000 specific validation
    level = validation_data.get('level', 0)
    sorrell = validation_data.get('sorrell', 0)
    bit_load = validation_data.get('bit_load', 0)
    cycles = validation_data.get('cycles', 0)
    
    # Mathematical decision logic
    if level == 16000 and passed_checks >= (total_checks * 0.8):  # 80% pass rate
        if sorrell > 0 and bit_load == 1600000 and cycles == 161:
            return "FALLBACK_MATH_PROCEED: All Level 16000 criteria satisfied"
        else:
            return "FALLBACK_MATH_RETRY: Level 16000 parameters need adjustment"
    elif passed_checks >= (total_checks * 0.6):  # 60% pass rate
        return "FALLBACK_MATH_RETRY: Partial validation, worth retrying"
    else:
        return "FALLBACK_MATH_HOLD: Insufficient validation criteria met"


def extract_recommendation(ai_response: str) -> str:
    """
    Extract mining recommendation from AI response

    Args:
        ai_response: Raw AI model response

    Returns:
        Extracted recommendation (PROCEED, HOLD, RETRY, or ERROR)
    """
    response_upper = ai_response.upper()

    # Handle fallback math decisions
    if "FALLBACK_MATH_PROCEED" in response_upper:
        return "PROCEED"
    elif "FALLBACK_MATH_RETRY" in response_upper:
        return "RETRY"
    elif "FALLBACK_MATH_HOLD" in response_upper:
        return "HOLD"
    # Handle AI model responses
    elif "PROCEED" in response_upper:
        return "PROCEED"
    elif "HOLD" in response_upper:
        return "HOLD"
    elif "RETRY" in response_upper:
        return "RETRY"
    elif "ERROR" in response_upper or "AI_" in response_upper:
        return "ERROR"
    else:
        # Default to HOLD if unclear
        return "HOLD"


def get_ai_recommendation_with_fallback(validation_data: Dict[str, Any], block_hash: str = "", enable_ai: bool = True) -> str:
    """
    Get mining recommendation using AI or mathematical fallback.
    
    Args:
        validation_data: Validation results from math processing
        block_hash: Optional block hash for context
        enable_ai: Whether to attempt AI analysis first
        
    Returns:
        Mining recommendation with reasoning
    """
    if enable_ai:
        # Try AI first
        ai_response = call_ai_model(validation_data, block_hash)
        
        # If AI is working, use its response
        if not any(error in ai_response.upper() for error in ["AI_ERROR", "AI_TIMEOUT", "AI_CALL_ERROR", "AI_UNKNOWN_ERROR"]):
            return ai_response
    
    # Fallback to mathematical decision
    return fallback_math_decision(validation_data)
