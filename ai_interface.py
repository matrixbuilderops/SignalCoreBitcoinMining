import subprocess
import json
from typing import Dict, Any, Optional


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
        result = subprocess.run(
            ['ollama', 'run', 'mixtral:8x7b-instruct-v0.1-q6_K'],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=30
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


def extract_recommendation(ai_response: str) -> str:
    """
    Extract mining recommendation from AI response
    
    Args:
        ai_response: Raw AI model response
        
    Returns:
        Extracted recommendation (PROCEED, HOLD, RETRY, or ERROR)
    """
    response_upper = ai_response.upper()
    
    if "PROCEED" in response_upper:
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

