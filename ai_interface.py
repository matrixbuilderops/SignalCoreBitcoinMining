import subprocess
import shlex

def call_ai_model(input_data: bytes) -> str:
    cmd = (
        'echo "{0}" | '
        'ollama run mixtral:8x7b-instruct-v0.1-q6_K --stdin'
    ).format(input_data.hex())
    result = subprocess.run(
        shlex.split(cmd),
        capture_output=True,
        text=True
    )
    return result.stdout.strip()

