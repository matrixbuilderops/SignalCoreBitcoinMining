import subprocess
import json
from typing import Any, Dict, Optional


class ModelInterface:
    def __init__(self, model_command: Optional[str] = None) -> None:
        # Allow override, but default to the provided model call
        self.model_command = model_command or "ollama run mixtral:8x7b-instruct-v0.1-q6_K"

    def query_model(self, prompt: str) -> Dict[str, Any]:
        try:
            process = subprocess.Popen(
                self.model_command.split(),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            stdout, stderr = process.communicate(input=prompt)

            if stderr:
                return {"error": stderr.strip()}

            try:
                return json.loads(stdout)
            except json.JSONDecodeError:
                return {"raw_output": stdout.strip()}

        except Exception as e:
            return {"error": str(e)}


if __name__ == "__main__":
    interface = ModelInterface()
    prompt = "Calculate optimal nonce using Knuth(10, 3, 16000) and verify ForkIntegrity."
    result = interface.query_model(prompt)
    print(json.dumps(result, indent=2))
