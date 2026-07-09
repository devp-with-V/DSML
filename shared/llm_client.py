import os
import requests
import json
import time
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class OpenRouterClient:
    """
    A wrapper for the OpenRouter API that dynamically fetches free models
    and provides fallback handling for rate limits (429s).
    """
    def __init__(self):
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable not set. Please set it in your .env file.")
        
        self.base_url = "https://openrouter.ai/api/v1"
        self.free_models = self._fetch_free_models()
        
        if not self.free_models:
            print("Warning: No free models found dynamically. Falling back to a hardcoded default.")
            self.free_models = ["meta-llama/llama-3-8b-instruct:free"]
            
    def _fetch_free_models(self):
        """Fetches the list of all available models and filters for completely free ones."""
        try:
            response = requests.get(f"{self.base_url}/models")
            response.raise_for_status()
            models_data = response.json().get("data", [])
            
            free_models = []
            for model in models_data:
                pricing = model.get("pricing", {})
                # Both prompt and completion must cost "0"
                if pricing.get("prompt") == "0" and pricing.get("completion") == "0":
                    free_models.append(model["id"])
            return free_models
        except Exception as e:
            print(f"Error fetching models: {e}")
            return []

    def generate(self, prompt, system_prompt="You are a helpful data science assistant.", json_mode=False):
        """
        Sends a generation request, trying free models sequentially until one succeeds.
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        for model_id in self.free_models:
            data = {
                "model": model_id,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ]
            }
            
            # Some OpenRouter models support forced JSON mode
            if json_mode:
                # Appending instructions to system prompt just in case the model doesn't natively support response_format
                data["messages"][0]["content"] += " You must respond in valid JSON format."
                # We can also add response_format, though not all free models support it perfectly
                data["response_format"] = {"type": "json_object"}
                
            try:
                response = requests.post(f"{self.base_url}/chat/completions", headers=headers, json=data)
                
                if response.status_code == 429:
                    print(f"Rate limited (429) on model {model_id}. Trying next free model...")
                    time.sleep(1)
                    continue
                
                response.raise_for_status()
                result = response.json()
                return result["choices"][0]["message"]["content"]
                
            except requests.exceptions.RequestException as e:
                print(f"Error with model {model_id}: {e}")
                continue
                
        raise Exception("All free models failed or were rate limited.")

if __name__ == '__main__':
    # Simple test
    client = OpenRouterClient()
    print(f"Found {len(client.free_models)} free models.")
    print("Testing a generation...")
    try:
        response = client.generate("Say 'Hello, World!'")
        print("Response:", response)
    except Exception as e:
        print("Generation failed:", e)
