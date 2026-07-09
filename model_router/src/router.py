import json
from shared.registry import list_methods
from shared.llm_client import OpenRouterClient

# Pre-load implemented methods to register them
from shared.methods.classification import random_forest
from shared.methods.regression import linear
from shared.methods.clustering import kmeans

def recommend_method(dataset_description, profile):
    """
    Calls the LLM to recommend a method based on the dataset profile and available methods.
    """
    client = OpenRouterClient()
    available_methods = list_methods()
    
    system_prompt = (
        "You are an expert Data Scientist router. Your job is to select the most appropriate "
        "machine learning method from a fixed toolbox based on the dataset profile."
    )
    
    prompt = f"""
    Dataset Description: {dataset_description}
    
    Dataset Profile:
    {json.dumps(profile, indent=2)}
    
    Available Methods in Toolbox:
    {available_methods}
    
    Recommend ONE method from the available toolbox. You must return your response as a valid JSON object with the following structure:
    {{
        "problem_type": "classification|regression|clustering",
        "recommended_method": "<exact_method_name_from_toolbox>",
        "reasoning": "Why this method was chosen...",
        "preprocessing_notes": "Any scaling or encoding needed..."
    }}
    """
    
    response_text = client.generate(prompt, system_prompt=system_prompt, json_mode=True)
    
    # Try to parse the JSON output from the LLM
    try:
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()
            
        recommendation = json.loads(response_text)
        
        # Validate against registry
        if recommendation.get("recommended_method") not in available_methods:
            print(f"Warning: LLM recommended '{recommendation.get('recommended_method')}' which is not in the registry. Falling back.")
            recommendation["recommended_method"] = available_methods[0]
            
        return recommendation
    except Exception as e:
        print(f"Error parsing LLM output: {e}\nRaw Output: {response_text}")
        raise
