import json
import os
from shared.llm_client import OpenRouterClient

def generate_model_report(recommendation, results, output_path="reports/model_report.html"):
    client = OpenRouterClient()
    
    # Process importances if available
    top_importances = "None"
    if results.get("importances"):
        sorted_imp = sorted(results["importances"].items(), key=lambda x: abs(x[1]), reverse=True)[:10]
        top_importances = json.dumps(dict(sorted_imp), indent=2)
        
    metrics_str = json.dumps(results.get("metrics", {}), indent=2)
    
    prompt = f"""
    You are an elite Data Scientist and an award-winning Frontend Developer.
    I have executed a Machine Learning pipeline based on your AI routing recommendation.
    
    ROUTING CONTEXT:
    Problem Type: {recommendation.get('problem_type')}
    Selected Method: {recommendation.get('recommended_method')}
    Reasoning: {recommendation.get('reasoning')}
    
    EXECUTION RESULTS:
    Metrics: {metrics_str}
    Top Feature Importances: {top_importances}
    
    TASK:
    Generate a standalone, ultra-premium HTML report summarizing this execution.
    
    DESIGN REQUIREMENTS:
    - Use a stunning modern aesthetic (e.g., vibrant dark mode, glassmorphism UI elements, sleek gradients).
    - Use modern typography from Google Fonts (like 'Inter', 'Outfit', or 'Roboto').
    - Include subtle CSS micro-animations on hover or load (e.g., fade-in, scale up).
    - Organize the content into a beautiful CSS Grid or Flexbox layout.
    - Write a compelling executive summary interpreting the metrics in a business context.
    - Output ONLY the raw HTML string starting with <!DOCTYPE html> and ending with </html>. Do not wrap it in markdown code blocks.
    """
    
    print("Asking AI to generate an ultra-premium HTML report...")
    try:
        html_content = client.generate(prompt, system_prompt="You are an elite Data Scientist and Frontend Developer.")
        
        # Strip markdown wrapping if the LLM includes it
        if html_content.startswith('```html'):
            html_content = html_content[7:].strip()
        elif html_content.startswith('```'):
            html_content = html_content[3:].strip()
        if html_content.endswith('```'):
            html_content = html_content[:-3].strip()
            
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)
            
        print(f"Premium AI HTML Report successfully generated at {os.path.abspath(output_path)}")
    except Exception as e:
        print(f"Failed to generate AI HTML Report: {e}")
        # Fallback to a basic template if the LLM fails
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(f"<html><body><h1>Fallback Report</h1><pre>{metrics_str}</pre></body></html>")
