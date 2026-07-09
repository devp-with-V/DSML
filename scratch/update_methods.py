import os
import glob
import re

def update_methods():
    methods_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'shared', 'methods'))
    
    # Find all .py files in subdirectories
    py_files = glob.glob(os.path.join(methods_dir, '**', '*.py'), recursive=True)
    
    for filepath in py_files:
        if '__init__' in filepath:
            continue
            
        with open(filepath, 'r') as f:
            content = f.read()
            
        # We look for return {"model": ..., "metrics": ..., "importances": ...}
        # and replace it to include y_test and preds using locals().get() to avoid UnboundLocalError
        # if the method didn't declare them (like clustering).
        
        # Regex to match the return statement. 
        # Most of them are: return {"model": model, "metrics": metrics, "importances": importances}
        # or return {"model": model, "metrics": metrics, "importances": None}
        
        # Pattern: return \{"model": [^,]+, "metrics": [^,]+, "importances": [^\}]+\}
        pattern = r'(return\s*\{\s*"model"\s*:\s*[^,]+,\s*"metrics"\s*:\s*[^,]+,\s*"importances"\s*:\s*[^\}]+\})'
        
        def replacer(match):
            original = match.group(1)
            # Remove the closing brace
            original = original.strip()[:-1]
            return original + ', "y_test": locals().get("y_test"), "preds": locals().get("preds")}'
            
        new_content = re.sub(pattern, replacer, content)
        
        if new_content != content:
            with open(filepath, 'w') as f:
                f.write(new_content)
            print(f"Updated: {os.path.basename(filepath)}")
        else:
            print(f"No match/Already updated: {os.path.basename(filepath)}")

if __name__ == "__main__":
    update_methods()
