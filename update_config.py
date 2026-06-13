import json
import re

def update_config():
    with open("best_params.json", "r") as f:
        params = json.load(f)
        
    with open("config.py", "r") as f:
        content = f.read()
        
    for k, v in params.items():
        # Match lines like: VAR_NAME = float(os.getenv("VAR_NAME", "1.5"))
        # Or: VAR_NAME = int(os.getenv("VAR_NAME", "15"))
        
        # We need to replace the default value string inside the os.getenv call
        pattern = re.compile(rf'({k}\s*=\s*(?:float|int)\(\s*os\.getenv\("{k}",\s*")([^"]+)("\)\s*\))')
        
        # Format the new value
        if isinstance(v, float):
            new_val_str = f"{v:.4f}"
        else:
            new_val_str = str(int(v))
            
        # Replace it
        def replacer(match):
            return f'{match.group(1)}{new_val_str}{match.group(3)}'
            
        content = pattern.sub(replacer, content)
        
    with open("config.py", "w") as f:
        f.write(content)
        
    print("Successfully hardcoded all parameters into config.py!")

if __name__ == "__main__":
    update_config()