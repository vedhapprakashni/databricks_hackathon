import os
import json
import re

notebooks_dir = r"d:\vedhap\databricks hackathon\hacakthon notebooks w outputs"

for filename in os.listdir(notebooks_dir):
    if filename.endswith(".ipynb"):
        filepath = os.path.join(notebooks_dir, filename)
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        modified = False
        
        # Look through all cells
        for cell in data.get('cells', []):
            if cell.get('cell_type') == 'code':
                # source can be a list of strings
                if isinstance(cell.get('source'), list):
                    for i, line in enumerate(cell['source']):
                        # Regex to find gsk_ followed by alphanumeric characters
                        if 'gsk_' in line:
                            new_line = re.sub(r'gsk_[a-zA-Z0-9]+', 'YOUR_GROQ_API_KEY_HERE', line)
                            if new_line != line:
                                cell['source'][i] = new_line
                                modified = True
                elif isinstance(cell.get('source'), str):
                    if 'gsk_' in cell['source']:
                        new_source = re.sub(r'gsk_[a-zA-Z0-9]+', 'YOUR_GROQ_API_KEY_HERE', cell['source'])
                        if new_source != cell['source']:
                            cell['source'] = new_source
                            modified = True
                            
        if modified:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=1)
            print(f"Masked API key in {filename}")

print("Done masking API keys.")
