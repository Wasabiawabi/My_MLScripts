import json

def is_heading_or_formula(line):
    s = line.strip()
    if not s:
        return True
    if s.startswith('#'):
        return True
    if s.startswith('$$') or s.startswith('$'):
        return True
    if s.startswith('```'):
        return True
    return False

notebook_path = r"c:\Users\suzuk\Documents\A-Workspace\ML-learn\My_MLScripts\Elementary_classification\Note.ipynb"
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

changed = False
for cell in nb.get('cells', []):
    if cell.get('cell_type') == 'markdown':
        source = cell.get('source', [])
        new_lines = []
        for i, line in enumerate(source):
            nl = '\n' if line.endswith('\n') else ''
            content = line[:-1] if nl else line
            if content.strip() and not is_heading_or_formula(content):
                next_content = ''
                if i+1 < len(source):
                    ln = source[i+1]
                    next_content = ln[:-1] if ln.endswith('\n') else ln
                if not is_heading_or_formula(next_content):
                    if not content.rstrip().endswith('<br>'):
                        content = content + '<br>'
                        changed = True
            new_lines.append(content + nl)
        if changed:
            cell['source'] = new_lines

if changed:
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print('Notebook updated with <br> tags')
else:
    print('No changes needed')
