import json

def is_heading(line):
    return line.strip().startswith('#')

def is_formula_line(line):
    s = line.strip()
    if s.startswith('$$') or s.startswith('$') or s.endswith('$$') or s.endswith('$'):
        return True
    return False

notebook_path = r"c:\Users\suzuk\Documents\A-Workspace\ML-learn\My_MLScripts\Advanced_classification\Note.ipynb"
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)
changed = False
for cell in nb.get('cells', []):
    if cell.get('cell_type') == 'markdown':
        new_source = []
        inside_math = False
        for i, line in enumerate(cell.get('source', [])):
            nl = '\n' if line.endswith('\n') else ''
            content = line[:-1] if nl else line
            if content.strip().startswith('$$'):
                inside_math = not inside_math
            if inside_math or is_heading(content) or is_formula_line(content):
                if content.rstrip().endswith('<br>'):
                    content = content.rstrip()[:-4]
                    changed = True
                new_source.append(content + nl)
                continue
            next_content = ''
            if i+1 < len(cell.get('source', [])):
                ln = cell.get('source', [])[i+1]
                next_content = ln[:-1] if ln.endswith('\n') else ln
            if is_heading(next_content) or is_formula_line(next_content):
                if content.rstrip().endswith('<br>'):
                    content = content.rstrip()[:-4]
                    changed = True
                new_source.append(content + nl)
                continue
            if content.strip() and not content.rstrip().endswith('<br>'):
                content = content + '<br>'
                changed = True
            new_source.append(content + nl)
        if changed:
            cell['source'] = new_source

if changed:
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print('Cleaned and updated notebook')
else:
    print('No changes needed after cleaning')
