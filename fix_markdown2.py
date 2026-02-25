import json

def is_heading(line):
    return line.strip().startswith('#')

def is_formula_line(line):
    s = line.strip()
    if s.startswith('$$') or s.startswith('$') or s.endswith('$$') or s.endswith('$'):
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
        inside_math = False
        for i, line in enumerate(source):
            nl = '\n' if line.endswith('\n') else ''
            content = line[:-1] if nl else line
            if content.strip().startswith('$$'):
                inside_math = not inside_math
            if not content.strip():
                new_lines.append(content + nl)
                continue
            if inside_math or is_heading(content) or is_formula_line(content):
                new_lines.append(content + nl)
                continue
            next_content = ''
            if i+1 < len(source):
                ln = source[i+1]
                next_content = ln[:-1] if ln.endswith('\n') else ln
            if inside_math or is_heading(next_content) or is_formula_line(next_content):
                new_lines.append(content + nl)
                continue
            if not content.rstrip().endswith('<br>'):
                content = content + '<br>'
                changed = True
            new_lines.append(content + nl)
        if changed:
            cell['source'] = new_lines

if changed:
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print('Notebook updated with <br> tags (refined)')
else:
    print('No changes needed')
