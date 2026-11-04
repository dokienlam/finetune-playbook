"""Kiem tra moi cell code trong notebook co parse duoc khong. Chay TRUOC khi push Kaggle."""
import ast, json, sys
bad=0
for p in sys.argv[1:]:
    d=json.load(open(p)); n=0
    for i,c in enumerate(d["cells"]):
        if c["cell_type"]!="code": continue
        src=''.join(c["source"])
        if src.lstrip().startswith(("!","%")): continue   # magic/shell, khong phai Python
        n+=1
        try: ast.parse(src)
        except SyntaxError as e:
            print(f"  {p} cell {i}: {e}")
            for j,l in enumerate(src.splitlines()[max(0,(e.lineno or 1)-3):(e.lineno or 1)+1]):
                print(f"      | {l}")
            bad+=1
    print(f"{p}: {len(d['cells'])} cell ({n} cell Python) -> {'OK' if bad==0 else 'CO LOI'}")
sys.exit(1 if bad else 0)
