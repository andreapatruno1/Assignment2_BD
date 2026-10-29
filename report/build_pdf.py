import markdown, pathlib, sys
from playwright.sync_api import sync_playwright
d = pathlib.Path(__file__).parent
md = (d/"report.md").read_text()
html = markdown.markdown(md, extensions=["tables", "fenced_code"])
page = f'<!doctype html><html><head><meta charset="utf-8"><style>{(d/"style.css").read_text()}</style></head><body>{html}</body></html>'
(d/"report.html").write_text(page)
with sync_playwright() as p:
    b = p.chromium.launch(); pg = b.new_page()
    pg.goto((d/"report.html").resolve().as_uri()); pg.wait_for_load_state("networkidle")
    pg.pdf(path=sys.argv[1], format="A4", margin=dict(top="1.6cm", bottom="1.6cm", left="1.8cm", right="1.8cm"), print_background=True)
    b.close()
