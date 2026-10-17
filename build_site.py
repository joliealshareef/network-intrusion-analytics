"""Injects results.json and mosaic.txt into the site template -> index.html"""
import json
from pathlib import Path
here = Path(__file__).parent
r = json.loads((here / "results.json").read_text())
html = (here / "site_template.html").read_text()
html = html.replace("__RESULTS__", json.dumps(r, separators=(",", ":")))
html = html.replace("__MOSAIC__", (here / "mosaic.txt").read_text())
(here / "index.html").write_text(html)
print("index.html", len(html), "bytes")
