#!/usr/bin/env python3
"""Generate the branch-root GitHub Pages entry from the editable docs/index.html."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
source = (ROOT / 'docs/index.html').read_text()
page = source.replace('./assets/', './docs/assets/')
page = page.replace('<!doctype html>', '<!doctype html>\n<!-- Generated from docs/index.html by scripts/sync_pages.py. -->', 1)
(ROOT / 'index.html').write_text(page)
print('Updated index.html for GitHub Pages at /lambda_reach/.')
