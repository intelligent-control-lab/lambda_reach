# λ-Reachability project website

A static project page for **λ-Reachability: Geometric-Horizon Safety Bellman Equations for Humanoid Safety**, accepted to **CoRL 2026**. The page follows the warm palette, serif typography, section navigation, and research presentation of the lab's Safe-Stoppability website.

## View locally

No npm install or build step is needed. Python 3 is enough:

```bash
cd /home/yifan/Documents/lambda_reachibility
python3 scripts/serve.py
```

Open **http://127.0.0.1:8000**. Stop the server with Ctrl+C. If that port is in use, run `python3 scripts/serve.py --port 8001` and open port 8001 instead. This server supports HTTP byte ranges, so video chapter buttons and seeking work correctly.

## Files

- `docs/index.html`: editable page content, safety value explanation, method, project links, authors, and BibTeX.
- `index.html`: generated entry point for the public project URL.
- `scripts/sync_pages.py`: synchronizes the generated entry with `docs/index.html`.
- `docs/assets/css/site.css`: responsive visual design.
- `docs/assets/js/site.js`: experiment filters, paired video/curve trial selection, chapter navigation, horizon illustration, citation copying.
- `docs/assets/videos/`: eight prepared hardware clips plus the full research video.
- `docs/assets/posters/`: thumbnails extracted only from the prepared videos.
- `docs/assets/curves/`: eight complete recorded safety-value plots, paired with the hardware trials.
- `scripts/prepare_media.py`: reproducible FFmpeg privacy edits and web encodes.
- `scripts/prepare_curves.py`: extracts complete plot frames from the supplied curve clips with FFmpeg.
- `scripts/serve.py`: local preview server, restricted to the `docs` directory.
- `media-manifest.json`: source-to-output mapping and exact redaction parameters.
- `curve-manifest.json`: plot source files and extraction timestamps.
- `REVIEW.md`: paper review, website plan, media decisions, and verification notes.

## Media preparation

The original `final_1.MOV` and `web-material/` files are unchanged. Only prepared derivatives are loaded by the site. Recreate them with FFmpeg installed:

```bash
python3 scripts/prepare_media.py
python3 scripts/prepare_curves.py
```

Frontal push videos have a deliberately coarse mosaic over the operator's complete head path. Collision trials use an alternate camera angle or a crop that excludes the operator's face. The full research video retains its existing framing and receives an enlarged opaque mask for its distant operator. Gallery audio is removed; the authored overview audio is preserved. Slow-motion timing is retained except for small black-frame trims.

## GitHub Pages hosting

This is the **`gh-pages` website branch** of
[`intelligent-control-lab/lambda_reach`](https://github.com/intelligent-control-lab/lambda_reach).
It has independent history and contains only the website and its supporting files.
The `main` branch contains the research code. Do not merge this branch into `main`.
The layout follows [`spark`'s `gh-pages` branch](https://github.com/intelligent-control-lab/spark/tree/gh-pages), which keeps its site in `docs/`. This repository also has a generated root entry so visitors can open the project URL directly.

Publishing source in [repository Settings → Pages](https://github.com/intelligent-control-lab/lambda_reach/settings/pages):

- Source: **Deploy from a branch**
- Branch: **gh-pages**
- Folder: **/(root)**

GitHub enabled Pages automatically when this branch was pushed. Every push to
`gh-pages` republishes the site at:
**https://intelligent-control-lab.github.io/lambda_reach/**
No custom domain, build dependencies, or custom Actions workflow is required.
The root `.nojekyll` tells Pages to serve the prepared static files.
`docs/` remains the editable source, and `scripts/sync_pages.py` generates
`index.html` with the correct asset paths for the public URL.

All website work can be done from this directory:

```bash
cd /home/yifan/Documents/lambda_reachibility
git switch gh-pages
# Edit the website, then preview it with python3 scripts/serve.py.
python3 scripts/sync_pages.py
git add index.html docs README.md REVIEW.md scripts media-manifest.json curve-manifest.json
git commit -m "Update project website"
git push origin gh-pages
```

The original videos, `web-material/`, and `.work/` are ignored and must stay local.
Publish only prepared website assets, including privacy-edited MP4s, their derived
posters, and the recorded value plots.
The research repository's license and notice files are preserved on this branch.

## Content sources

- Paper: https://arxiv.org/pdf/2606.16022 (v1, June 14, 2026).
- Code and citation: https://github.com/intelligent-control-lab/lambda_reach.
- CoRL 2026 acceptance: supplied by the project owner.
- Visual reference: https://intelligent-control-lab.github.io/humanoid_stoppability/ and its local source.
- Figures: extracted from the paper; institutional logos reused from the reference project.

The page uses the project owner's CoRL 2026 `@inproceedings` BibTeX record,
omitting `index` and `abstract` and retaining the paper, website, YouTube, and code
links. The page centers on safety value functions,
the geometric-horizon learning idea, and hardware demonstrations; quantitative
evaluation remains in the paper and authored research video.

Each hardware trial appears beside its complete recorded value plot. Selecting
a trial switches both the video and plot; selecting a plot opens the full-size
image. Plots retain their source time axes and show the full recording, without
claiming synchronization with playback of the separate slow-motion camera clips.
