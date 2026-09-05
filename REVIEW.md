# Paper and media review

## Paper

Reviewed the 19-page arXiv manuscript, including the formulation, main experiments, limitations, proofs, and implementation appendices.

The website centers on policy evaluation: estimating the worst-over-future safety signal for a **fixed control policy**. It does not describe λ-Reachability as a new robot controller or claim neural-network safety certification.

The method samples a geometric rollout horizon, computes a max-over-segment target, and includes the terminal value with probability δⁿ. For the ideal operator, δ < 1 induces contraction, while λ → 1 recovers the undiscounted value. Practical finite-horizon truncation and noisy hardware observations remain limitations.

Values on the page were checked against Tables 1–2. The temporal recall comparison includes all three simulation tasks and both hardware tasks, with mean and standard deviation. Featured values use λ = 0.99. Temporal recall is labeled as a warning-horizon metric, not classification accuracy. The page explains that lower λ settings can perform better on some hardware balance metrics. Author order and affiliations follow page 1.

## Website plan implemented

1. Hero: complete title, CoRL 2026 acceptance, authors, CMU/ICL branding, paper/code links, and a hardware clip.
2. Research overview: full authored video with four chapter buttons and visual descriptions.
3. Hardware gallery: four outcome cards, eight selectable clips, and balance/collision filters.
4. Method: safety-value definition, three-step learning explanation, and an interactive geometric-horizon illustration.
5. Results: precisely labeled metrics, original training curves, and a simulation/hardware comparison table.
6. Resources: manuscript, repository, copyable arXiv BibTeX, and reference-site attribution.

The reference's cream background, dark blue-green text, rust accent, serif headings, and rounded media panels are retained. The implementation uses plain HTML/CSS/JavaScript with responsive layouts and keyboard-accessible controls.

## Source inventory and privacy edits

Reviewed the compiled 179.35-second video, all ten raw camera clips, and all eight plot clips using metadata and contact sheets. Some camera/plot pairs have different durations, so independent raw plots were not assumed to be synchronized. The authored research video supplies the existing synchronized comparisons. Raw gallery clips retain supplied slow-motion timing.

| Website output | Source | Privacy treatment |
| --- | --- | --- |
| `push-safe-1.mp4` | `push-safe 1/vid.mov` | Overhead camera; no operator face in reviewed frames. |
| `push-unsafe-1.mp4` | `push-unsafe 1/vid.mov` | Overhead camera; no operator face in reviewed frames. |
| `push-safe-2.mp4` | `push-safe 2/vid.mov` | Full-duration 6×5-block mosaic over x=1280, y=0, w=400, h=300 in the original 1920×1080 frame. |
| `push-unsafe-2.mp4` | `push-unsafe 2/vid.mov` | Full-duration 6×5-block mosaic over x=1120, y=0, w=440, h=320 in the original frame. |
| `avoid-safe-1.mp4` | `avoid-safe 1/right vid.mov` | Alternate camera excludes the operator's face. |
| `avoid-unsafe-1.mp4` | `avoid-unsafe 1/right vid.mov` | Alternate camera excludes the operator's face; black tail trimmed. |
| `avoid-safe-2.mp4` | `avoid-safe 2/vid.mov` | Crop x=340, y=200, w=1560, h=878 excludes the operator at left. |
| `avoid-unsafe-2.mp4` | `avoid-unsafe 2/vid.mov` | Camera view excludes the operator's face. |
| `overview.mp4` | `final_1.MOV` | Existing crops/masks retained. Expanded opaque head mask x=265, y=148, w=56, h=45 from 39.5–49 s. |

All masking and cropping are burned into the exported video pixels. Poster images are extracted afterward. Outputs are H.264, 1280×720, 30 fps, yuv420p, with fast-start metadata and stripped source metadata. Ambient audio is removed from gallery clips; the authored overview audio is retained. The nine exported videos total approximately 17.3 MB.

Original and exported contact sheets were visually reviewed. Mosaic regions, the crop, and the overview mask were checked at 0.25-second intervals, including head movement and scene transitions. This is a sampled visual review, not a claim that an automated detector certified every frame. Review images and the downloaded paper remain in the local, ignored `.work/` directory and are not part of the served site.

## Verification

- Chromium desktop/mobile layouts checked at 320, 390, 768, and 1440 px: no horizontal page overflow.
- All eight gallery variants loaded, decoded, and advanced during playback.
- Balance/collision filtering, trial switching, video chapters, λ slider, citation clipboard copy, and expandable limitations exercised.
- Slider checked at λ = 0, 0.5, 0.95, and 0.99 (mean horizons 1, 2, 20, 100).
- No JavaScript exceptions or HTTP asset errors during the browser interaction check.
- Preview uses a byte-range-capable server because Python's basic `http.server` can clamp a chapter seek to the start before the media is buffered.
- Hosting layout: a separate `gh-pages` branch, with `docs/` selected as the GitHub Pages source. Research code remains on `main`.

## Website refinements

The hero video panel is aligned with the page; institutional logos are larger.
The three method expressions use native MathML. The horizon chart uses equal
five-step bins up to 100 steps and shows the remaining probability separately:
at λ = 0.99, P(n ≥ 101) = λ¹⁰⁰ ≈ 36.6%. This avoids comparing a single combined
tail against finite-width bins. Revised layouts were checked from 320–1440 px.
