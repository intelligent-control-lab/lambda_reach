#!/usr/bin/env python3
"""Build web derivatives from original footage. Requires FFmpeg; originals are read-only."""
from pathlib import Path
import json, subprocess
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs/assets'
RAW = ROOT / 'web-material/web-material'
for folder in ['videos', 'posters']:
    (OUT / folder).mkdir(parents=True, exist_ok=True)

def run(args):
    subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', *args], check=True)

def mosaic(x, y, w, h):
    # A fixed region covers the full head path, including profile turns.
    # Deliberately coarse: the head occupies only a few output mosaic blocks.
    return (f'[0:v]split[base][region];[region]crop={w}:{h}:{x}:{y},'
            f'scale=6:5:flags=area,scale={w}:{h}:flags=neighbor[pixels];'
            f'[base][pixels]overlay={x}:{y},scale=1280:720,setsar=1[v]')

jobs = [
    ('push-safe-1', 'push-safe 1/vid.mov', 0, 19.28, None, 'Reviewed overhead view; operator face outside frame.'),
    ('push-unsafe-1', 'push-unsafe 1/vid.mov', 0, 17.43, None, 'Reviewed overhead view; operator face outside frame.'),
    ('push-safe-2', 'push-safe 2/vid.mov', .15, 14.10, mosaic(1280, 0, 400, 300), 'Coarse full-duration mosaic: x=1280 y=0 w=400 h=300, before scaling.'),
    ('push-unsafe-2', 'push-unsafe 2/vid.mov', .15, 14.10, mosaic(1120, 0, 440, 320), 'Coarse full-duration mosaic: x=1120 y=0 w=440 h=320, before scaling.'),
    ('avoid-safe-1', 'avoid-safe 1/right vid.mov', 0, 10.3, None, 'Use alternate camera: operator face outside frame.'),
    ('avoid-unsafe-1', 'avoid-unsafe 1/right vid.mov', 0, 10.0, None, 'Use alternate camera: operator face outside frame. Trim black tail.'),
    ('avoid-safe-2', 'avoid-safe 2/vid.mov', 0, 11.3, '[0:v]crop=1560:878:340:200,scale=1280:720,setsar=1[v]', 'Crop removes operator at left; retains robot and incoming ball.'),
    ('avoid-unsafe-2', 'avoid-unsafe 2/vid.mov', 0, 8.7, None, 'Reviewed camera view; operator face outside frame.'),
]
manifest = []
for name, source, start, duration, filters, privacy in jobs:
    output = OUT / 'videos' / (name + '.mp4')
    filters = filters or '[0:v]scale=1280:720,setsar=1[v]'
    run(['-ss', str(start), '-i', str(RAW / source), '-t', str(duration),
         '-filter_complex', filters, '-map', '[v]', '-an', '-map_metadata', '-1',
         '-c:v', 'libx264', '-crf', '23', '-preset', 'medium', '-threads', '4',
         '-r', '30', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(output)])
    run(['-ss', '1.5', '-i', str(output), '-frames:v', '1', '-q:v', '3', str(OUT / 'posters' / (name + '.jpg'))])
    manifest.append(dict(output=str(output.relative_to(ROOT)), source=str((RAW/source).relative_to(ROOT)),
                         start=start, duration=duration, privacy=privacy, audio='removed'))
    print('Prepared', name, flush=True)
# The authored explainer already crops the frontal operator and masks the distant face.
# Expand the existing white face mask through the complete unsafe two-camera segment.
output = OUT / 'videos/overview.mp4'
run(['-i', str(ROOT/'final_1.MOV'), '-vf', "drawbox=x=265:y=148:w=56:h=45:color=white:t=fill:enable='between(t,39.5,49)'",
     '-map', '0:v:0', '-map', '0:a:0?', '-map_metadata', '-1', '-c:v', 'libx264', '-crf', '22',
     '-preset', 'medium', '-threads', '4', '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '128k',
     '-movflags', '+faststart', str(output)])
run(['-ss', '26', '-i', str(output), '-frames:v', '1', '-q:v', '2', str(OUT/'posters/overview.jpg')])
manifest.append(dict(output=str(output.relative_to(ROOT)),source='final_1.MOV', privacy='Original crops retained; expanded opaque face mask x=265 y=148 w=56 h=45 from 39.5 to 49 seconds.',audio='authored audio preserved'))
(ROOT/'media-manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
print('Prepared overview and manifest', flush=True)
