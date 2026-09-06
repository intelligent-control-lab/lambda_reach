#!/usr/bin/env python3
"""Extract complete recorded value plots for the hardware gallery using FFmpeg."""
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'web-material/web-material'
OUT = ROOT / 'docs/assets/curves'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = []
    for task in ['push', 'avoid']:
        for outcome in ['safe', 'unsafe']:
            for trial in [1, 2]:
                name = f'{task}-{outcome}-{trial}'
                extension = 'mov' if task == 'push' else 'mp4'
                source = RAW / f'{task}-{outcome} {trial}' / f'curve.{extension}'
                info = json.loads(subprocess.check_output([
                    'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                    '-of', 'json', str(source),
                ]))
                # The last tenth-second contains the completed trace in these sources.
                timestamp = float(info['format']['duration']) - 0.1
                output = OUT / f'{name}.png'
                subprocess.run([
                    'ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
                    '-ss', f'{timestamp:.6f}', '-i', str(source),
                    '-frames:v', '1', '-map_metadata', '-1', str(output),
                ], check=True)
                manifest.append({
                    'output': str(output.relative_to(ROOT)),
                    'source': str(source.relative_to(ROOT)),
                    'timestamp': round(timestamp, 6),
                    'presentation': 'Complete recorded curve; no playback time alignment implied.',
                })
                print('Prepared', name, flush=True)
    (ROOT / 'curve-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
