#!/usr/bin/env python3
"""Build synchronized camera/curve replays from prepared, privacy-edited footage."""
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs/assets/monitoring'
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'

# Curve source time = prepared camera time + offset (seconds of slow-motion playback).
# No time stretching is applied. Collision offsets are visually calibrated estimates.
ALIGNMENTS = {
    'push-safe-1': (0, 'Matching original camera/curve timelines; no camera trim.'),
    'push-unsafe-1': (0, 'Matching original camera/curve timelines; no camera trim.'),
    'push-safe-2': (.15, 'Matching original timelines; prepared camera removes the first 0.15 s.'),
    'push-unsafe-2': (.15, 'Matching original timelines; prepared camera removes the first 0.15 s.'),
    'avoid-safe-1': (-.1, 'Estimated from camera-frame and curve-progress matches to overview 31–37 s.'),
    'avoid-unsafe-1': (.633333, 'Estimated from camera-frame and curve-progress matches to overview 42–47 s.'),
    'avoid-safe-2': (2, 'Estimated from ball approach and closest passage: camera approximately 5.5 s, curve approximately 7.5 s.'),
    'avoid-unsafe-2': (2, 'Estimated from ball approach and contact: camera approximately 3.6–4.0 s, curve approximately 5.6–6.0 s.'),
}


def run(args):
    subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', *args], check=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = []
    for name, (offset, evidence) in ALIGNMENTS.items():
        task, outcome, trial = name.split('-')
        camera = ROOT / f'docs/assets/videos/{name}.mp4'
        curve = ROOT / f'web-material/web-material/{task}-{outcome} {trial}/curve.{"mov" if task == "push" else "mp4"}'
        probe = json.loads(subprocess.check_output([
            'ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'json', str(camera),
        ]))
        duration = float(probe['format']['duration'])
        speed = '0.125x' if task == 'push' else '0.25x'
        label = f"drawtext=fontfile={FONT}:fontsize=20:fontcolor=0x172f37:x=18:y=7"
        shift = f'trim=start={offset},setpts=PTS-STARTPTS' if offset >= 0 else f'setpts=PTS-STARTPTS,tpad=start_mode=clone:start_duration={-offset}'
        filters = (
            f'[0:v]setpts=PTS-STARTPTS,fps=30,scale=960:540,setsar=1,'
            f'pad=960:576:0:36:white,{label}:text=On the real robot | {speed},split[cw][cs];'
            f'[1:v]{shift},tpad=stop_mode=clone:stop_duration={duration},fps=30,'
            f'scale=960:540,setsar=1,pad=960:576:0:36:white,'
            f'{label}:text=Safety value monitor,split[vw][vs];'
            '[cw][vw]hstack=inputs=2[wide];[cs][vs]vstack=inputs=2[stacked]'
        )
        args = ['-i', str(camera), '-i', str(curve), '-filter_complex_threads', '2', '-filter_complex', filters]
        outputs = []
        for layout in ['wide', 'stacked']:
            output = OUT / f'{name}-{layout}.mp4'
            args += ['-map', f'[{layout}]', '-t', str(duration), '-an', '-map_metadata', '-1',
                     '-c:v', 'libx264', '-crf', '22', '-preset', 'medium', '-threads', '3',
                     '-g', '30', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(output)]
            outputs.append(str(output.relative_to(ROOT)))
        run(args)
        for layout in ['wide', 'stacked']:
            run(['-i', str(OUT / f'{name}-{layout}.mp4'), '-frames:v', '1', '-q:v', '2',
                 str(OUT / f'{name}-{layout}.jpg')])
        manifest.append({
            'trial': name, 'camera_source': str(camera.relative_to(ROOT)),
            'curve_source': str(curve.relative_to(ROOT)), 'curve_offset_seconds': offset,
            'alignment_evidence': evidence, 'outputs': outputs, 'duration': duration,
            'endpoint_handling': 'Hold first/last curve frame only outside available curve recording.',
            'playback': 'Camera and curve share one encoded video timeline in both responsive layouts.',
            'privacy': 'Camera uses the existing privacy-edited derivative; curve recording contains only the plot.',
        })
        print('Prepared monitoring replay:', name, flush=True)
    (ROOT / 'monitoring-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
