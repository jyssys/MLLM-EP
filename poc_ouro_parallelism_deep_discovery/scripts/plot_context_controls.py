"""Render editable SVG plots from the closed constant-output context controls."""
import argparse
import html
from pathlib import Path

from poc_ouro_parallelism_deep_discovery.contract import ROOT, read, sha, write


def validate_cells(cells, size):
    selected = [row for row in cells if row['size'] == size]
    expected = {(tp, context) for tp in [1, 2, 4] for context in [1024, 8192, 16384, 32768]}
    actual = [(row['TP'], row['context']) for row in selected]
    if len(actual) != 12 or set(actual) != expected:
        raise ValueError('Require twelve unique context cells for this model.')
    if any(row['timing_trials'] != 6 or row['frozen_prompt_patterns'] != 2 or
           row['median_wall_ms'] <= 0 for row in selected):
        raise ValueError('Context-control denominators or wall times changed.')
    return selected


def render(cells, size):
    selected = validate_cells(cells, size)
    values = {(r['TP'], r['context']): r['median_wall_ms'] for r in selected}
    contexts = [1024, 8192, 16384, 32768]
    maximum = max(values.values())*1.12
    left, top, width, height = 94, 72, 670, 325
    x = lambda index: left+index*width/3
    y = lambda value: top+height*(1-value/maximum)
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="840" height="565" viewBox="0 0 840 565">',
             '<rect width="840" height="565" fill="white"/>',
             '<g font-family="sans-serif" font-size="14" fill="#222">',
             f'<text x="42" y="29" font-size="20">Base Ouro-{html.escape(size)}: native fixed-R4 controls</text>',
             '<text x="42" y="51">Generate-API wall, 128 controlled outputs, not natural reasoning latency</text>']
    for tick in range(6):
        value = maximum*tick/5
        py = y(value)
        parts.extend([f'<line x1="{left}" y1="{py:.2f}" x2="{left+width}" y2="{py:.2f}" stroke="#ddd"/>',
                      f'<text x="{left-10}" y="{py+5:.2f}" text-anchor="end">{value:.0f}</text>'])
    parts.append('<text x="22" y="255" transform="rotate(-90 22 255)">Wall time (ms)</text>')
    for index, context in enumerate(contexts):
        parts.append(f'<text x="{x(index):.2f}" y="425" text-anchor="middle">{context}</text>')
    parts.append('<text x="425" y="453" text-anchor="middle">Prompt tokens (categorical positions)</text>')
    for tp, color in [(1, '#2364aa'), (2, '#c05213'), (4, '#237d43')]:
        points = ' '.join(f'{x(i):.2f},{y(values[tp, c]):.2f}' for i, c in enumerate(contexts))
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        for index, context in enumerate(contexts):
            value = values[tp, context]
            parts.append(f'<circle cx="{x(index):.2f}" cy="{y(value):.2f}" r="4" fill="{color}">'
                         f'<title>TP{tp}, N{context}: {value:.6f} ms</title></circle>')
        lx = 170+(tp.bit_length()-1)*210
        parts.append(f'<text x="{lx}" y="486" fill="{color}">TP{tp}</text>')
    parts.extend(['<text x="42" y="519">Each point: six trials across two frozen prompt patterns. No population CI.</text>',
                  '<text x="42" y="543">Separate protocol: 52 GiB aggregate KV per TP group. No candidate result.</text>',
                  '</g></svg>'])
    return '\n'.join(parts)+'\n'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    target = ROOT/args.output
    if target.exists():
        raise FileExistsError(target)
    source = ROOT/'analysis/long_context_complete_v2/summary.json'
    data = read(source)
    for relative, digest in data['input_sha256'].items():
        if sha(ROOT/relative) != digest:
            raise ValueError('Changed admitted context input: '+relative)
    target.mkdir(parents=True)
    for size in ['1.4B', '2.6B']:
        (target/f'base{size.replace(".", "").replace("B", "")}.svg').write_text(render(data['cells'], size))
    write(target/'manifest.json', dict(source=str(source.relative_to(ROOT)), source_sha256=sha(source),
        renderer_sha256=sha(Path(__file__)),
        evidence_class='MEASURED_CONVENTIONAL_CONTROL',
        scope='Editable SVG of admitted median API-wall times. Categorical context axis. No candidate speedup or new statistics.',
        graphical_validation='XML structure and source-cell tests. No screenshot or browser rendering check.'))


if __name__ == '__main__':
    main()
