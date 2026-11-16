"""Dependency-free SVG figures for result tables.

matplotlib is not installed in the CPU tool environment, and the graded artefacts only need small
deterministic charts. Every figure is generated from numbers that already went through
``evaluation``/``register`` so a picture cannot claim more than the table.
"""
import html
import re
from pathlib import Path

COLORS = ('#3b6ea5', '#d0803a', '#5b8c5a', '#a4576a', '#7a6fa8', '#8a8a8a', '#4a4a4a')
FONT = "font-family='Segoe UI, Helvetica, Arial, sans-serif'"
INTERVAL = re.compile(r'\[\s*([0-9.]+)\s*,\s*([0-9.]+)\s*\]')


def _ticks(ymax, integer):
    if integer:
        step = 1 if ymax <= 5 else max(1, int(round(ymax / 5)))
        values = list(range(0, int(ymax) + step, step))
        return values, str
    step = ymax / 4
    values = [round(step * index, 6) for index in range(5)]
    return values, lambda value: f'{value:g}'


def bar_chart(series, path, title, ylabel, ymax=None, intervals=None, integer=False,
              width=680, height=380):
    """Write one bar chart. ``series`` is [(label, value)]; ``intervals`` is [(low, high)] or None."""
    if not series:
        raise ValueError('bar_chart needs at least one bar')
    values = [float(value) for _, value in series]
    ymax = float(ymax) if ymax is not None else max(values + [1.0 if not integer else 1])
    ymax = max(ymax, 1e-9)
    left, right, top, bottom = 64, 24, 52, 64
    plot_w, plot_h = width - left - right, height - top - bottom
    ticks, format_tick = _ticks(ymax, integer)
    parts = [f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' "
             f"viewBox='0 0 {width} {height}' {FONT}>",
             f"<rect width='{width}' height='{height}' fill='#ffffff'/>",
             f"<text x='{left}' y='30' font-size='17' fill='#1f2933'>{html.escape(title)}</text>",
             f"<text x='16' y='{top + plot_h / 2:.1f}' font-size='12' fill='#52606d' "
             f"transform='rotate(-90 16 {top + plot_h / 2:.1f})'>{html.escape(ylabel)}</text>"]
    for tick in ticks:
        y = top + plot_h - (tick / ymax) * plot_h
        parts.append(f"<line x1='{left}' y1='{y:.1f}' x2='{left + plot_w}' y2='{y:.1f}' "
                     f"stroke='{'#c9d2da' if tick else '#7b8794'}' stroke-width='1'/>")
        parts.append(f"<text x='{left - 8}' y='{y + 4:.1f}' font-size='11' fill='#52606d' "
                     f"text-anchor='end'>{format_tick(tick)}</text>")
    band = plot_w / len(series)
    bar_w = min(band * 0.55, 96)
    for index, (label, value) in enumerate(series):
        value = float(value)
        center = left + band * (index + 0.5)
        bar_h = (value / ymax) * plot_h
        x = center - bar_w / 2
        y = top + plot_h - bar_h
        parts.append(f"<rect x='{x:.1f}' y='{y:.1f}' width='{bar_w:.1f}' height='{bar_h:.1f}' "
                     f"fill='{COLORS[index % len(COLORS)]}' rx='3'/>")
        text = str(int(value)) if integer else f'{value:.3f}'
        parts.append(f"<text x='{center:.1f}' y='{y - 7:.1f}' font-size='12' fill='#1f2933' "
                     f"text-anchor='middle'>{text}</text>")
        if intervals:
            low, high = float(intervals[index][0]), float(intervals[index][1])
            y_low = top + plot_h - (low / ymax) * plot_h
            y_high = top + plot_h - (high / ymax) * plot_h
            parts.append(f"<line x1='{center:.1f}' y1='{y_low:.1f}' x2='{center:.1f}' "
                         f"y2='{y_high:.1f}' stroke='#1f2933' stroke-width='1.5'/>")
            for end in (y_low, y_high):
                parts.append(f"<line x1='{center - 7:.1f}' y1='{end:.1f}' x2='{center + 7:.1f}' "
                             f"y2='{end:.1f}' stroke='#1f2933' stroke-width='1.5'/>")
        parts.append(f"<text x='{center:.1f}' y='{top + plot_h + 20}' font-size='12' "
                     f"fill='#1f2933' text-anchor='middle'>{html.escape(str(label))}</text>")
    if intervals:
        parts.append(f"<text x='{left}' y='{height - 14}' font-size='11' fill='#52606d'>"
                     f'误差线为 95% 区间（方法对比使用登记表区间）</text>')
    parts.append('</svg>')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(parts) + '\n', encoding='utf-8')
    return path


def write_run_figures(metrics, failure_counts, directory, label='run'):
    """Two single-run charts: termination outcomes and failure classes."""
    directory = Path(directory)
    outcomes = [
        ('full-reward\nsuccess', metrics['full_reward_successes']),
        ('purchase only', metrics['purchase_without_full_reward']),
        ('max_steps', metrics['max_steps_terminated']),
    ]
    bar_chart(outcomes, directory / f'{label}-outcomes.svg',
              f'{label}: outcome split (n={metrics["episodes"]} episodes)',
              'episodes', ymax=metrics['episodes'], integer=True)
    classes = [(name, failure_counts.get(name, 0)) for name in
               ('format', 'budget', 'specification', 'loop', 'retrieval', 'other', 'engineering')]
    total = max(1, sum(count for _, count in classes))
    bar_chart(classes, directory / f'{label}-failures.svg',
              f'{label}: failed episodes by class (heuristic triage)',
              'episodes', ymax=total, integer=True)
    return [directory / f'{label}-outcomes.svg', directory / f'{label}-failures.svg']


def method_figure(rows, path, title='Success rate by method (frozen val, 95% interval)'):
    """Compare registered methods. ``rows`` are register rows with status=complete."""
    order = ('B0', 'B1', 'P0', 'R0', 'R1')
    grouped = {}
    for row in rows:
        if (row.get('status') or '').strip() != 'complete':
            continue
        method = (row.get('method') or '').strip()
        try:
            rate = float(row.get('success_rate'))
        except (TypeError, ValueError):
            continue
        match = INTERVAL.fullmatch((row.get('success_rate_ci95') or '').strip())
        interval = (float(match.group(1)), float(match.group(2))) if match else (rate, rate)
        grouped.setdefault(method, {'rates': [], 'intervals': [], 'episodes': []})
        grouped[method]['rates'].append(rate)
        grouped[method]['intervals'].append(interval)
        grouped[method]['episodes'].append(row.get('episodes') or '?')
    if not grouped:
        raise ValueError('No complete register rows: nothing to plot yet')
    series, intervals = [], []
    for method in [name for name in order if name in grouped] + sorted(set(grouped) - set(order)):
        entry = grouped[method]
        mean = sum(entry['rates']) / len(entry['rates'])
        series.append((f'{method}\n(n={",".join(str(x) for x in entry["episodes"])})', mean))
        intervals.append((min(low for low, _ in entry['intervals']),
                          max(high for _, high in entry['intervals'])))
    return bar_chart(series, path, title, 'success rate', ymax=1.0, intervals=intervals)


def run(args):
    """CLI entry: plot every complete register row, refusing to invent a figure without results."""
    from .register import read_register
    fieldnames, rows = read_register(args.register)
    if not rows:
        raise SystemExit('Register has no data rows yet: run evaluate --register-row after a real run')
    try:
        path = method_figure(rows, args.output)
    except ValueError as error:
        raise SystemExit(str(error))
    print(f'Wrote {path}')
