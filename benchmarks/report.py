"""Render the recorded Chronos-2 serving benchmark; never synthesize missing cells.

Usage: python -m benchmarks.report --results results --output report
Relative paths are resolved against the repository root. Matplotlib is the
only additional dependency, and is imported only when rendering the figures.
"""
import argparse
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import statistics
from tempfile import gettempdir
from urllib.parse import quote

try:
    from .common import summarize
except ImportError:  # direct script execution
    from common import summarize


HERE = Path(__file__).resolve().parents[1]
ENGINES = ('native', 'vllm')
LABELS = {'native': 'Native Chronos2Pipeline', 'vllm': 'Chronos-2 vLLM plugin'}
COLORS = {'native': '#23689b', 'vllm': '#d55e00'}


def read_json(path, warnings):
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (ValueError, OSError) as error:
        warnings.append(f'{path.name}: could not read JSON ({error}).')
        return {}


def read_jsonl(path, warnings):
    records = []
    if path.exists():
        for number, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError('record is not an object')
                records.append(value)
            except ValueError as error:
                warnings.append(f'{path.name}:{number}: ignored incomplete/invalid JSON ({error}).')
    return records


def cell_key(row):
    return row['engine'], int(row['batch_size']), int(row['context'])


def numeric(value):
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def load_results(results):
    warnings = []
    config = read_json(results / 'config.json', warnings)
    plugin = config.get('plugin_backend', 'chronos2')
    LABELS['vllm'] = f'{plugin} vLLM plugin'
    metadata = read_json(results / 'metadata.json', warnings)
    errors = read_jsonl(results / 'errors.jsonl', warnings)
    points = read_jsonl(results / 'points.jsonl', warnings)
    expected_blocks = int(config.get('blocks', 3))
    repetitions = int(config.get('repetitions', 50))
    if not config:
        warnings.append('config.json is unavailable; completion uses the planned 3 × 50 requests.')
    summaries = {}
    if (results / 'summary.csv').exists():
        with (results / 'summary.csv').open(newline='') as source:
            for original in csv.DictReader(source):
                try:
                    key = cell_key(original)
                except (KeyError, TypeError, ValueError):
                    warnings.append('Ignored a summary.csv row without a valid cell identifier.')
                    continue
                summaries[key] = {name: (value if name in ('engine', 'payload_sha256') else numeric(value))
                                  for name, value in original.items()}
    unique = {}
    for point in points:
        try:
            key = cell_key(point)
            samples = point['http_latencies_s']
            if not samples or any(numeric(x) is None or float(x) <= 0 for x in samples):
                raise ValueError('HTTP latencies must be finite and positive')
            block_key = (*key, point.get('block'))
        except (KeyError, TypeError, ValueError) as error:
            warnings.append(f'Ignored invalid points.jsonl record: {error}.')
            continue
        if block_key in unique:
            warnings.append(f'Duplicate cell/block {block_key}; using its last record only.')
        unique[block_key] = point
    grouped = {}
    for block_key, point in unique.items():
        grouped.setdefault(block_key[:3], []).append(point)
    cells = {}
    for key in set(summaries) | set(grouped):
        engine, batch, context = key
        group = grouped.get(key, [])
        if group:
            latencies = [float(x) for point in group for x in point['http_latencies_s']]
            internal = [float(x) for point in group for x in point.get('inprocess_latencies_s', [])]
            row = {'engine': engine, 'batch_size': batch, 'context': context,
                   **summarize(latencies, batch), 'blocks': len(group),
                   'native_inprocess_n': len(internal),
                   'native_inprocess_p50_ms': summarize(internal, batch)['p50_ms'] if internal else None}
            for field, location in (
                ('loaded_gpu_mib', ('loaded_gpu_bytes',)),
                ('warmed_gpu_mib', ('warmed_gpu_bytes',)),
                ('sampled_peak_gpu_mib', ('sampled_peak_gpu_bytes',)),
                ('peak_allocated_mib', ('memory_after', 'peak_allocated_bytes')),
                ('peak_reserved_mib', ('memory_after', 'peak_reserved_bytes')),
            ):
                values = []
                for point in group:
                    value = point
                    for part in location:
                        value = value.get(part) if isinstance(value, dict) else None
                    value = numeric(value)
                    if value is not None:
                        values.append(value / 2**20)
                row[field] = max(values) if values else None
            hashes = {point.get('payload_sha256') for point in group} - {None, ''}
            parity = all(point.get('parity', {}).get('passed') is True
                         and numeric(point.get('timed_parity_max_abs')) is not None
                         and ('timed_parity' not in point or
                              (len(point['timed_parity']) == len(point['http_latencies_s'])
                               and all(check.get('passed') is True for check in point['timed_parity'])))
                         for point in group)
            row['parity_verified'] = parity
            for field, parity_key in (
                ('initial_quantile_crossings_max', 'actual_quantile_crossings'),
                ('reference_quantile_crossings_max', 'native_quantile_crossings'),
            ):
                counts = [numeric(point.get('parity', {}).get(parity_key)) for point in group]
                counts = [count for count in counts if count is not None]
                row[field] = max(counts) if counts else None
            row['payload_sha256'] = next(iter(hashes)) if len(hashes) == 1 else ''
            complete = (len(group) == expected_blocks
                        and {point.get('block') for point in group} == set(range(expected_blocks))
                        and all(len(point['http_latencies_s']) == repetitions for point in group)
                        and parity and len(hashes) == 1)
            row['status'] = 'complete' if complete else 'partial'
            if len(hashes) > 1:
                warnings.append(f'{key}: workload hashes differ across blocks.')
            if key in summaries and numeric(summaries[key].get('n')) != len(latencies):
                warnings.append(f'{key}: stale summary.csv sample count; raw points take precedence.')
        else:
            row = dict(summaries[key])
            row.update(engine=engine, batch_size=batch, context=context,
                       status='summary only', parity_verified=False)
        cells[key] = row
    failure_keys = set()
    for error in errors:
        try:
            failure_keys.add(cell_key(error))
        except (KeyError, TypeError, ValueError):
            pass
    grid = sorted({(int(b), int(c)) for b, c in config.get('grid', [])}
                  | {(batch, context) for _, batch, context in cells}
                  | {(batch, context) for _, batch, context in failure_keys})
    for batch, context in grid:
        for engine in ENGINES:
            key = engine, batch, context
            if key not in cells:
                cells[key] = {'engine': engine, 'batch_size': batch, 'context': context,
                              'n': 0, 'blocks': 0, 'status': 'failed' if key in failure_keys else 'missing'}
            elif key in failure_keys:
                cells[key]['status'] = 'partial with failure'
    for batch, context in grid:
        hashes = {cells[(engine, batch, context)].get('payload_sha256') for engine in ENGINES} - {None, ''}
        if len(hashes) > 1:
            warnings.append(f'B={batch}, context={context}: payloads differ between engines; do not compare as paired.')
            for engine in ENGINES:
                cells[(engine, batch, context)]['status'] = 'payload mismatch'
    return {'config': config, 'metadata': metadata, 'errors': errors, 'warnings': warnings,
            'cells': cells, 'grid': grid, 'points': list(unique.values()),
            'expected_blocks': expected_blocks, 'repetitions': repetitions}


def render_plots(data, output):
    os.environ.setdefault('MPLCONFIGDIR', str(Path(gettempdir()) / 'vtsfm-matplotlib-cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.titleweight': 'bold', 'savefig.facecolor': 'white'})
    contexts = sorted({context for _, context in data['grid']})
    if not contexts:
        return []
    batches = sorted({batch for batch, _ in data['grid']})
    artifacts = []
    definitions = (
        ('performance', 'Chronos-2 serving performance',
         [('series_per_s', 'Series per second', 1), ('p95_ms', 'HTTP p95 latency (ms)', 1)]),
        ('memory', 'Chronos-2 GPU memory during warmed requests',
         [('peak_allocated_mib', 'Torch peak allocated (GiB)', 1 / 1024),
          ('sampled_peak_gpu_mib', 'NVML sampled resident peak (GiB)', 1 / 1024)]),
    )
    for filename, title, metrics in definitions:
        figure, axes = plt.subplots(2, len(contexts), figsize=(max(7, 3.35 * len(contexts)), 7.2), squeeze=False)
        for column, context in enumerate(contexts):
            for row_index, (metric, ylabel, scale) in enumerate(metrics):
                axis = axes[row_index][column]
                visible = []
                for engine in ENGINES:
                    rows = [data['cells'].get((engine, batch, context), {}) for batch in batches]
                    values = [numeric(row.get(metric)) for row in rows]
                    visible.extend(value * scale for value in values if value is not None)
                    axis.plot(batches, [value * scale if value is not None else math.nan for value in values],
                              color=COLORS[engine], linewidth=1.7, alpha=.85)
                    for batch, value, record in zip(batches, values, rows):
                        if value is not None:
                            axis.scatter(batch, value * scale, s=34, edgecolors=COLORS[engine],
                                         marker='o' if engine == 'native' else 's',
                                         facecolors=COLORS[engine] if record.get('status') == 'complete' else 'white', zorder=3)
                axis.set_xscale('log', base=2)
                axis.set_xticks(batches, labels=[str(batch) for batch in batches])
                axis.set_xlim(min(batches) / 1.3, max(batches) * 1.3)
                axis.set_ylim(0, max(visible) * 1.1 if visible and max(visible) > 0 else 1)
                axis.grid(axis='y', alpha=.18)
                if row_index == 0:
                    axis.set_title(f'Context {context:,}')
                else:
                    axis.set_xlabel('Series per request (log scale)')
                if column == 0:
                    axis.set_ylabel(ylabel)
                if not any(numeric(data['cells'].get((engine, batch, context), {}).get(metric)) is not None
                           for engine in ENGINES for batch in batches):
                    axis.text(.5, .5, 'No measurements', ha='center', va='center', transform=axis.transAxes, color='#666666')
        handles = [Line2D([0], [0], color=COLORS[engine], marker='o' if engine == 'native' else 's',
                          label=LABELS[engine]) for engine in ENGINES]
        handles.append(Line2D([0], [0], color='#666666', marker='o', markerfacecolor='white',
                              linestyle='none', label='Partial / summary only'))
        figure.suptitle(title, x=.07, ha='left', fontsize=17, weight='bold')
        figure.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .035), ncol=3, frameon=False, fontsize=9)
        note = (f"Exploratory results · target {data['expected_blocks']} × {data['repetitions']} HTTP requests/cell · "
                'missing cells omitted; hollow points are incomplete')
        figure.text(.07, .014, note, color='#555555', fontsize=8)
        figure.tight_layout(rect=(.015, .105, 1, .94), h_pad=2.4, w_pad=1.5)
        for extension in ('png', 'pdf'):
            path = output / f'{filename}.{extension}'
            figure.savefig(path, dpi=180, bbox_inches='tight')
            artifacts.append(path.name)
        plt.close(figure)
    return artifacts


def fmt(value, decimals=2):
    value = numeric(value)
    return '—' if value is None else f'{value:,.{decimals}f}'


def relative_link(path, output):
    return quote(os.path.relpath(path, output), safe='/')


def measured_observations(data):
    if data['config'].get('synthetic'):
        return None
    planned = {(int(batch), int(context)) for batch, context in data['config'].get('grid', [])}
    cells = data['cells']
    expected = {(engine, batch, context) for engine in ENGINES for batch, context in planned}
    if (not expected or set(cells) != expected
            or any(cells[key]['status'] != 'complete' for key in expected)):
        return None
    overheads = [100 * (cells[('vllm', batch, context)]['mean_ms']
                        / cells[('native', batch, context)]['mean_ms'] - 1)
                 for batch, context in sorted(planned)]
    faster, slower = sum(value < 0 for value in overheads), sum(value > 0 for value in overheads)
    equal = len(overheads) - faster - slower
    peaks = {}
    for engine in ENGINES:
        values = [numeric(cells[(engine, batch, context)].get('sampled_peak_gpu_mib'))
                  for batch, context in planned]
        values = [value for value in values if value is not None]
        peaks[engine] = max(values) / 1024 if values else None
    errors = [numeric(point.get('timed_parity_max_abs')) for point in data['points']]
    errors = [value for value in errors if value is not None]
    maximum_error = f'{max(errors):.6g}' if errors else 'not recorded'
    return (f"Across {len(planned)} matched shapes, plugin mean HTTP latency overhead versus native "
            f"(100 × (plugin/native − 1)) was minimum {min(overheads):.1f}%, "
            f"median {statistics.median(overheads):.1f}%, maximum {max(overheads):.1f}%. "
            f"The plugin was faster on {faster}/{len(planned)} shapes and slower on {slower}/{len(planned)} shapes"
            + (f' ({equal} equal)' if equal else '') + '. '
            f"Maximum sampled resident GPU memory was native {fmt(peaks['native'])} GiB and "
            f"plugin {fmt(peaks['vllm'])} GiB. The maximum timed-response absolute numerical "
            f"difference from the native reference was {maximum_error}.")


def markdown_report(data, results, output, plots):
    config, metadata, cells = data['config'], data['metadata'], data['cells']
    observed = [row for row in cells.values() if numeric(row.get('n')) and row['n'] > 0]
    complete = [row for row in cells.values() if row['status'] == 'complete']
    total = len(data['grid']) * len(ENGINES)
    manifest = metadata.get('source_manifest', {})
    source, time_axis = manifest.get('source', {}), manifest.get('time_axis', {})
    shape = manifest.get('array', {}).get('shape', [])
    driver = re.search(r'Driver Version:\s*([\d.]+)', metadata.get('nvidia_smi', ''))
    driver_version = metadata.get('gpu', {}).get('driver') or (driver.group(1) if driver else 'not recorded')
    packages = metadata.get('installed_packages', {})
    lines = ['# Chronos-2 serving benchmark', '',
             f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} from recorded measurements.", '',
             f"Compared plugin: `{config.get('plugin_backend', 'chronos2')}` (raw engine key `vllm`).", '',
             f"**Exploratory baseline:** {len(complete)}/{total} planned engine/workload cells are complete; "
             f"{len(observed)} have measurements and {len(data['errors'])} failure records were logged. "
             f"The target is {data['expected_blocks']} blocks × {data['repetitions']} requests "
             f"({data['expected_blocks'] * data['repetitions']} samples) per cell. "
             'Missing values are shown as —; incomplete cells are not promoted to complete results.', '']
    if config.get('synthetic'):
        lines += ['**Synthetic smoke data:** these inputs validate integration only; they are not AssetOpsBench observations or performance evidence.', '']
    if not observed:
        lines += ['**No measured cells are available yet.** No performance values have been inferred.', '']
    observations = measured_observations(data)
    if observations:
        lines += ['## Measured observations', '', observations, '']
    crossings = [numeric(point.get('parity', {}).get('actual_quantile_crossings')) for point in data['points']]
    crossings = [count for count in crossings if count is not None]
    lines += ['Quantile crossings are recorded without rejecting or reordering native forecasts. '
              'Numerical parity, rather than quantile monotonicity, is required; this is not a forecast-quality evaluation.', '']
    if crossings:
        lines += [f"The initial correctness responses recorded maximum {int(max(crossings))} crossing positions in one response "
                  f"and crossings in {sum(count > 0 for count in crossings)}/{len(crossings)} response checks. "
                  'A crossing means q0.1 > q0.5 + 1e-6 or q0.5 > q0.9 + 1e-6 at a series/horizon position. '
                  'These counts are from the initial check in each available block, not a total over all timed responses.', '']
    if plots:
        lines += ['## Performance', '', '![Throughput and HTTP p95 by context](performance.png)', '',
                  'Filled points meet the configured sample counts and recorded correctness checks. '
                  'Hollow points are partial or lack raw verification; missing points break the lines. '
                  f"The pooled p95 is descriptive: even the planned {data['expected_blocks'] * data['repetitions']} repeated requests "
                  'per cell do not establish a production tail-latency guarantee.', '',
                  '## GPU memory', '', '![Torch allocator and sampled GPU residency](memory.png)', '',
                  'Torch allocated memory and NVML resident memory measure different things; do not subtract them as a precise overhead estimate. '
                  'Peaks are maxima across available blocks after warmup. NVML residency is sampled rather than an exact instantaneous peak.', '']
    representative = [(batch, context) for batch, context in data['grid']
                      if batch in (1, 16, 128) and context in (512, 4096)]
    if not representative:
        representative = data['grid'][:10]
    lines += ['## Representative measurements', '',
              '| Context | Series/request | Engine | Samples / blocks | p50 ms | p95 ms | Series/s | Torch peak MiB | Resident peak MiB | Status |',
              '| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |']
    for batch, context in sorted(representative, key=lambda pair: (pair[1], pair[0])):
        for engine in ENGINES:
            row = cells[(engine, batch, context)]
            lines.append(f"| {context} | {batch} | {engine} | {int(row.get('n') or 0)} / {int(row.get('blocks') or 0)} | "
                         f"{fmt(row.get('p50_ms'))} | {fmt(row.get('p95_ms'))} | {fmt(row.get('series_per_s'))} | "
                         f"{fmt(row.get('peak_allocated_mib'))} | {fmt(row.get('sampled_peak_gpu_mib'))} | {row['status']} |")
    lines += ['', 'All planned cells, including missing/failing cells, are listed in [cells.csv](cells.csv).', '']
    if any(numeric(cells[('native', batch, context)].get('native_inprocess_n'))
           for batch, context in representative):
        lines += ['### Native in-process reference', '',
                  'These measurements exclude HTTP and have their own sample count; compare them only as a native reference.', '',
                  '| Context | Series/request | Samples | Native in-process p50 ms |',
                  '| ---: | ---: | ---: | ---: |']
        for batch, context in sorted(representative, key=lambda pair: (pair[1], pair[0])):
            row = cells[('native', batch, context)]
            lines.append(f"| {context} | {batch} | {int(row.get('native_inprocess_n') or 0)} | "
                         f"{fmt(row.get('native_inprocess_p50_ms'))} |")
        lines.append('')
    lines += [
              '## Data provenance', '',
              f"Dataset: **{manifest.get('dataset_name', 'not recorded')}**. "
              f"Asset: `{manifest.get('asset_id', 'not recorded')}`. "
              f"Prepared array: `{shape or 'not recorded'}` "
              f"(channels × {'positions' if config.get('synthetic') else 'hourly positions'}).", '']
    if source.get('url'):
        lines += [f"Pinned [AssetOpsBench source fixture]({source['url']}) at commit `{source.get('commit', 'unknown')}`; "
                  f"SHA-256 `{source.get('sha256', 'unknown')}`.", '']
    imputation = manifest.get('imputation', {})
    if manifest and not config.get('synthetic'):
        lines += [f"{time_axis.get('observations', 'Unknown')} positions span {time_axis.get('first', 'unknown')} through "
                  f"{time_axis.get('last', 'unknown')}. {imputation.get('imputed_rows', 'Unknown')} wholly missing rows were "
                  'causally forward-filled from the latest observed values. No backward fill or additional hourly aggregation was performed. '
                  'Inputs are float32 and are not normalized by the harness.', '']
        columns = manifest.get('columns', [])
        if columns:
            lines += ['Sensors: ' + ', '.join(str(column.get('name', '?')) for column in columns) + '.', '']
        if source.get('upstream_doi'):
            lines += [f"Upstream attribution: {source.get('upstream_citation', 'MetroPT-3')} "
                      f"[Dataset DOI]({source['upstream_doi']}). "
                      f"Fixture repository license: {source.get('repository_license', 'not recorded')}; "
                      f"upstream license: {source.get('upstream_license', 'not recorded')}.", '']
        if source.get('provenance_limit'):
            lines += [source['provenance_limit'], '']
    lines += ['## Method', '',
              f"- Model `{config.get('model', 'amazon/chronos-2')}`, revision `{config.get('revision', 'not recorded')}`, "
              f"{config.get('dtype', 'float32')}; horizon {config.get('forecast_horizon', 96)}; "
              f"quantiles {config.get('quantiles', [.1, .5, .9])}; cross-learning disabled.",
              f"- Each engine runs alone. {config.get('warmups', 10)} warmups precede each point. "
              'Workload order is randomized within each block, and engine order alternates between blocks.',
              '- One request is outstanding at a time. HTTP timing starts with submission of a preencoded JSON body '
              'and ends after decoding the response JSON, using the same persistent client. Numerical input batch sizes match.',
              '- Pooled p50/p95 use linear interpolation over recorded request latencies. Series/s equals '
              'series per request divided by mean request latency; it is not a concurrent-load saturation measurement.',
              '- Native forecasts establish numerical references; recorded parity uses rtol=1e-4 and atol=1e-4. '
              'A complete cell requires both initial and timed-response correctness records, all expected blocks, and all expected samples.',
              f"- CUDA allocator peaks reset after warmup. Resident GPU memory is sampled every "
              f"{config.get('nvml_interval_s', .02)} s; loaded/warmed/allocated/reserved values are preserved in the raw results.",
              '- Native in-process measurements are a separate reference: they include preprocessing, synchronized inference, '
              'and construction of CPU forecast lists, but exclude HTTP and JSON transport. They are not GPU-kernel-only timings.', '',
              '## Scope and limitations', '',
              ('- Synthetic smoke inputs test integration and numerical parity only. Their timings do not characterize an observational workload.'
               if config.get('synthetic') else
               '- These are repeated windows from seven channels of one compressor asset. Windows can overlap and are correlated; '
               '128 series per request does not mean 128 independent physical assets. Repetitions reuse the same payload at each shape.'),
              ('- Synthetic values have no physical units, observation timestamps, or data-quality interpretation.'
               if config.get('synthetic') else
               '- Forward-filled values preserve the hourly grid and alter the input distribution. This measures serving performance; '
               'it does not establish forecast accuracy or generalize to other assets.'),
              ('- Synthetic context lengths count generated positions; a separate 96-position horizon is reserved.'
               if config.get('synthetic') else
               '- Context lengths count hourly grid positions, including imputed positions. The next 96 positions are reserved; '
               'the fixture does not support context 8,192 plus that horizon.'),
              '- The experimental plugin delegates model computation to Chronos2Model, uses eager execution, and serializes engine requests. '
              'This evaluates batching inside a request, not general vLLM text-generation optimizations.',
              '- Alternating blocks do not eliminate clock, thermal, process, or order effects. No confidence intervals '
              'or production p95 claims are inferred from the pooled samples.', '',
              '## Environment and audit trail', '',
              f"Host: `{metadata.get('hostname', 'not recorded')}`; NVIDIA driver: `{driver_version}`. "
              f"Torch: `{packages.get('torch', 'not recorded')}`; vLLM: `{packages.get('vllm', 'not recorded')}`. "
              f"Chronos code commit: `{metadata.get('chronos_git_commit', 'not recorded')}`.", '']
    settings = []
    for engine in ENGINES:
        files = sorted(results.glob(f'{engine}-config-block-*.json'))
        if files:
            recorded = read_json(files[0], data['warnings'])
            values = recorded.get('torch_settings', recorded)
            settings.append(f"{engine}: matmul precision `{values.get('float32_matmul_precision', 'unknown')}`, "
                            f"TF32 matmul `{values.get('cuda_matmul_allow_tf32', 'unknown')}`, "
                            f"Torch threads `{values.get('num_threads', 'unknown')}`")
    if settings:
        lines += ['; '.join(settings) + '.', '']
    cleanup = read_json(results / 'cleanup.json', data['warnings'])
    if cleanup:
        lines += [f"Owned backend cleanup: {'verified' if cleanup.get('all_stopped') else 'incomplete; inspect cleanup.json'}. Existing services are never stopped by this harness.", '']
    for filename in ('summary.csv', 'points.jsonl', 'config.json', 'metadata.json', 'errors.jsonl', 'cleanup.json'):
        if (results / filename).exists():
            lines.append(f"- [{filename}]({relative_link(results / filename, output)})")
    if (results / 'harness').is_dir():
        lines.append(f"- [Executed harness snapshot]({relative_link(results / 'harness', output)}/)")
    lines += ['', 'Raw points take precedence over summary.csv when both are available. Summary-only cells remain explicitly unverified.', '']
    if data['errors']:
        lines += ['## Recorded failures', '', '| Engine | Series/request | Context | Block | Error |',
                  '| --- | ---: | ---: | ---: | --- |']
        for error in data['errors']:
            message = str(error.get('error', 'unknown')).replace('|', '\\|').replace('\n', ' ')[:400]
            lines.append(f"| {error.get('engine', '?')} | {error.get('batch_size', '?')} | {error.get('context', '?')} | "
                         f"{error.get('block', '?')} | {message} |")
        lines.append('')
    if data['warnings']:
        lines += ['## Incomplete input / audit warnings', '']
        lines += [f'- {warning}' for warning in data['warnings']]
        lines.append('')
    return '\n'.join(lines)


def generate_report(results, output, plots=True):
    results, output = Path(results).resolve(), Path(output).resolve()
    if not results.is_dir():
        raise FileNotFoundError(f'Results directory does not exist: {results}')
    output.mkdir(parents=True, exist_ok=True)
    data = load_results(results)
    artifacts = render_plots(data, output) if plots else []
    fields = ['engine', 'batch_size', 'context', 'status', 'n', 'blocks', 'p50_ms', 'p95_ms',
              'series_per_s', 'mean_ms', 'loaded_gpu_mib', 'warmed_gpu_mib', 'sampled_peak_gpu_mib',
              'peak_allocated_mib', 'peak_reserved_mib', 'native_inprocess_n', 'native_inprocess_p50_ms',
              'parity_verified', 'initial_quantile_crossings_max', 'reference_quantile_crossings_max', 'payload_sha256']
    with (output / 'cells.csv').open('w', newline='') as target:
        writer = csv.DictWriter(target, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(data['cells'][key] for key in sorted(data['cells']))
    (output / 'report.md').write_text(markdown_report(data, results, output, artifacts))
    (output / 'README.md').write_text(
        '# Chronos-2 benchmark report\n\n'
        'Read [report.md](report.md) for measurements, provenance, methods, failures, and limitations.\n\n'
        '- [All cells, including missing and partial results](cells.csv)\n'
        + ''.join(f'- [{name}]({name})\n' for name in artifacts)
        + '\nRegenerate with `python -m benchmarks.report --results RESULTS --output OUTPUT` from the benchmark environment. '
        'Relative arguments resolve against the repository root. Install `matplotlib` in that environment if needed. '
        'No model or GPU is used by reporting.\n')
    outcome = {'output': str(output), 'planned_cells': len(data['grid']) * len(ENGINES),
               'complete_cells': sum(row['status'] == 'complete' for row in data['cells'].values()),
               'observed_cells': sum(bool(numeric(row.get('n')) and row['n'] > 0) for row in data['cells'].values()),
               'error_records': len(data['errors']), 'warnings': data['warnings'], 'plots': artifacts}
    (output / 'audit.json').write_text(json.dumps(outcome, indent=2, allow_nan=False) + '\n')
    return outcome


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=Path('results'))
    parser.add_argument('--output', type=Path, default=Path('report'))
    parser.add_argument('--no-plots', action='store_true', help='Write tables and Markdown without Matplotlib')
    args = parser.parse_args()
    results = args.results if args.results.is_absolute() else HERE / args.results
    output = args.output if args.output.is_absolute() else HERE / args.output
    print(json.dumps(generate_report(results, output, plots=not args.no_plots), indent=2))


if __name__ == '__main__':
    main()
