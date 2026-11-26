"""Reproduce the independent native oracle and the concurrent untimed HTTP gates."""
import argparse
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

from benchmarks import continuous
from model_integration_support import compare_response, native_reference


def sha256(body):
    return hashlib.sha256(body).hexdigest()


def make_oracle(fixture, output):
    result = native_reference('moirai2', continuous._cases(fixture, 4))
    for row in result['cases']:
        body = {'model': result['model_id'], 'task': 'plugin', 'data': row['case']['data']}
        row['payload_sha256'] = sha256(json.dumps(body, separators=(',', ':')).encode())
    output.write_text(json.dumps(result, allow_nan=False))


def run_arm(args):
    import requests

    rows = json.loads(args.oracle.read_text())['cases']
    by_hash = {row['payload_sha256']: row for row in rows}
    if len(by_hash) != 4:
        raise ValueError('Expected four distinct fixture inputs')
    replay = continuous.replay
    args.output.mkdir(parents=True, exist_ok=False)

    def replay_with_checks(url, arrivals, **kwargs):
        prefix = kwargs['request_prefix']
        if not prefix.startswith('warmup-'):
            return replay(url, arrivals, **kwargs)
        captured = {}

        def send(identity, payload):
            with requests.Session() as client:
                client.trust_env = False
                response = client.post(url + '/pooling', data=payload, timeout=kwargs['timeout'],
                    headers={'Content-Type': 'application/json', 'X-Request-Id': identity})
                body = response.json()
            if response.ok:
                captured[identity] = (payload, body)
            return response.status_code, body

        result = replay(url, arrivals, send=send, **kwargs)
        records = []
        for sample in result['samples']:
            identity = sample['request_id']
            if sample['status'] != 'passed':
                records.append(dict(request_id=identity, passed=False, error=sample.get('error')))
                continue
            payload, body = captured[identity]
            digest = sha256(payload)
            row = by_hash[digest]
            try:
                check = compare_response(body, row['case'], row['expected'])
                records.append(dict(request_id=identity, passed=True, payload_sha256=digest, **check))
            except AssertionError as error:
                records.append(dict(request_id=identity, passed=False, payload_sha256=digest, error=str(error)))
        passed = len(records) == len(arrivals) and all(r['passed'] for r in records)
        (args.output / f'{prefix}-native.json').write_text(json.dumps({'passed': passed, 'samples': records}))
        if not passed:
            raise RuntimeError(f'{prefix} failed independent native parity. See the saved records.')
        return result

    with patch.object(continuous, 'replay', replay_with_checks):
        return continuous.main(['--adapter', 'moirai2', '--data-dir', str(args.fixture),
            '--output', str(args.output / 'bench'), '--modes', args.modes, '--scenarios', 'saturated,bursty',
            '--requests', '100', '--warmup-requests', '20', '--queue-limit', '128', '--diagnostic-requests', '0',
            '--port', str(args.port)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    oracle = sub.add_parser('oracle')
    oracle.add_argument('fixture', type=Path)
    oracle.add_argument('output', type=Path)
    arm = sub.add_parser('replay')
    arm.add_argument('fixture', type=Path)
    arm.add_argument('oracle', type=Path)
    arm.add_argument('output', type=Path)
    arm.add_argument('--modes', choices=('whole,continuous','continuous,whole'), required=True)
    arm.add_argument('--port', type=int, default=18589)
    args = parser.parse_args()
    return make_oracle(args.fixture, args.output) if args.command == 'oracle' else run_arm(args)


if __name__ == '__main__':
    raise SystemExit(main())
