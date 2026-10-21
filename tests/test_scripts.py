"""Exercise script boundaries without package downloads, CUDA, or live services."""
import json
import os
from pathlib import Path
import shutil
import shlex
import subprocess
import sys
import tempfile
import unittest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def python_launcher(body):
    # A shell trampoline also supports a test interpreter inside a spaced path.
    return '#!/bin/sh\nexec ' + shlex.quote(sys.executable) + ' -c ' + shlex.quote(body) + ' "$@"\n'


class ScriptChecks(unittest.TestCase):
    def setUp(self):
        self.assertTrue((SCRIPTS / "setup.sh").is_file(), "Setup scripts are not implemented")
        self.temporary = tempfile.TemporaryDirectory(prefix="vtsfm scripts ")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve() / "clone with spaces"
        shutil.copytree(SCRIPTS, self.root / "scripts")
        (self.root / "examples").mkdir()
        (self.root / "examples/request.json").write_text('{"data":{"inputs":[]}}\n')
        self.bin = Path(self.temporary.name).resolve() / "commands"
        self.bin.mkdir()
        self.log = Path(self.temporary.name) / "calls.jsonl"
        self.env = os.environ.copy()
        self.env.update(PATH=f"{self.bin}:/usr/bin:/bin", STUB_LOG=str(self.log))
        for name in ("UV_CACHE_DIR", "UV_PYTHON_INSTALL_DIR", "UV_INSTALL_DIR", "UV_UNMANAGED_INSTALL"):
            self.env.pop(name, None)
        self.write_command("uname", "print('Linux')")

    def write_command(self, name, body, target=None):
        path = target or self.bin / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(python_launcher(body))
        path.chmod(0o755)
        return path

    def fake_python(self, version="3.11.17"):
        body = f"""import json, os, sys
if sys.argv[1:2] == ['-c']:
    print({version!r})
else:
    with open(os.environ['STUB_LOG'], 'a') as log:
        log.write(json.dumps({{'tool':'python', 'args':sys.argv[1:], 'cwd':os.getcwd()}})+'\\n')
"""
        return self.write_command("python", body, self.root / ".venv/bin/python")

    def fake_uv(self, target=None, version="0.13.4"):
        body = f"""import json, os, pathlib, sys
with open(os.environ['STUB_LOG'], 'a') as log:
    log.write(json.dumps({{'tool':'uv', 'version':{version!r}, 'args':sys.argv[1:], 'cwd':os.getcwd(),
                          'cache':os.environ.get('UV_CACHE_DIR'),
                          'python_dir':os.environ.get('UV_PYTHON_INSTALL_DIR')}})+'\\n')
if sys.argv[1:2] == ['venv']:
    path = pathlib.Path(sys.argv[-1]) / 'bin/python'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text({python_launcher("print('3.11.17')")!r})
    path.chmod(0o755)
elif sys.argv[1:] == ['--version']:
    print('uv ' + {version!r})
"""
        return self.write_command("uv", body, target)

    def run_script(self, name, *arguments):
        return subprocess.run(["/bin/bash", str(self.root / "scripts" / name), *arguments],
                              cwd=self.temporary.name, env=self.env, capture_output=True,
                              text=True, timeout=15)

    def calls(self):
        return [json.loads(line) for line in self.log.read_text().splitlines()] if self.log.exists() else []

    def test_setup_help_has_no_environment_or_network_side_effects(self):
        result = self.run_script("setup.sh", "--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--cpu", result.stdout)
        self.assertFalse((self.root / ".venv").exists())
        self.assertFalse((self.root / ".tools").exists())
        self.assertEqual(self.calls(), [])

    def test_non_linux_gpu_setup_refuses_before_installation(self):
        self.write_command("uname", "print('Darwin')")
        result = self.run_script("setup.sh")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--cpu", result.stderr)
        self.assertEqual(self.calls(), [])
        self.assertFalse((self.root / ".venv").exists())

    def test_cpu_setup_uses_pinned_local_environment_and_honors_cache_paths(self):
        self.fake_uv()
        self.env.update(UV_CACHE_DIR=str(self.root / "custom cache"),
                        UV_PYTHON_INSTALL_DIR=str(self.root / "custom python"))
        result = self.run_script("setup.sh", "--cpu")
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = self.calls()
        self.assertTrue(any(call["args"] == ["python", "install", "3.11.17"] for call in calls))
        install = next(call for call in calls if call["args"][:2] == ["pip", "install"])
        self.assertIn(".[dev,benchmark]", install["args"])
        self.assertIn(str(self.root / ".venv/bin/python"), install["args"])
        self.assertEqual(install["cwd"], str(self.root))
        self.assertEqual(install["cache"], self.env["UV_CACHE_DIR"])
        self.assertEqual(install["python_dir"], self.env["UV_PYTHON_INSTALL_DIR"])

    def test_gpu_setup_selects_serve_extra(self):
        self.fake_uv()
        result = self.run_script("setup.sh")
        self.assertEqual(result.returncode, 0, result.stderr)
        install = next(call for call in self.calls() if call["args"][:2] == ["pip", "install"])
        self.assertIn(".[serve,dev,benchmark]", install["args"])

    def test_missing_uv_bootstraps_once_inside_the_clone_without_shell_changes(self):
        source = self.fake_uv(target=self.bin / "uv-source")
        self.env["STUB_UV_SOURCE"] = str(source)
        installer = '''#!/bin/sh
set -eu
[ "$UV_NO_MODIFY_PATH" = 1 ]
[ "$UV_INSTALL_DIR" = "$UV_UNMANAGED_INSTALL" ]
mkdir -p "$UV_UNMANAGED_INSTALL"
cp "$STUB_UV_SOURCE" "$UV_UNMANAGED_INSTALL/uv"
'''
        self.write_command("curl", f"""import json, os, pathlib, sys
with open(os.environ['STUB_LOG'], 'a') as log:
    log.write(json.dumps({{'tool':'curl', 'args':sys.argv[1:]}})+'\\n')
pathlib.Path(sys.argv[sys.argv.index('--output')+1]).write_text({installer!r})
""")
        first = self.run_script("setup.sh", "--cpu")
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertTrue((self.root / ".tools/uv/uv").is_file())
        second = self.run_script("setup.sh", "--cpu")
        self.assertEqual(second.returncode, 0, second.stderr)
        downloads = [call for call in self.calls() if call["tool"] == "curl"]
        self.assertEqual(len(downloads), 1)
        self.assertIn("https://astral.sh/uv/0.13.4/install.sh", downloads[0]["args"])

    def test_older_global_uv_uses_a_current_per_clone_uv(self):
        self.fake_uv(version="0.11.18")
        self.fake_uv(target=self.root / ".tools/uv/uv")
        result = self.run_script("setup.sh", "--cpu")
        self.assertEqual(result.returncode, 0, result.stderr)
        install = next(call for call in self.calls() if call["args"][:2] == ["pip", "install"])
        self.assertEqual(install["version"], "0.13.4")

    def test_setup_does_not_replace_an_existing_wrong_python(self):
        python = self.fake_python("3.12.0")
        before = python.read_bytes()
        self.fake_uv()
        result = self.run_script("setup.sh", "--cpu")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("3.11.17", result.stderr)
        self.assertEqual(python.read_bytes(), before)
        self.assertFalse(any(call["args"][:2] == ["pip", "install"] for call in self.calls()))

    def test_wrappers_preserve_arguments_and_work_from_another_directory(self):
        self.fake_python()
        cases = [
            ("doctor.sh", ["--help"], ["-m", "vtsfm.cli", "doctor", "--help"]),
            ("serve.sh", ["--backend", "vtsfm", "--port", "8001"],
             ["-m", "vtsfm.cli", "serve", "--backend", "vtsfm", "--port", "8001"]),
            ("smoke.sh", ["--synthetic", "--output", "result set"],
             ["-m", "benchmarks.run", "--smoke", "--synthetic", "--output", "result set"]),
            ("benchmark.sh", ["--output", "result set"], ["-m", "benchmarks.run", "--output", "result set"]),
            ("report.sh", ["--results", "result set"], ["-m", "benchmarks.report", "--results", "result set"]),
        ]
        for script, arguments, expected in cases:
            with self.subTest(script=script):
                result = self.run_script(script, *arguments)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(self.calls()[-1]["args"], expected)
                self.assertEqual(self.calls()[-1]["cwd"], str(self.root))

    def test_missing_environment_has_actionable_error(self):
        for script in ["doctor.sh", "serve.sh", "smoke.sh", "benchmark.sh", "report.sh"]:
            with self.subTest(script=script):
                result = self.run_script(script)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("setup.sh", result.stderr)

    def test_request_accepts_base_or_pooling_url_and_keeps_http_error_body(self):
        self.write_command("curl", """import json, os, sys
with open(os.environ['STUB_LOG'], 'a') as log:
    log.write(json.dumps({'tool':'curl', 'args':sys.argv[1:]})+'\\n')
print('response body')
sys.exit(int(os.environ.get('STUB_CURL_EXIT', '0')))
""")
        for url in [None, "http://localhost:9002", "http://localhost:9002/pooling"]:
            result = self.run_script("request.sh", *([] if url is None else [url]))
            self.assertEqual(result.returncode, 0, result.stderr)
            arguments = self.calls()[-1]["args"]
            self.assertIn("--fail-with-body", arguments)
            self.assertEqual(arguments[arguments.index("--write-out") + 1], "\\n")
            self.assertIn("@" + str(self.root / "examples/request.json"), arguments)
            self.assertIn("http://127.0.0.1:8001/pooling" if url is None else "http://localhost:9002/pooling", arguments)
        self.env["STUB_CURL_EXIT"] = "22"
        result = self.run_script("request.sh")
        self.assertEqual(result.returncode, 22)
        self.assertIn("response body", result.stdout)

    def test_all_scripts_have_valid_bash_syntax(self):
        for script in SCRIPTS.glob("*.sh"):
            result = subprocess.run(["/bin/bash", "-n", str(script)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, f"{script.name}: {result.stderr}")


if __name__ == "__main__":
    unittest.main()
