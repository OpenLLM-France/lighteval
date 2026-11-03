# MIT License

# Copyright (c) 2026 OpenLLM-France

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Execute a model-generated React component against its Jest tests.

SECURITY: this runs model-generated JavaScript with Node/Jest. It is NOT a sandbox — run it
only in an isolated/containerized environment, as for any code-execution benchmark.

The Node/Jest project (``node_modules`` + config) is prepared once via ``jsenv/setup.sh``; its
path defaults to the bundled ``jsenv/`` directory and can be overridden with the
``WEBAPP1K_JEST_PROJECT`` environment variable. See ``README.md``.
"""

import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path


logger = logging.getLogger(__name__)

_DEFAULT_PROJECT_DIR = str(Path(__file__).parent / "jsenv")
_MIN_NODE_MAJOR = 18

_node_checked = False


def _node_path() -> str:
    """PATH for locating/running Node, with ``WEBAPP1K_NODE_BIN`` prepended if set.

    On HPC the Node binary often lives on a shared FS outside the container image, and a login
    shell inside the container can reset ``PATH`` and drop a prepended entry. Setting
    ``WEBAPP1K_NODE_BIN=/path/to/node/bin`` makes the runner find Node regardless of the ambient
    PATH, which is the reliable way to expose it to the eval.
    """
    path = os.environ.get("PATH", "")
    node_bin = os.environ.get("WEBAPP1K_NODE_BIN")
    return f"{node_bin}{os.pathsep}{path}" if node_bin else path


def _which(tool: str) -> str | None:
    return shutil.which(tool, path=_node_path())


def _ensure_node_available() -> None:
    """Fail loudly if Node is not runnable, instead of silently scoring every sample 0.

    The metric executes model output with ``npx jest``; if ``node``/``npx`` are not on PATH in the
    eval environment (a common trap on HPC: the GPU container ships no Node), every run would fail
    and the whole benchmark would read 0 with no explanation. Checked once (cached).
    """
    global _node_checked
    if _node_checked:
        return
    for tool in ("node", "npx"):
        if _which(tool) is None:
            raise RuntimeError(
                f"WebApp1K needs Node.js, but '{tool}' is not on PATH in the eval environment. "
                "This task executes model output with Jest, so 'node' and 'npx' must be runnable "
                "wherever the metric runs (e.g. inside your Apptainer/Docker container, not just on "
                "the login node). Install Node >= 18 and put it on PATH, or set WEBAPP1K_NODE_BIN to "
                "a prebuilt Node's bin dir (reliable on an offline HPC node). See the WebApp1K README."
            )
    try:
        out = subprocess.run([_which("node"), "--version"], capture_output=True, timeout=30, check=True, text=True)
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"WebApp1K: 'node' is on PATH but not runnable ({e}). See the WebApp1K README.") from e
    # Guard against an ancient Node (e.g. the distro default) that passes the existence check but
    # then fails Jest on every sample — the same silent-all-zero trap, one layer deeper.
    m = re.match(r"v(\d+)\.", out.stdout.strip())
    if m and int(m.group(1)) < _MIN_NODE_MAJOR:
        raise RuntimeError(
            f"WebApp1K needs Node.js >= {_MIN_NODE_MAJOR}, but found {out.stdout.strip()} on PATH. "
            "Jest and the testing-library stack will not run on an older Node. Install a newer Node "
            "(on an offline HPC node, prepend a prebuilt Node's bin via the container PATH). "
            "See the WebApp1K README."
        )
    _node_checked = True


def resolve_project_dir() -> str:
    """Return the prepared Jest project dir, or raise an actionable error if it is not usable."""
    _ensure_node_available()
    project_dir = os.environ.get("WEBAPP1K_JEST_PROJECT", _DEFAULT_PROJECT_DIR)
    if not (Path(project_dir) / "node_modules").is_dir():
        raise RuntimeError(
            "WebApp1K needs a prepared Node/Jest project but none was found at "
            f"'{project_dir}'. Run the one-time setup:\n"
            f"    bash {_DEFAULT_PROJECT_DIR}/setup.sh\n"
            "or point WEBAPP1K_JEST_PROJECT at an already-prepared project. "
            "See the WebApp1K README for details."
        )
    return project_dir


def run_jest(
    project_dir: str,
    scenario: str,
    component_code: str,
    test_file: str,
    timeout: int = 120,
) -> dict:
    """Write the component + test into a throwaway dir under the project and run Jest on it.

    Returns a dict with ``passed`` (True only if both tests pass) and, on failure, diagnostic
    fields (``num_passed``, ``num_failed``, ``error``).
    """
    runs_root = Path(project_dir) / ".webapp1k_runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    work_dir = Path(tempfile.mkdtemp(dir=runs_root))
    test_path = work_dir / f"{scenario}.test.js"
    result_path = work_dir / "jest_result.json"
    try:
        (work_dir / f"{scenario}.js").write_text(component_code, encoding="utf-8")
        test_path.write_text(test_file, encoding="utf-8")
        cmd = [
            _which("npx") or "npx",
            "jest",
            str(test_path),
            "--rootDir",
            str(project_dir),
            "--runInBand",
            "--silent",
            "--ci",
            "--json",
            f"--outputFile={result_path}",
        ]
        env = {**os.environ, "PATH": _node_path()}
        try:
            subprocess.run(cmd, cwd=project_dir, capture_output=True, timeout=timeout, env=env)
        except subprocess.TimeoutExpired:
            return {"passed": False, "error": "timeout"}
        except FileNotFoundError as e:
            # node/npx vanished mid-run: systematic, not a per-sample failure — raise loudly
            # rather than silently scoring 0 (resolve_project_dir normally catches this upfront).
            raise RuntimeError(
                "WebApp1K: 'npx'/'node' not found while launching Jest. Node must be on PATH in "
                "the eval environment. See the WebApp1K README."
            ) from e

        if not result_path.exists():
            return {"passed": False, "error": "jest produced no result (compilation error?)"}
        data = json.loads(result_path.read_text(encoding="utf-8"))
        num_passed = data.get("numPassedTests", 0)
        num_failed = data.get("numFailedTests", 0)
        # Both the success and failure tests must pass, and nothing may fail.
        passed = num_failed == 0 and num_passed >= 2
        return {"passed": passed, "num_passed": num_passed, "num_failed": num_failed}
    except Exception as e:  # noqa: BLE001 - never let a bad generation crash the eval loop
        logger.warning("WebApp1K Jest run failed for %r: %s", scenario, e)
        return {"passed": False, "error": str(e)}
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
