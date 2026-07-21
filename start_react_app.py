"""Start the FastAPI backend and React development server together."""

import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"


def main() -> None:
    if not (FRONTEND_DIR / "node_modules").exists():
        print("React packages are missing. Run: cd frontend && npm install")
        raise SystemExit(1)

    print("Starting prediction API at http://127.0.0.1:8000")
    api = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "api:app",
            "--reload",
            "--host",
            "127.0.0.1",
            "--port",
            "8000",
        ],
        cwd=PROJECT_ROOT,
    )

    print("Starting React at http://127.0.0.1:5173")
    react = subprocess.Popen(
        ["npm", "run", "dev", "--", "--host", "127.0.0.1"],
        cwd=FRONTEND_DIR,
    )

    print("Both servers are running. Press Control+C to stop them.")
    try:
        while api.poll() is None and react.poll() is None:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        for process in [react, api]:
            if process.poll() is None:
                process.terminate()
        for process in [react, api]:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()

    if api.returncode not in (0, -15, None):
        print("The Python API stopped unexpectedly. Check the error above.")
    if react.returncode not in (0, -15, None):
        print("The React server stopped unexpectedly. Check the error above.")


if __name__ == "__main__":
    main()
