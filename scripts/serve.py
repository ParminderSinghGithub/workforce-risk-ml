"""FastAPI server launcher for Workforce Risk ML System."""

import argparse
import sys
from pathlib import Path
import uvicorn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch Workforce Risk ML FastAPI Serving API")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address to bind server to")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    args = parser.parse_args()

    print(f"[Starting Server] Workforce Risk Serving API launching on http://{args.host}:{args.port}")
    uvicorn.run("workforce_risk.serving.app:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
