"""FastAPI server launcher for Workforce Risk ML System."""

import argparse
import os
import sys
from pathlib import Path
import uvicorn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    # Set PyTorch thread allocation to reduce virtual memory footprint in containers
    try:
        import torch
        torch.set_num_threads(int(os.environ.get("TORCH_NUM_THREADS", "1")))
        torch.set_num_interop_threads(int(os.environ.get("TORCH_NUM_INTEROP_THREADS", "1")))
    except Exception:
        pass

    parser = argparse.ArgumentParser(description="Launch Workforce Risk ML FastAPI Serving API")
    is_prod = os.environ.get("ENVIRONMENT", "development").lower() in ("production", "prod") or bool(os.environ.get("RENDER"))
    default_host = os.environ.get("HOST", "0.0.0.0" if (is_prod or os.environ.get("PORT")) else "127.0.0.1")
    default_port = int(os.environ.get("PORT", "8000"))

    parser.add_argument("--host", type=str, default=default_host, help="Host address to bind server to")
    parser.add_argument("--port", type=int, default=default_port, help="Port to listen on")
    parser.add_argument("--reload", action="store_true", default=not is_prod, help="Enable auto-reload for development")
    parser.add_argument("--no-reload", dest="reload", action="store_false", help="Disable auto-reload")
    args = parser.parse_args()

    print(f"[Starting Server] Workforce Risk Serving API launching on http://{args.host}:{args.port} (reload={args.reload})")
    uvicorn.run("workforce_risk.serving.app:app", host=args.host, port=args.port, reload=args.reload, workers=1)


if __name__ == "__main__":
    main()
