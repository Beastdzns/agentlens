#!/usr/bin/env python3
"""
AgentLens Phase 6 Demo Launcher
===============================

Starts the unified AgentLens backend server hosting both the FastAPI endpoints
and the static frontend observability dashboard.

Usage:
    python run_demo.py [--port 8000] [--host 127.0.0.1]
"""

from __future__ import annotations

import argparse
import os
import sys
import webbrowser


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the AgentLens Phase 6 Demo Server")
    parser.add_argument(
        "--host", default="127.0.0.1", help="Host interface to bind (default: 127.0.0.1)"
    )
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on (default: 8000)")
    parser.add_argument(
        "--no-browser", action="store_true", help="Do not automatically open browser"
    )
    args = parser.parse_args()

    api_key_set = bool(os.getenv("GEMINI_API_KEY"))
    dashboard_url = f"http://{args.host}:{args.port}/app/"
    docs_url = f"http://{args.host}:{args.port}/docs"

    print("=" * 64)
    print(" AgentLens — Observability Platform (Phase 6 Demo)")
    print("=" * 64)
    print(f" • Dashboard URL : {dashboard_url}")
    print(f" • API Docs      : {docs_url}")
    print(
        f" • Gemini Key    : {'Configured (Live Gemini execution)' if api_key_set else 'None (Offline fallback simulation active)'}"
    )
    print("=" * 64)
    print("Starting server... Press Ctrl+C to stop.\n")

    if not args.no_browser and "pytest" not in sys.modules:
        try:
            webbrowser.open(dashboard_url)
        except Exception:
            pass

    import uvicorn

    uvicorn.run("agentlens.web_api:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
