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

from dotenv import load_dotenv
import os
import socket
import sys
import webbrowser

load_dotenv()

def is_port_in_use(host: str, port: int) -> bool:
    """Check if a network port is already in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def find_available_port(host: str, start_port: int, max_attempts: int = 30) -> int:
    """Find the next available TCP port starting from start_port."""
    for p in range(start_port, start_port + max_attempts):
        if not is_port_in_use(host, p):
            return p
    return start_port


def main() -> None:
    default_port = int(os.getenv("PORT", "8000"))

    parser = argparse.ArgumentParser(description="Run the AgentLens Phase 6 Demo Server")
    parser.add_argument(
        "--host", default="127.0.0.1", help="Host interface to bind (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=default_port,
        help=f"Port to listen on (default: {default_port})",
    )
    parser.add_argument(
        "--no-browser", action="store_true", help="Do not automatically open browser"
    )
    args = parser.parse_args()

    port = args.port
    if is_port_in_use(args.host, port):
        if "--port" in sys.argv:
            print(f"[!] Warning: Port {port} is already in use on {args.host}.")
            print("    A service (such as Docker container 'cap-backend') may be occupying it.")
            print(f"    To free it, stop that service or specify another port, e.g. --port {port + 1}.\n")
        else:
            fallback = find_available_port(args.host, port + 1)
            print(f"[!] Notice: Port {port} is occupied (e.g. by Docker container 'cap-backend').")
            print(f"    -> Automatically switching to available port: {fallback}\n")
            port = fallback

    api_key_set = bool(os.getenv("GEMINI_API_KEY"))
    dashboard_url = f"http://{args.host}:{port}/app/"
    docs_url = f"http://{args.host}:{port}/docs"

    print("=" * 64)
    print(" AgentLens — Observability Platform (Phase 6 Demo)")
    print("=" * 64)
    print(f" • Dashboard URL : {dashboard_url}")
    print(f" • API Docs      : {docs_url}")
    print(
        f" • Gemini Key    : {'Configured (Live Gemini execution)' if api_key_set else 'None (Offline fallback simulation active)'}"
    )
    print("=" * 64)
    print(f"Starting server on http://{args.host}:{port} ... Press Ctrl+C to stop.\n")

    if not args.no_browser and "pytest" not in sys.modules:
        try:
            webbrowser.open(dashboard_url)
        except Exception:
            pass

    import uvicorn

    uvicorn.run("agentlens.web_api:app", host=args.host, port=port, reload=False)


if __name__ == "__main__":
    main()
