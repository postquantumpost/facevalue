#!/usr/bin/env python3

import argparse
import asyncio
from pathlib import Path
from typing import Any, Sequence

import desk
import face


update_seconds: float = 10.0
state: dict[str, Any] = {
    "desktop_client": None,
    "desktop_config": None,
}


def parse_boolean(value: str) -> bool:
    normalized = value.lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise argparse.ArgumentTypeError("boolean values must be 'true' or 'false'")


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze face images or display a remote VNC desktop."
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="JSON file previously created by --output",
    )
    parser.add_argument(
        "--fdir",
        type=Path,
        help="directory containing images with a face",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="destination JSON file for face information",
    )
    parser.add_argument(
        "--desktop",
        type=Path,
        help="JSON file containing remote desktop login information",
    )
    parser.add_argument(
        "--window",
        type=parse_boolean,
        default=False,
        metavar="true|false",
        help="display requested face or desktop data (default: false)",
    )
    arguments = parser.parse_args(argv)
    for option_name in ("input", "output", "desktop"):
        path = getattr(arguments, option_name)
        if path is not None and path.suffix.lower() != ".json":
            parser.error(f"--{option_name} must name a file with a .json extension")

    exclusive_sources = [
        name
        for name in ("input", "fdir", "desktop")
        if getattr(arguments, name) is not None
    ]
    if len(exclusive_sources) > 1:
        parser.error("--input, --fdir, and --desktop are mutually exclusive")
    return arguments


def run(arguments: argparse.Namespace) -> None:
    if arguments.desktop is not None:
        config = desk.read_config(arguments.desktop)
        state["desktop_config"] = config
        asyncio.run(
            desk.display_desktop(
                config,
                update_seconds,
                state=state,
            )
        )
        return
    face.run(
        arguments.fdir,
        arguments.input,
        arguments.output,
        arguments.window,
    )


def main(argv: Sequence[str] | None = None) -> int:
    run(parse_arguments(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
