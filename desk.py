#!/usr/bin/env python3

import asyncio
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DESKTOP_WINDOW_WIDTH_PX = 640
DESKTOP_WINDOW_HEIGHT_PX = 480


@dataclass(frozen=True)
class DesktopConfig:
    host: str
    port: int = 5900
    username: str | None = None
    password: str | None = None


def read_config(config_path: Path) -> DesktopConfig:
    if not config_path.is_file():
        raise FileNotFoundError(f"desktop login file not found: {config_path}")
    try:
        with config_path.open("r", encoding="utf-8") as config_file:
            data = json.load(config_file)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"desktop login file is not valid JSON: {config_path}"
        ) from error
    if not isinstance(data, dict):
        raise ValueError("desktop login JSON root must be an object")

    host = data.get("host")
    if not isinstance(host, str) or not host.strip():
        raise ValueError("desktop login JSON requires a non-empty 'host'")
    port = data.get("port", 5900)
    if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
        raise ValueError("desktop login 'port' must be an integer from 1 to 65535")
    username = data.get("username")
    password = data.get("password")
    if username is not None and not isinstance(username, str):
        raise ValueError("desktop login 'username' must be a string")
    if password is not None and not isinstance(password, str):
        raise ValueError("desktop login 'password' must be a string")
    return DesktopConfig(
        host=host,
        port=port,
        username=username,
        password=password,
    )


def scale_desktop(
    image: Any,
    width_px: int = DESKTOP_WINDOW_WIDTH_PX,
    height_px: int = DESKTOP_WINDOW_HEIGHT_PX,
) -> Any:
    import numpy as np
    from PIL import Image

    source = Image.fromarray(np.asarray(image).copy())
    source.thumbnail(
        (width_px, height_px),
        Image.Resampling.LANCZOS,
    )
    canvas = Image.new("RGBA", (width_px, height_px), (0, 0, 0, 255))
    offset = ((width_px - source.width) // 2, (height_px - source.height) // 2)
    canvas.paste(source, offset)
    return np.asarray(canvas)


async def display_desktop(
    config: DesktopConfig,
    update_seconds: float,
    state: dict[str, Any] | None = None,
) -> None:
    if update_seconds <= 0:
        raise ValueError("update_seconds must be greater than zero")
    try:
        import asyncvnc
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError(
            "desktop dependencies are missing; install requirements.txt"
        ) from error

    figure, axis = plt.subplots(
        figsize=(
            DESKTOP_WINDOW_WIDTH_PX / 100,
            DESKTOP_WINDOW_HEIGHT_PX / 100,
        ),
        dpi=100,
    )
    figure.canvas.manager.set_window_title(f"FaceValue Desktop: {config.host}")
    axis.set_position((0, 0, 1, 1))
    axis.axis("off")
    plt.ion()
    plt.show(block=False)

    try:
        async with asyncvnc.connect(
            config.host,
            config.port,
            username=config.username,
            password=config.password,
        ) as client:
            if state is not None:
                state["desktop_client"] = client
            while plt.fignum_exists(figure.number):
                cycle_started_seconds = time.monotonic()
                screenshot = await client.screenshot()
                axis.clear()
                axis.imshow(scale_desktop(screenshot))
                axis.axis("off")
                figure.canvas.draw_idle()
                figure.canvas.flush_events()

                remaining_seconds = update_seconds - (
                    time.monotonic() - cycle_started_seconds
                )
                deadline_seconds = time.monotonic() + max(0.0, remaining_seconds)
                while (
                    plt.fignum_exists(figure.number)
                    and time.monotonic() < deadline_seconds
                ):
                    figure.canvas.flush_events()
                    await asyncio.sleep(0.05)
    finally:
        if state is not None:
            state["desktop_client"] = None
        plt.ioff()
        plt.close(figure)
