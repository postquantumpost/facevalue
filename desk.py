#!/usr/bin/env python3

import asyncio
import json
import queue
import ssl
import threading
import warnings
import zlib
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


DESKTOP_WINDOW_WIDTH_PX = 640
DESKTOP_WINDOW_HEIGHT_PX = 480
CONNECT_TIMEOUT_SECONDS = 5.0


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


@asynccontextmanager
async def open_desktop(config: DesktopConfig):
    try:
        import asyncvnc
        from cryptography.hazmat.primitives.ciphers import Cipher, modes
        from cryptography.hazmat.decrepit.ciphers.algorithms import TripleDES
        from cryptography.utils import CryptographyDeprecationWarning
    except ImportError as error:
        raise RuntimeError(
            "desktop dependencies are missing; install requirements.txt"
        ) from error

    reader, writer = await asyncio.wait_for(
        asyncio.open_connection(config.host, config.port),
        timeout=CONNECT_TIMEOUT_SECONDS,
    )
    try:
        intro = await reader.readline()
        if intro[:4] != b"RFB ":
            raise ValueError("remote endpoint is not a VNC server")
        writer.write(b"RFB 003.008\n")
        await writer.drain()

        auth_type_count = (await reader.readexactly(1))[0]
        auth_types = set(await reader.readexactly(auth_type_count))
        if 19 in auth_types:
            writer.write(b"\x13")
            await writer.drain()
            version = await reader.readexactly(2)
            if version != b"\x00\x02":
                raise ValueError(
                    f"unsupported VeNCrypt version: {version.hex()}"
                )
            writer.write(version)
            await writer.drain()
            if (await reader.readexactly(1))[0] != 0:
                raise ValueError("VNC server rejected VeNCrypt version 0.2")
            subtype_count = (await reader.readexactly(1))[0]
            subtypes = [
                await asyncvnc.read_int(reader, 4)
                for _ in range(subtype_count)
            ]
            if 258 not in subtypes:
                raise ValueError(
                    f"server lacks VeNCrypt TLS+VNC support: {subtypes}"
                )
            writer.write((258).to_bytes(4, "big"))
            await writer.drain()
            if (await reader.readexactly(1))[0] != 1:
                raise ValueError("VNC server rejected the selected security type")

            tls_context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
            tls_context.check_hostname = False
            tls_context.verify_mode = ssl.CERT_NONE
            tls_context.maximum_version = ssl.TLSVersion.TLSv1_2
            tls_context.options &= ~0x10
            tls_context.set_ciphers(
                "AECDH-AES256-SHA:AECDH-AES128-SHA:@SECLEVEL=0"
            )
            await writer.start_tls(tls_context)

            if config.password is None:
                raise ValueError("server requires a VNC password")
            challenge = await reader.readexactly(16)
            key = config.password.encode("ascii")[:8].ljust(8, b"\0")
            key = bytes(int(bin(value)[:1:-1].ljust(8, "0"), 2) for value in key)
            with warnings.catch_warnings():
                warnings.simplefilter(
                    "ignore", CryptographyDeprecationWarning
                )
                encryptor = Cipher(TripleDES(key), modes.ECB()).encryptor()
            writer.write(encryptor.update(challenge) + encryptor.finalize())
            await writer.drain()
        elif 2 in auth_types:
            if config.password is None:
                raise ValueError("server requires a VNC password")
            writer.write(b"\x02")
            await writer.drain()
            challenge = await reader.readexactly(16)
            key = config.password.encode("ascii")[:8].ljust(8, b"\0")
            key = bytes(int(bin(value)[:1:-1].ljust(8, "0"), 2) for value in key)
            with warnings.catch_warnings():
                warnings.simplefilter(
                    "ignore", CryptographyDeprecationWarning
                )
                encryptor = Cipher(TripleDES(key), modes.ECB()).encryptor()
            writer.write(encryptor.update(challenge) + encryptor.finalize())
            await writer.drain()
        elif 1 in auth_types:
            writer.write(b"\x01")
            await writer.drain()
        else:
            raise ValueError(f"unsupported VNC authentication types: {auth_types}")

        auth_result = await asyncvnc.read_int(reader, 4)
        if auth_result == 1:
            raise PermissionError("VNC authentication failed")
        if auth_result == 2:
            raise PermissionError("VNC authentication failed (too many attempts)")
        if auth_result != 0:
            raise PermissionError(f"VNC authentication failed: {auth_result}")

        video = await asyncvnc.Video.create(reader, writer)
        await writer.drain()
        client = asyncvnc.Client(
            reader=reader,
            writer=writer,
            clipboard=asyncvnc.Clipboard(writer),
            keyboard=asyncvnc.Keyboard(writer),
            mouse=asyncvnc.Mouse(writer),
            video=video,
            host_key=None,
        )
        yield client
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except ssl.SSLError:
            pass


async def read_hextile_rectangle(
    client: Any,
    target: np.ndarray,
    x: int,
    y: int,
    width: int,
    height: int,
) -> None:
    bytes_per_pixel = 2
    background: bytes | None = None
    foreground: bytes | None = None
    current_subencoding = 0
    current_subrectangle_count = 0

    async def read_pixels(size: int) -> bytes:
        try:
            return await asyncio.wait_for(
                client.reader.readexactly(size),
                timeout=CONNECT_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError as error:
            raise TimeoutError(
                "VNC Hextile update timed out at "
                f"rectangle {x},{y} size {width}x{height}, "
                f"tile {tile_x},{tile_y}, "
                f"subencoding {current_subencoding:#x}, "
                f"subrectangles {current_subrectangle_count}"
            ) from error

    for tile_y in range(y, y + height, 16):
        for tile_x in range(x, x + width, 16):
            tile_width = min(16, x + width - tile_x)
            tile_height = min(16, y + height - tile_y)
            subencoding = (await read_pixels(1))[0]
            current_subencoding = subencoding
            if subencoding & 1:
                pixels = np.frombuffer(
                    await read_pixels(
                        tile_width * tile_height * bytes_per_pixel
                    ),
                    dtype=np.uint8,
                ).reshape(tile_height, tile_width, bytes_per_pixel)
                target[tile_y : tile_y + tile_height, tile_x : tile_x + tile_width] = (
                    decode_rgb565_region(pixels)
                )
                continue

            if subencoding & 2:
                background = await read_pixels(bytes_per_pixel)
            if subencoding & 4:
                foreground = await read_pixels(bytes_per_pixel)
            if background is None:
                raise ValueError("Hextile tile used an undefined background color")
            tile = np.frombuffer(
                background, dtype=np.uint8
            ).reshape(1, 1, bytes_per_pixel)
            target[
                tile_y : tile_y + tile_height,
                tile_x : tile_x + tile_width,
            ] = decode_rgb565_region(tile)

            if not subencoding & 8:
                continue
            subrectangle_count = (await read_pixels(1))[0]
            current_subrectangle_count = subrectangle_count
            for _ in range(subrectangle_count):
                color = (
                    await read_pixels(bytes_per_pixel)
                    if subencoding & 16
                    else foreground
                )
                if color is None:
                    raise ValueError(
                        "Hextile subrectangle used an undefined foreground color"
                    )
                position_and_size = await read_pixels(2)
                sub_x = position_and_size[0] >> 4
                sub_y = position_and_size[0] & 15
                sub_width = (position_and_size[1] >> 4) + 1
                sub_height = (position_and_size[1] & 15) + 1
                target[
                    tile_y + sub_y : tile_y + sub_y + sub_height,
                    tile_x + sub_x : tile_x + sub_x + sub_width,
                ] = decode_rgb565_region(
                    np.frombuffer(color, dtype=np.uint8).reshape(
                        1, 1, bytes_per_pixel
                    )
                )


async def read_tight_rectangle(
    client: Any,
    target: np.ndarray,
    x: int,
    y: int,
    width: int,
    height: int,
    streams: dict[int, Any],
) -> None:
    from PIL import Image

    control = (await client.reader.readexactly(1))[0]
    if control & 0x80:
        tight_type = control >> 4
        if tight_type == 8:
            color = await client.reader.readexactly(2)
            target[y : y + height, x : x + width] = decode_rgb565_region(
                np.frombuffer(color, dtype=np.uint8).reshape(1, 1, 2)
            )
            return
        if tight_type == 9:
            compressed_length = await read_compact_length(client)
            jpeg = await client.reader.readexactly(compressed_length)
            image = Image.open(__import__("io").BytesIO(jpeg)).convert("RGBA")
            target[y : y + height, x : x + width] = np.asarray(image)
            return
        raise ValueError(f"unsupported Tight compression control: {control:#x}")

    stream_id = (control >> 4) & 3
    if control & (1 << stream_id):
        streams.pop(stream_id, None)
    filter_id = 0
    if control & 0x40:
        filter_id = (await client.reader.readexactly(1))[0]

    bytes_per_pixel = 2
    palette: list[bytes] = []
    if filter_id == 1:
        palette_size = (await client.reader.readexactly(1))[0] + 1
        palette = [
            await client.reader.readexactly(bytes_per_pixel)
            for _ in range(palette_size)
        ]

    data_size = width * height
    if filter_id == 0 or filter_id == 2:
        compressed_pixels_size = data_size * bytes_per_pixel
    elif filter_id == 1 and len(palette) == 2:
        compressed_pixels_size = (width + 7) // 8 * height
    elif filter_id == 1:
        compressed_pixels_size = data_size
    else:
        raise ValueError(f"unsupported Tight filter: {filter_id}")

    if compressed_pixels_size < 12:
        decoded = await client.reader.readexactly(compressed_pixels_size)
    else:
        compressed_length = await read_compact_length(client)
        try:
            compressed = await asyncio.wait_for(
                client.reader.readexactly(compressed_length),
                timeout=CONNECT_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError as error:
            raise TimeoutError(
                "VNC Tight update timed out at "
                f"rectangle {x},{y} size {width}x{height}, "
                f"compressed length {compressed_length}"
            ) from error
        decompressor = streams.setdefault(stream_id, zlib.decompressobj())
        decoded = decompressor.decompress(compressed)
    if len(decoded) < compressed_pixels_size:
        raise ValueError(
            "incomplete Tight rectangle: "
            f"expected {compressed_pixels_size}, received {len(decoded)}, "
            f"rectangle {x},{y} size {width}x{height}, "
            f"control {control:#x}, filter {filter_id}"
        )

    if filter_id == 0:
        rgb = np.frombuffer(decoded, dtype=np.uint8).reshape(
            height, width, bytes_per_pixel
        )
    elif filter_id == 2:
        rgb = np.frombuffer(decoded, dtype=np.uint8).reshape(
            height, width, bytes_per_pixel
        ).astype(np.int16)
        for row in range(height):
            for column in range(width):
                left = rgb[row, column - 1] if column else 0
                above = rgb[row - 1, column] if row else 0
                upper_left = rgb[row - 1, column - 1] if row and column else 0
                rgb[row, column] = (rgb[row, column] + left + above - upper_left) & 255
    else:
        indices = np.empty((height, width), dtype=np.uint8)
        encoded = iter(decoded)
        if len(palette) == 2:
            for row in range(height):
                remaining = width
                column = 0
                while remaining:
                    packed = next(encoded)
                    for bit in range(7, -1, -1):
                        if column >= width:
                            break
                        indices[row, column] = (packed >> bit) & 1
                        column += 1
                        remaining -= 1
        else:
            indices[:] = np.frombuffer(decoded, dtype=np.uint8).reshape(height, width)
        rgb = np.frombuffer(
            b"".join(palette[index] for index in indices.ravel()),
            dtype=np.uint8,
        ).reshape(height, width, bytes_per_pixel)

    target[y : y + height, x : x + width] = decode_rgb565_region(rgb)


async def read_compact_length(client: Any) -> int:
    length = 0
    shift = 0
    while True:
        byte = (await client.reader.readexactly(1))[0]
        length |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return length
        shift += 7


def decode_rgb565_region(pixels: np.ndarray) -> np.ndarray:
    values = pixels[..., 0].astype(np.uint16) | (
        pixels[..., 1].astype(np.uint16) << 8
    )
    return np.dstack(
        (
            ((values >> 11) & 31) * 255 // 31,
            ((values >> 5) & 63) * 255 // 63,
            (values & 31) * 255 // 31,
            np.full(values.shape, 255, dtype=np.uint16),
        )
    ).astype(np.uint8)


async def capture_region(
    client: Any,
    framebuffer: np.ndarray,
    request_x: int,
    request_y: int,
    request_width: int,
    request_height: int,
) -> None:
    import asyncvnc

    encodings = (7, 5, 16, 6, 0)
    client.writer.write(
        b"\x00\x00\x00\x00"
        b"\x10\x10\x00\x01"
        b"\x00\x1f\x00\x3f\x00\x1f"
        b"\x0b\x05\x00\x00\x00\x00"
    )
    client.writer.write(
        b"\x02\x00"
        + len(encodings).to_bytes(2, "big")
        + b"".join(encoding.to_bytes(4, "big") for encoding in encodings)
    )
    client.video.data = None
    client.video.refresh(
        request_x, request_y, request_width, request_height
    )
    await client.drain()
    covered = np.zeros((request_height, request_width), dtype=np.bool_)
    tight_streams: dict[int, Any] = {}

    while not covered.all():
        message_type = await asyncvnc.read_int(client.reader, 1)
        if message_type == 2:
            await client.reader.readexactly(3)
            text_length = await asyncvnc.read_int(client.reader, 4)
            await client.reader.readexactly(text_length)
            continue
        if message_type == 3:
            continue
        if message_type != 0:
            raise ValueError(f"unsupported VNC update type: {message_type}")

        await client.reader.readexactly(1)
        rectangle_count = await asyncvnc.read_int(client.reader, 2)
        for _ in range(rectangle_count):
            x = await asyncvnc.read_int(client.reader, 2)
            y = await asyncvnc.read_int(client.reader, 2)
            width = await asyncvnc.read_int(client.reader, 2)
            height = await asyncvnc.read_int(client.reader, 2)
            encoding = await asyncvnc.read_int(client.reader, 4)
            if encoding == 7:
                await read_tight_rectangle(
                    client,
                    framebuffer,
                    x,
                    y,
                    width,
                    height,
                    tight_streams,
                )
            elif encoding == 5:
                await read_hextile_rectangle(
                    client, framebuffer, x, y, width, height
                )
            elif encoding == 0:
                pixels = np.frombuffer(
                    await client.reader.readexactly(width * height * 2),
                    dtype=np.uint8,
                ).reshape(height, width, 2)
                framebuffer[y : y + height, x : x + width] = (
                    decode_rgb565_region(pixels)
                )
            else:
                raise ValueError(
                    f"unsupported framebuffer encoding: {encoding}"
                )
            covered[
                y - request_y : y + height - request_y,
                x - request_x : x + width - request_x,
            ] = True


async def capture_full_desktop(
    config: DesktopConfig,
    state: dict[str, Any] | None = None,
) -> np.ndarray:
    async with open_desktop(config) as first_client:
        width_px = first_client.video.width
        height_px = first_client.video.height
    framebuffer = np.zeros((height_px, width_px, 4), dtype=np.uint8)
    tile_width_px = 640
    tile_height_px = 100
    for request_y in range(0, height_px, tile_height_px):
        for request_x in range(0, width_px, tile_width_px):
            request_width = min(tile_width_px, width_px - request_x)
            request_height = min(tile_height_px, height_px - request_y)
            async with open_desktop(config) as client:
                if state is not None:
                    state["desktop_client"] = client
                await capture_region(
                    client,
                    framebuffer,
                    request_x,
                    request_y,
                    request_width,
                    request_height,
                )
            if state is not None:
                state["desktop_client"] = None
    return framebuffer


async def capture_desktop(
    config: DesktopConfig,
    update_seconds: float,
    frame_queue: queue.Queue[Any],
    stop_event: threading.Event,
    state: dict[str, Any] | None = None,
) -> None:
    def publish(item: Any) -> None:
        try:
            frame_queue.put_nowait(item)
        except queue.Full:
            frame_queue.get_nowait()
            frame_queue.put_nowait(item)

    try:
        publish("Connecting to VNC desktop...")
        while not stop_event.is_set():
            cycle_started = asyncio.get_running_loop().time()
            screenshot = await asyncio.wait_for(
                capture_full_desktop(config, state),
                timeout=max(30.0, update_seconds),
            )
            publish(
                f"Connected to {config.host}; full desktop captured"
            )
            publish(scale_desktop(screenshot))
            await asyncio.sleep(
                max(
                    0.0,
                    update_seconds
                    - (asyncio.get_running_loop().time() - cycle_started),
                )
            )
    except Exception as error:
        publish(RuntimeError(f"desktop connection failed: {error}"))
    finally:
        if state is not None:
            state["desktop_client"] = None


def display_desktop(
    config: DesktopConfig,
    update_seconds: float,
    state: dict[str, Any] | None = None,
) -> None:
    if update_seconds <= 0:
        raise ValueError("update_seconds must be greater than zero")
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError(
            "desktop dependencies are missing; install requirements.txt"
        ) from error

    frame_queue: queue.Queue[Any] = queue.Queue(maxsize=1)
    stop_event = threading.Event()
    worker = threading.Thread(
        target=lambda: asyncio.run(
            capture_desktop(config, update_seconds, frame_queue, stop_event, state)
        ),
        daemon=True,
    )
    worker.start()

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
    axis.text(
        0.5,
        0.5,
        f"Connecting to {config.host}...",
        color="white",
        ha="center",
        va="center",
    )
    figure.patch.set_facecolor("black")
    last_frame = None

    def update_window() -> bool:
        nonlocal last_frame
        try:
            while True:
                item = frame_queue.get_nowait()
                if isinstance(item, str):
                    last_frame = None
                    axis.clear()
                    axis.text(
                        0.5,
                        0.5,
                        item,
                        color="white",
                        ha="center",
                        va="center",
                        wrap=True,
                    )
                    axis.axis("off")
                    figure.canvas.draw_idle()
                elif isinstance(item, Exception):
                    last_frame = None
                    axis.clear()
                    axis.text(
                        0.5,
                        0.5,
                        str(item),
                        color="white",
                        ha="center",
                        va="center",
                        wrap=True,
                    )
                    axis.axis("off")
                    figure.canvas.draw_idle()
                else:
                    last_frame = item
        except queue.Empty:
            pass
        if last_frame is not None:
            axis.clear()
            axis.imshow(last_frame)
            axis.axis("off")
            figure.canvas.draw_idle()
            last_frame = None
        return True

    timer = figure.canvas.new_timer(interval=50)
    timer.add_callback(update_window)
    timer.start()
    try:
        plt.show()
    finally:
        stop_event.set()
        timer.stop()
        worker.join(timeout=1.0)
        plt.close(figure)
