#!/usr/bin/env python3

import argparse
import json
import os
import tempfile
import urllib.request
from pathlib import Path
from typing import Any, Sequence


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
MODEL_CACHE_DIRECTORY = Path(
    os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")
) / "facevalue"


def parse_boolean(value: str) -> bool:
    normalized = value.lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise argparse.ArgumentTypeError("boolean values must be 'true' or 'false'")


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract and visualize face geometry from an image directory."
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
        "--window",
        type=parse_boolean,
        default=False,
        metavar="true|false",
        help="display a grid of extracted face information (default: false)",
    )
    arguments = parser.parse_args(argv)
    if arguments.output is not None and arguments.output.suffix.lower() != ".json":
        parser.error("--output must name a file with a .json extension")
    return arguments


def first_image(directory: Path) -> Path:
    if not directory.is_dir():
        raise NotADirectoryError(f"face image directory not found: {directory}")

    images = sorted(
        (
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ),
        key=lambda path: path.name.casefold(),
    )
    if not images:
        raise FileNotFoundError(f"no supported images found in: {directory}")
    return images[0]


def model_path() -> Path:
    destination = MODEL_CACHE_DIRECTORY / "face_landmarker.task"
    if destination.is_file():
        return destination

    MODEL_CACHE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=MODEL_CACHE_DIRECTORY) as temporary_dir:
        download_path = Path(temporary_dir) / destination.name
        try:
            urllib.request.urlretrieve(MODEL_URL, download_path)
        except Exception as error:
            raise RuntimeError(
                f"could not download MediaPipe face model from {MODEL_URL}"
            ) from error
        download_path.replace(destination)
    return destination


def extract_mediapipe_geometry(image_path: Path) -> dict[str, Any]:
    try:
        import cv2
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import (
            FaceLandmarker,
            FaceLandmarkerOptions,
        )
    except ImportError as error:
        raise RuntimeError(
            "MediaPipe dependencies are missing; install requirements.txt"
        ) from error

    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise ValueError(f"could not read image: {image_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path())),
        num_faces=1,
        output_facial_transformation_matrixes=True,
    )
    with FaceLandmarker.create_from_options(options) as landmarker:
        result = landmarker.detect(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        )

    if not result.face_landmarks:
        raise ValueError(f"no face was detected in: {image_path}")

    face = result.face_landmarks[0]
    landmarks = [
        {"x": landmark.x, "y": landmark.y, "z": landmark.z}
        for landmark in face
    ]
    embedding = [
        coordinate
        for landmark in landmarks
        for coordinate in (landmark["x"], landmark["y"], landmark["z"])
    ]
    transformation_matrixes = [
        matrix.tolist()
        for matrix in result.facial_transformation_matrixes
    ]

    return {
        "landmark_count": len(landmarks),
        "landmarks": landmarks,
        "embedding": embedding,
        "facial_transformation_matrixes": transformation_matrixes,
        "image": {
            "path": str(image_path),
            "width_px": int(image_rgb.shape[1]),
            "height_px": int(image_rgb.shape[0]),
        },
    }


def build_output(geometry: dict[str, Any]) -> dict[str, Any]:
    return {
        "deepface": {"embeddings": []},
        "mediapipe": {"geometry": [geometry]},
    }


def write_output(output_path: Path, data: dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(data, output_file, indent=2)
        output_file.write("\n")


def show_window(data: dict[str, Any]) -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as error:
        raise RuntimeError(
            "visualization dependencies are missing; install requirements.txt"
        ) from error

    geometry = data["mediapipe"]["geometry"][0]
    image_info = geometry["image"]
    image_path = Path(image_info["path"])

    from PIL import Image

    with Image.open(image_path) as source_image:
        image_rgb = np.asarray(source_image.convert("RGB"))

    landmarks = geometry["landmarks"]
    x_pixels = np.asarray([point["x"] for point in landmarks]) * image_info["width_px"]
    y_pixels = np.asarray([point["y"] for point in landmarks]) * image_info["height_px"]
    embedding = np.asarray(geometry["embedding"])

    figure, axes = plt.subplots(1, 3, figsize=(15, 5))
    figure.canvas.manager.set_window_title("FaceValue")

    axes[0].imshow(image_rgb)
    axes[0].set_title("Source image")
    axes[0].axis("off")

    axes[1].imshow(image_rgb)
    axes[1].scatter(x_pixels, y_pixels, c="red", s=3)
    axes[1].set_title("MediaPipe geometry")
    axes[1].axis("off")

    normalized = embedding.reshape(1, -1)
    axes[2].imshow(normalized, cmap="viridis", aspect="auto")
    axes[2].set_yticks([])
    axes[2].set_xlabel("MediaPipe normalized coordinates")
    if data["deepface"]["embeddings"]:
        axes[2].set_title("DeepFace / MediaPipe embeddings")
    else:
        axes[2].set_title("MediaPipe embedding\n(DeepFace pending)")
    plt.tight_layout()
    plt.show()
    plt.close(figure)


def run(arguments: argparse.Namespace) -> None:
    if arguments.fdir is None:
        return

    image_path = first_image(arguments.fdir)
    data = build_output(extract_mediapipe_geometry(image_path))
    if arguments.output is not None:
        write_output(arguments.output, data)
    if arguments.window:
        show_window(data)


def main(argv: Sequence[str] | None = None) -> int:
    run(parse_arguments(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
