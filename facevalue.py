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
DEEPFACE_MODEL_NAME = "SFace"
PROCRUSTES_ANCHOR_INDICES = (263, 33, 10, 152, 1, 168)
PROCRUSTES_TARGET = (
    (1.0, 0.0, 0.0),
    (-1.0, 0.0, 0.0),
    (0.0, 0.0, 1.5),
    (0.0, 0.0, -1.5),
    (0.0, -0.45, -0.15),
    (0.0, -0.15, 0.5),
)


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


def first_images(directory: Path, count: int = 2) -> list[Path]:
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
    if len(images) < count:
        raise FileNotFoundError(
            f"expected at least {count} supported images in {directory}; "
            f"found {len(images)}"
        )
    return images[:count]


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


def procrustes_face_forward(landmarks: list[dict[str, float]]) -> dict[str, Any]:
    try:
        import numpy as np
    except ImportError as error:
        raise RuntimeError(
            "MediaPipe dependencies are missing; install requirements.txt"
        ) from error

    required_index = max(PROCRUSTES_ANCHOR_INDICES)
    if len(landmarks) <= required_index:
        raise ValueError(
            "not enough MediaPipe landmarks for Procrustes face alignment"
        )

    points = np.asarray(
        [
            (landmark["x"], landmark["z"], -landmark["y"])
            for landmark in landmarks
        ]
    )
    source_anchors = points[list(PROCRUSTES_ANCHOR_INDICES)]
    target_anchors = np.asarray(PROCRUSTES_TARGET)
    source_center = source_anchors.mean(axis=0)
    target_center = target_anchors.mean(axis=0)
    centered_source = source_anchors - source_center
    centered_target = target_anchors - target_center

    left_vectors, singular_values, right_vectors = np.linalg.svd(
        centered_source.T @ centered_target
    )
    reflection_signs = np.ones(3)
    if np.linalg.det(left_vectors @ right_vectors) < 0:
        reflection_signs[-1] = -1
    rotation = left_vectors @ np.diag(reflection_signs) @ right_vectors
    source_squared_scale = np.sum(centered_source * centered_source)
    if source_squared_scale == 0:
        raise ValueError("cannot align degenerate MediaPipe face geometry")
    scale = np.sum(singular_values * reflection_signs) / source_squared_scale
    translation = target_center - scale * source_center @ rotation
    aligned_points = scale * points @ rotation + translation
    aligned_anchors = aligned_points[list(PROCRUSTES_ANCHOR_INDICES)]
    residual = np.sqrt(
        np.mean(np.sum((aligned_anchors - target_anchors) ** 2, axis=1))
    )

    transform = np.eye(4)
    transform[:3, :3] = scale * rotation.T
    transform[:3, 3] = translation
    aligned_landmarks = [
        {"x": float(point[0]), "y": float(point[1]), "z": float(point[2])}
        for point in aligned_points
    ]

    return {
        "method": "3d-similarity-procrustes-fixed-frontal-template",
        "anchor_landmark_indices": list(PROCRUSTES_ANCHOR_INDICES),
        "target_anchor_points": [
            {"x": point[0], "y": point[1], "z": point[2]}
            for point in PROCRUSTES_TARGET
        ],
        "landmark_count": len(aligned_landmarks),
        "landmarks": aligned_landmarks,
        "embedding": [
            coordinate
            for landmark in aligned_landmarks
            for coordinate in (landmark["x"], landmark["y"], landmark["z"])
        ],
        "scale": float(scale),
        "rms_anchor_error": float(residual),
        "transform_matrix": transform.tolist(),
    }


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
    procrustes = procrustes_face_forward(landmarks)

    return {
        "landmark_count": len(landmarks),
        "landmarks": landmarks,
        "embedding": embedding,
        "facial_transformation_matrixes": transformation_matrixes,
        "face_forward": procrustes,
        "image": {
            "path": str(image_path),
            "width_px": int(image_rgb.shape[1]),
            "height_px": int(image_rgb.shape[0]),
        },
    }


def extract_deepface_embedding(image_path: Path) -> dict[str, Any]:
    try:
        from deepface import DeepFace
    except ImportError as error:
        raise RuntimeError(
            "DeepFace dependencies are missing; install requirements.txt"
        ) from error

    results = DeepFace.represent(
        img_path=str(image_path),
        model_name=DEEPFACE_MODEL_NAME,
        detector_backend="opencv",
        enforce_detection=True,
        max_faces=1,
    )
    if not results:
        raise ValueError(f"DeepFace did not detect a face in: {image_path}")

    result = results[0]
    return {
        "model": DEEPFACE_MODEL_NAME,
        "dimension_count": len(result["embedding"]),
        "embedding": [float(value) for value in result["embedding"]],
        "face_confidence": result["face_confidence"],
        "facial_area": result["facial_area"],
        "image_path": str(image_path),
    }


def build_output(
    geometries: list[dict[str, Any]],
    deepface_embeddings: list[dict[str, Any]],
) -> dict[str, Any]:
    image_records = []
    for index, (geometry, deepface_embedding) in enumerate(
        zip(geometries, deepface_embeddings, strict=True), start=1
    ):
        image_id = f"image-{index}"
        geometry["image_id"] = image_id
        deepface_embedding["image_id"] = image_id
        image_records.append(
            {
                "id": image_id,
                "path": geometry["image"]["path"],
            }
        )

    return {
        "images": image_records,
        "deepface": {"embeddings": deepface_embeddings},
        "mediapipe": {"geometry": geometries},
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
        from matplotlib.animation import FuncAnimation
    except ImportError as error:
        raise RuntimeError(
            "visualization dependencies are missing; install requirements.txt"
        ) from error

    from PIL import Image

    figure = plt.figure(figsize=(16, 20))
    figure.canvas.manager.set_window_title("FaceValue")
    rotating_axes = []

    for column, (image_record, geometry, deepface_record) in enumerate(
        zip(
            data["images"],
            data["mediapipe"]["geometry"],
            data["deepface"]["embeddings"],
            strict=True,
        ),
        start=1,
    ):
        image_info = geometry["image"]
        with Image.open(image_info["path"]) as source_image:
            image_rgb = np.asarray(source_image.convert("RGB"))

        landmarks = geometry["landmarks"]
        x_pixels = (
            np.asarray([point["x"] for point in landmarks]) * image_info["width_px"]
        )
        y_pixels = (
            np.asarray([point["y"] for point in landmarks]) * image_info["height_px"]
        )
        mediapipe_embedding = np.asarray(geometry["embedding"])
        deepface_embedding = np.asarray(deepface_record["embedding"])
        x_points = np.asarray([point["x"] for point in landmarks])
        depth_points = np.asarray([point["z"] for point in landmarks])
        vertical_points = np.asarray([-point["y"] for point in landmarks])

        image_axis = figure.add_subplot(5, 2, column)
        geometry_axis = figure.add_subplot(5, 2, 2 + column)
        embedding_axis = figure.add_subplot(5, 2, 4 + column)
        rotating_axis = figure.add_subplot(5, 2, 6 + column, projection="3d")
        face_forward_axis = figure.add_subplot(
            5, 2, 8 + column, projection="3d"
        )
        rotating_axes.append(rotating_axis)

        image_axis.imshow(image_rgb)
        image_axis.set_title(
            f"{image_record['id']}: {Path(image_record['path']).name}"
        )
        image_axis.axis("off")

        geometry_axis.imshow(image_rgb)
        geometry_axis.scatter(x_pixels, y_pixels, c="red", s=3)
        geometry_axis.set_title("MediaPipe 2D geometry")
        geometry_axis.axis("off")

        embedding_axis.plot(
            np.linspace(0, 1, len(deepface_embedding)),
            deepface_embedding,
            label=f"DeepFace {deepface_record['model']}",
        )
        embedding_axis.plot(
            np.linspace(0, 1, len(mediapipe_embedding)),
            mediapipe_embedding,
            label="MediaPipe geometry",
        )
        embedding_axis.set_title("Face embeddings")
        embedding_axis.set_xlabel("Normalized vector position")
        embedding_axis.set_ylabel("Value")
        embedding_axis.legend()

        rotating_axis.scatter(
            x_points,
            depth_points,
            vertical_points,
            c=np.asarray(range(len(landmarks))),
            cmap="viridis",
            s=3,
        )
        rotating_axis.set_title("Rotating 3D face points")
        rotating_axis.set_box_aspect((1, 1, 1))
        rotating_axis.set_axis_off()

        face_forward_landmarks = geometry["face_forward"]["landmarks"]
        face_forward_axis.scatter(
            [point["x"] for point in face_forward_landmarks],
            [point["y"] for point in face_forward_landmarks],
            [point["z"] for point in face_forward_landmarks],
            c=np.asarray(range(len(face_forward_landmarks))),
            cmap="viridis",
            s=3,
        )
        face_forward_axis.view_init(elev=0, azim=-90)
        face_forward_axis.set_title("Procrustes face-forward points")
        face_forward_axis.set_box_aspect((1, 1, 1))
        face_forward_axis.set_axis_off()

    def rotate(frame: int) -> tuple[Any, ...]:
        for rotating_axis in rotating_axes:
            rotating_axis.view_init(elev=10, azim=frame * 0.5)
        return ()

    animation = FuncAnimation(
        figure,
        rotate,
        frames=720,
        interval=50,
        repeat=True,
        cache_frame_data=False,
        blit=False,
    )
    plt.tight_layout()
    figure.canvas.draw()
    plt.show()
    plt.close(figure)


def run(arguments: argparse.Namespace) -> None:
    if arguments.fdir is None:
        return

    image_paths = first_images(arguments.fdir)
    geometries = []
    deepface_embeddings = []
    for image_path in image_paths:
        geometries.append(extract_mediapipe_geometry(image_path))
        deepface_embeddings.append(extract_deepface_embedding(image_path))
    data = build_output(geometries, deepface_embeddings)
    if arguments.output is not None:
        write_output(arguments.output, data)
    if arguments.window:
        show_window(data)


def main(argv: Sequence[str] | None = None) -> int:
    run(parse_arguments(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
