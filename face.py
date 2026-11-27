#!/usr/bin/env python3

import json
import os
import tempfile
import urllib.request
from pathlib import Path
from typing import Any


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


class FaceDetectionError(ValueError):
    pass


def supported_images(directory: Path) -> list[Path]:
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
    return images


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
        raise FaceDetectionError(f"no face was detected in: {image_path}")

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

    try:
        results = DeepFace.represent(
            img_path=str(image_path),
            model_name=DEEPFACE_MODEL_NAME,
            detector_backend="opencv",
            enforce_detection=True,
            max_faces=1,
        )
    except ValueError as error:
        raise FaceDetectionError(str(error)) from error
    if not results:
        raise FaceDetectionError(f"DeepFace did not detect a face in: {image_path}")

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
    image_results: list[dict[str, Any]],
) -> dict[str, Any]:
    try:
        import numpy as np
    except ImportError as error:
        raise RuntimeError(
            "numeric dependencies are missing; install requirements.txt"
        ) from error

    image_records = [
        {
            "id": result["id"],
            "path": str(result["path"]),
            "status": result["status"],
            **({"error": result["error"]} if result["status"] == "failed" else {}),
        }
        for result in image_results
    ]
    successful_results = [
        result for result in image_results if result["status"] == "ok"
    ]
    geometries = [result["mediapipe"] for result in successful_results]
    deepface_embeddings = [
        result["deepface"] for result in successful_results
    ]
    for result, geometry, deepface_embedding in zip(
        successful_results, geometries, deepface_embeddings, strict=True
    ):
        geometry["image_id"] = result["id"]
        deepface_embedding["image_id"] = result["id"]

    if not successful_results:
        return {
            "images": image_records,
            "deepface": {"embeddings": []},
            "mediapipe": {"geometry": []},
            "consensus": {
                "id": "consensus",
                "status": "unavailable",
                "source_image_ids": [],
                "message": "No faces were recognized; consensus was not generated.",
            },
        }

    deepface_vectors = np.asarray(
        [record["embedding"] for record in deepface_embeddings]
    )
    vector_norms = np.linalg.norm(deepface_vectors, axis=1)
    if np.any(vector_norms == 0):
        raise ValueError("cannot create a consensus from a zero DeepFace embedding")
    normalized_deepface_vectors = deepface_vectors / vector_norms[:, np.newaxis]
    mean_deepface_vector = normalized_deepface_vectors.mean(axis=0)
    consensus_deepface_norm = np.linalg.norm(mean_deepface_vector)
    if consensus_deepface_norm == 0:
        raise ValueError("DeepFace embeddings cancel during consensus generation")
    mean_deepface_vector /= consensus_deepface_norm
    deepface_cosine_similarities = normalized_deepface_vectors @ mean_deepface_vector

    aligned_geometry = np.asarray(
        [
            [
                (landmark["x"], landmark["y"], landmark["z"])
                for landmark in geometry["face_forward"]["landmarks"]
            ]
            for geometry in geometries
        ]
    )
    if aligned_geometry.shape[0:2] != (
        len(geometries),
        aligned_geometry.shape[1],
    ):
        raise ValueError("MediaPipe landmarks cannot be aligned for consensus")
    consensus_landmarks_array = aligned_geometry.mean(axis=0)
    landmark_variance_array = aligned_geometry.var(axis=0)
    consensus_landmarks = [
        {"x": float(point[0]), "y": float(point[1]), "z": float(point[2])}
        for point in consensus_landmarks_array
    ]
    landmark_variance = [
        {"x": float(point[0]), "y": float(point[1]), "z": float(point[2])}
        for point in landmark_variance_array
    ]
    source_point_deviations = np.linalg.norm(
        aligned_geometry - consensus_landmarks_array, axis=2
    )

    return {
        "images": image_records,
        "deepface": {"embeddings": deepface_embeddings},
        "mediapipe": {"geometry": geometries},
        "consensus": {
            "id": "consensus",
            "status": "ok",
            "source_image_ids": [
                result["id"] for result in successful_results
            ],
            "deepface": {
                "method": "l2-normalized-mean",
                "model": deepface_embeddings[0]["model"],
                "dimension_count": len(mean_deepface_vector),
                "embedding": mean_deepface_vector.tolist(),
                "source_cosine_similarity": [
                    {
                        "image_id": image_record["id"],
                        "value": float(similarity),
                    }
                    for image_record, similarity in zip(
                        successful_results,
                        deepface_cosine_similarities,
                        strict=True,
                    )
                ],
                "mean_source_cosine_similarity": float(
                    deepface_cosine_similarities.mean()
                ),
            },
            "mediapipe": {
                "geometry": {
                    "method": "mean-aligned-procrustes-landmarks",
                    "landmark_count": len(consensus_landmarks),
                    "landmarks": consensus_landmarks,
                    "landmark_variance": landmark_variance,
                    "mean_source_point_deviation": float(
                        source_point_deviations.mean()
                    ),
                    "max_source_point_deviation": float(
                        source_point_deviations.max()
                    ),
                },
                "embedding": {
                    "method": "flattened-consensus-landmarks",
                    "dimension_count": len(consensus_landmarks) * 3,
                    "embedding": [
                        coordinate
                        for landmark in consensus_landmarks
                        for coordinate in (
                            landmark["x"],
                            landmark["y"],
                            landmark["z"],
                        )
                    ],
                },
                "source_transforms": [
                    {
                        "image_id": image_record["id"],
                        "method": geometry["face_forward"]["method"],
                        "scale": geometry["face_forward"]["scale"],
                        "rms_anchor_error": geometry["face_forward"][
                            "rms_anchor_error"
                        ],
                        "anchor_landmark_indices": geometry["face_forward"][
                            "anchor_landmark_indices"
                        ],
                        "transform_matrix": geometry["face_forward"][
                            "transform_matrix"
                        ],
                    }
                    for image_record, geometry in zip(
                        successful_results, geometries, strict=True
                    )
                ],
            },
        },
    }


def write_output(output_path: Path, data: dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(data, output_file, indent=2)
        output_file.write("\n")


def read_input(input_path: Path) -> dict[str, Any]:
    if not input_path.is_file():
        raise FileNotFoundError(f"input JSON file not found: {input_path}")
    try:
        with input_path.open("r", encoding="utf-8") as input_file:
            data = json.load(input_file)
    except json.JSONDecodeError as error:
        raise ValueError(f"input file is not valid JSON: {input_path}") from error

    if not isinstance(data, dict):
        raise ValueError("input JSON root must be an object")
    required_sections = ("images", "deepface", "mediapipe", "consensus")
    missing_sections = [
        section for section in required_sections if section not in data
    ]
    if missing_sections:
        raise ValueError(
            "input JSON is missing required sections: "
            + ", ".join(missing_sections)
        )
    return data


def fit_axis_text(figure: Any, minimum_font_size: float = 4.0) -> None:
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()

    def fit_text_artist(
        text_artist: Any,
        maximum_width: float,
        maximum_height: float,
    ) -> None:
        if not text_artist.get_text():
            return
        for _ in range(20):
            bounds = text_artist.get_window_extent(renderer=renderer)
            width_scale = maximum_width / max(bounds.width, 1.0)
            height_scale = maximum_height / max(bounds.height, 1.0)
            scale = min(width_scale, height_scale, 1.0)
            if scale >= 0.99:
                break
            next_size = max(
                minimum_font_size,
                text_artist.get_fontsize() * scale * 0.95,
            )
            if abs(next_size - text_artist.get_fontsize()) < 0.05:
                break
            text_artist.set_fontsize(next_size)

    for axis in figure.axes:
        axis_bounds = axis.get_window_extent(renderer=renderer)
        if axis_bounds.width <= 0 or axis_bounds.height <= 0:
            continue
        fit_text_artist(
            axis.title,
            axis_bounds.width * 0.94,
            axis_bounds.height * 0.20,
        )
        fit_text_artist(
            axis.xaxis.label,
            axis_bounds.width * 0.90,
            axis_bounds.height * 0.12,
        )
        fit_text_artist(
            axis.yaxis.label,
            axis_bounds.height * 0.35,
            axis_bounds.width * 0.12,
        )
        for text_artist in axis.texts:
            fit_text_artist(
                text_artist,
                axis_bounds.width * 0.92,
                axis_bounds.height * 0.80,
            )

        for tick_label in axis.get_xticklabels():
            tick_label.set_fontsize(
                min(tick_label.get_fontsize(), 6.0)
            )
        for tick_label in axis.get_yticklabels():
            tick_label.set_fontsize(
                min(tick_label.get_fontsize(), 6.0)
            )

        legend = axis.get_legend()
        if legend is not None:
            for text_artist in legend.get_texts():
                fit_text_artist(
                    text_artist,
                    axis_bounds.width * 0.35,
                    axis_bounds.height * 0.10,
                )
            legend_font_sizes = [
                text_artist.get_fontsize()
                for text_artist in legend.get_texts()
            ]
            if legend_font_sizes:
                legend_size = min(legend_font_sizes, default=6.0)
                for text_artist in legend.get_texts():
                    text_artist.set_fontsize(legend_size)
    figure.canvas.draw_idle()


def update_figure(figure: Any, data: dict[str, Any]) -> Any:
    try:
        import numpy as np
        from matplotlib.animation import FuncAnimation
    except ImportError as error:
        raise RuntimeError(
            "visualization dependencies are missing; install requirements.txt"
        ) from error

    from PIL import Image

    if hasattr(figure, "_facevalue_animation"):
        figure._facevalue_animation.event_source.stop()
    figure.clear()
    column_count = len(data["images"]) + 1
    rotating_axes = []
    geometry_by_id = {
        record["image_id"]: record for record in data["mediapipe"]["geometry"]
    }
    deepface_by_id = {
        record["image_id"]: record for record in data["deepface"]["embeddings"]
    }

    for column, image_record in enumerate(data["images"], start=1):
        image_axis = figure.add_subplot(5, column_count, column)
        geometry_axis = figure.add_subplot(
            5, column_count, column_count + column
        )
        embedding_axis = figure.add_subplot(
            5, column_count, 2 * column_count + column
        )
        rotating_axis = figure.add_subplot(
            5,
            column_count,
            3 * column_count + column,
            projection="3d",
        )
        face_forward_axis = figure.add_subplot(
            5,
            column_count,
            4 * column_count + column,
            projection="3d",
        )
        if image_record["status"] != "ok":
            image_axis.set_title(Path(image_record["path"]).name)
            image_axis.text(
                0.5,
                0.5,
                f"Recognition failed\n{image_record['error']}",
                ha="center",
                va="center",
                wrap=True,
            )
            for axis in (
                image_axis,
                geometry_axis,
                embedding_axis,
                rotating_axis,
                face_forward_axis,
            ):
                axis.axis("off")
            continue

        geometry = geometry_by_id[image_record["id"]]
        deepface_record = deepface_by_id[image_record["id"]]
        image_info = geometry["image"]
        image_path = Path(image_info["path"])
        image_available = image_path.is_file()
        if image_available:
            with Image.open(image_path) as source_image:
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

        rotating_axes.append(rotating_axis)

        image_axis.set_title(
            f"{image_record['id']}: {Path(image_record['path']).name}"
        )
        if image_available:
            image_axis.imshow(image_rgb)
        else:
            image_axis.text(
                0.5,
                0.5,
                "Source image unavailable",
                ha="center",
                va="center",
            )
        image_axis.axis("off")

        if image_available:
            geometry_axis.imshow(image_rgb)
            geometry_axis.scatter(x_pixels, y_pixels, c="red", s=3)
        else:
            geometry_axis.scatter(
                np.asarray([point["x"] for point in landmarks]),
                np.asarray([point["y"] for point in landmarks]),
                c="red",
                s=3,
            )
            geometry_axis.invert_yaxis()
            geometry_axis.set_aspect("equal")
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

    consensus = data["consensus"]
    consensus_column = column_count
    consensus_source_axis = figure.add_subplot(
        5, column_count, consensus_column
    )
    consensus_geometry_axis = figure.add_subplot(
        5, column_count, column_count + consensus_column
    )
    consensus_embedding_axis = figure.add_subplot(
        5, column_count, 2 * column_count + consensus_column
    )
    consensus_rotating_axis = figure.add_subplot(
        5,
        column_count,
        3 * column_count + consensus_column,
        projection="3d",
    )
    consensus_forward_axis = figure.add_subplot(
        5,
        column_count,
        4 * column_count + consensus_column,
        projection="3d",
    )
    if consensus["status"] != "ok":
        consensus_source_axis.text(
            0.5,
            0.5,
            consensus["message"],
            ha="center",
            va="center",
            wrap=True,
        )
        for axis in (
            consensus_source_axis,
            consensus_geometry_axis,
            consensus_embedding_axis,
            consensus_rotating_axis,
            consensus_forward_axis,
        ):
            axis.axis("off")
        figure.subplots_adjust(
            left=0.04, right=0.99, bottom=0.04, top=0.95,
            wspace=0.3, hspace=0.45,
        )
        fit_axis_text(figure)
        return None

    consensus_geometry = consensus["mediapipe"]["geometry"]
    consensus_landmarks = consensus_geometry["landmarks"]
    consensus_x = np.asarray([point["x"] for point in consensus_landmarks])
    consensus_y = np.asarray([point["y"] for point in consensus_landmarks])
    consensus_z = np.asarray([point["z"] for point in consensus_landmarks])
    consensus_deepface = np.asarray(consensus["deepface"]["embedding"])
    consensus_mediapipe = np.asarray(
        consensus["mediapipe"]["embedding"]["embedding"]
    )

    consensus_source_axis.set_title("Consensus\nno source image")
    consensus_source_axis.text(
        0.5,
        0.5,
        "\n".join(consensus["source_image_ids"]),
        ha="center",
        va="center",
    )
    consensus_source_axis.axis("off")
    consensus_geometry_axis.scatter(consensus_x, consensus_z, c="red", s=3)
    consensus_geometry_axis.set_title("Consensus 2D geometry")
    consensus_geometry_axis.set_aspect("equal")
    consensus_geometry_axis.axis("off")
    consensus_embedding_axis.plot(
        np.linspace(0, 1, len(consensus_deepface)),
        consensus_deepface,
        label=f"DeepFace {consensus['deepface']['model']}",
    )
    consensus_embedding_axis.plot(
        np.linspace(0, 1, len(consensus_mediapipe)),
        consensus_mediapipe,
        label="MediaPipe consensus",
    )
    consensus_embedding_axis.set_title("Consensus embeddings")
    consensus_embedding_axis.set_xlabel("Normalized vector position")
    consensus_embedding_axis.set_ylabel("Value")
    consensus_embedding_axis.legend()
    consensus_rotating_axis.scatter(
        consensus_x,
        consensus_y,
        consensus_z,
        c=np.asarray(range(len(consensus_landmarks))),
        cmap="viridis",
        s=3,
    )
    consensus_rotating_axis.set_title("Rotating consensus 3D face points")
    consensus_rotating_axis.set_box_aspect((1, 1, 1))
    consensus_rotating_axis.set_axis_off()
    rotating_axes.append(consensus_rotating_axis)
    consensus_forward_axis.scatter(
        consensus_x,
        consensus_y,
        consensus_z,
        c=np.asarray(range(len(consensus_landmarks))),
        cmap="viridis",
        s=3,
    )
    consensus_forward_axis.view_init(elev=0, azim=-90)
    consensus_forward_axis.set_title("Consensus face-forward points")
    consensus_forward_axis.set_box_aspect((1, 1, 1))
    consensus_forward_axis.set_axis_off()

    def rotate(frame: int) -> tuple[Any, ...]:
        for rotating_axis in rotating_axes:
            rotating_axis.view_init(
                elev=10,
                azim=(frame * 4.0) % 360.0,
            )
        return ()

    animation = FuncAnimation(
        figure,
        rotate,
        frames=90,
        interval=20,
        repeat=True,
        cache_frame_data=False,
        blit=False,
    )
    figure._facevalue_animation = animation
    figure.subplots_adjust(
        left=0.04, right=0.99, bottom=0.04, top=0.95,
        wspace=0.3, hspace=0.45,
    )
    fit_axis_text(figure)
    animation.event_source.start()
    return animation


def show_window(
    data: dict[str, Any],
    block: bool = True,
) -> Any:
    try:
        import tkinter as tk
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure
    except ImportError as error:
        raise RuntimeError(
            "visualization dependencies are missing; install requirements.txt"
        ) from error

    window = tk.Tk()
    window.title("FaceValue")
    window.geometry("1024x768")
    window.minsize(1, 1)
    figure = Figure(figsize=(10.24, 7.68), dpi=100)
    canvas = FigureCanvasTkAgg(figure, master=window)
    canvas.draw()
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    figure._facevalue_tk_window = window
    figure._facevalue_canvas = canvas
    update_figure(figure, data)
    if block:
        window.mainloop()
        window.destroy()
    return figure


def run(
    fdir: Path | None,
    input_path: Path | None,
    output_path: Path | None,
    window: bool,
) -> None:
    if input_path is not None:
        data = read_input(input_path)
        if output_path is not None:
            write_output(output_path, data)
        if window:
            show_window(data)
        return
    if fdir is None:
        return
    image_results = []
    for index, image_path in enumerate(supported_images(fdir), start=1):
        result: dict[str, Any] = {
            "id": f"image-{index}",
            "path": image_path,
            "status": "ok",
        }
        try:
            result["mediapipe"] = extract_mediapipe_geometry(image_path)
            result["deepface"] = extract_deepface_embedding(image_path)
        except FaceDetectionError as error:
            result["status"] = "failed"
            result["error"] = str(error)
            result.pop("mediapipe", None)
            result.pop("deepface", None)
        image_results.append(result)
    data = build_output(image_results)
    if output_path is not None:
        write_output(output_path, data)
    if window:
        show_window(data)


def add_display_only_image(
    data: dict[str, Any],
    image_path: Path,
    image_id: str = "desktop-face",
) -> None:
    result: dict[str, Any] = {
        "id": image_id,
        "path": str(image_path),
        "status": "ok",
    }
    try:
        result["mediapipe"] = extract_mediapipe_geometry(image_path)
        result["deepface"] = extract_deepface_embedding(image_path)
    except FaceDetectionError as error:
        result["status"] = "failed"
        result["error"] = str(error)

    data["images"][:] = [
        record for record in data["images"] if record["id"] != image_id
    ]
    data["images"].append(
        {
            "id": result["id"],
            "path": result["path"],
            "status": result["status"],
            **({"error": result["error"]} if result["status"] == "failed" else {}),
        }
    )
    if result["status"] == "ok":
        result["mediapipe"]["image_id"] = image_id
        result["deepface"]["image_id"] = image_id
        data["mediapipe"]["geometry"][:] = [
            record
            for record in data["mediapipe"]["geometry"]
            if record["image_id"] != image_id
        ]
        data["deepface"]["embeddings"][:] = [
            record
            for record in data["deepface"]["embeddings"]
            if record["image_id"] != image_id
        ]
        data["mediapipe"]["geometry"].append(result["mediapipe"])
        data["deepface"]["embeddings"].append(result["deepface"])


def add_display_only_placeholder(
    data: dict[str, Any],
    image_id: str = "desktop-face",
) -> None:
    data["images"][:] = [
        record for record in data["images"] if record["id"] != image_id
    ]
    data["images"].append(
        {
            "id": image_id,
            "path": "Waiting for desktop face",
            "status": "failed",
            "error": "Waiting for the first desktop frame",
        }
    )
