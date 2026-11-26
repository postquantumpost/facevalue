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

# UI location info
# Nominal screen size
nominal_screen_width: int = 3840
nominal_screen_height: int = 2160
# VNC reports Retina framebuffer pixels, but pointer coordinates use logical pixels.
desktop_pointer_scale: float = 0.5
# The following screen locations are based on the
# nominal screen size defined above. Scale them according
# to the actual screen size at runtime.
# The coordinates are in the format [ x1, y1, x2, y2 ] with
# zero being the top-left corner of the screen
nominal_cosmetics = [ 231, 82, 507, 145 ]
nominal_alter_face_and_hair = [ 684, 241, 2000, 295 ]
# x1, y1
nominal_menu_anchor = [ 270, 322 ]
# For a menu item the bounding box is width x height. The
# text label space is the left side of the bounding box
# to the slider start. The slider space is from slider
# start to number start. The number is from number start
# to the end of the bounding box.
nominal_item_width = 1726
nominal_item_height = 112
nominal_slider_start = 700
nominal_number_start = 1585

alter_face_and_hair_names = [
    "Adjust Face Template",
    "Face Structure",
    "",
    "Hair",
    "Eyebrows",
    "Facial Hair",
    "Eyelashes",
    "Eyes",
    "Skin Features",
    "Cosmetics",
    "Tattoo/Mark/Eyepatch",
    "",
    "Similar Face"
]

alter_face_structure = [
    "Facial Balance",
    "Forehead/Glabella",
    "Brow Ridge",
    "Eyes",
    "Nose Ridge",
    "Nostrils",
    "Cheeks",
    "Lips",
    "Mouth",
    "Chin",
    "Jaw"
]

alter_facial_balance = [
    "Nose Size",
    "Nose/Forehead Ratio",
    "Face Protrusion",
    "Vert. Face Ratio",
    "Facial Feature Slant",
    "Horiz. Face Ratio"
]

alter_forehead_glabella = [
    "Forehead Depth",
    "Forehead Protrusion",
    "Nose Bridge Height",
    "Bridge Protrusion 1",
    "Bridge Protrusion 2",
    "Nose Bridge Width"
]

alter_brow_ridge = [
    "Brow Ridge Height",
    "Inner Brow Ridge",
    "Outer Brow Ridge"
]

alter_eyes = [
    "Eye Position",
    "Eye Size",
    "Eye Slant",
    "Eye Spacing"
]

alter_nose_ridge = [
    "Nose Ridge Depth",
    "Nose Ridge Length",
    "Nose Position",
    "Nose Tip Height",
    "Nose Protrusion",
    "Nose Height",
    "Nose Slant"
]

alter_nostrils = [
    "Nostril Slant",
    "Nostril Size",
    "Nostril Width"
]

alter_cheeks = [
    "Cheekbone Height",
    "Cheekbone Depth",
    "Cheekbone Width",
    "Cheekbone Protrusion",
    "Cheeks"
]

alter_lips = [
    "Lip Shape",
    "Mouth Expression",
    "Lip Fullness",
    "Lip Size",
    "Lip Protrusion",
    "Lip Thickness"
]

alter_mouth = [
    "Mouth Protrusion",
    "Mouth Slant",
    "Occlusion",
    "Mouth Position",
    "Mouth Width",
    "Mouth-Chin Distance"
]

alter_chin = [
    "Chin Tip Protrusion",
    "Chin Length",
    "Chin Protrusion",
    "Chin Depth",
    "Chin Size",
    "Chin Height",
    "Chin Width"
]

alter_jaw = [
    "Jaw Protrusion",
    "Jaw Width",
    "Lower Jaw",
    "Jaw Contour"
]


def verify_desktop_display(screenshot: Any) -> tuple[bool, str]:
    cosmetics_found = desk.region_contains_text(
        screenshot,
        nominal_cosmetics,
        nominal_screen_width,
        nominal_screen_height,
        "Cosmetics",
    )
    if not cosmetics_found:
        return False, "Cosmetics is not displayed; exiting."
    print("Verified Cosmetics is displayed.", flush=True)

    alter_face_and_hair_found = desk.region_contains_text(
        screenshot,
        nominal_alter_face_and_hair,
        nominal_screen_width,
        nominal_screen_height,
        "Alter Face & Hair",
    )
    if alter_face_and_hair_found:
        return (
            True,
            "Verified Alter Face & Hair is displayed; continuing desktop monitoring.",
        )
    return False, "Alter Face & Hair is not displayed; exiting."


async def verify_desktop_controls(
    client: Any,
    screenshot: Any,
) -> tuple[bool, str, Any]:
    try:
        item_index = alter_face_and_hair_names.index("Adjust Face Template")
    except ValueError as error:
        raise RuntimeError(
            "Adjust Face Template is missing from the menu definition"
        ) from error

    screen_height, screen_width = screenshot.shape[:2]
    nominal_x = nominal_menu_anchor[0] + nominal_slider_start / 2
    nominal_y = (
        nominal_menu_anchor[1]
        + nominal_item_height * item_index
        + nominal_item_height / 2
    )
    x = round(
        nominal_x
        * screen_width
        / nominal_screen_width
        * desktop_pointer_scale
    )
    y = round(
        nominal_y
        * screen_height
        / nominal_screen_height
        * desktop_pointer_scale
    )

    await desk.click_at(client, x, y)
    await asyncio.sleep(1.0)
    updated_screenshot = await desk.capture_full_desktop(client)
    found = desk.region_contains_text(
        updated_screenshot,
        nominal_alter_face_and_hair,
        nominal_screen_width,
        nominal_screen_height,
        "Alter Face & Hair > Adjust Face Template",
    )
    if not found:
        return (
            False,
            "Unable to click Adjust Face Template; exiting.",
            updated_screenshot,
        )
    print(
        "Verified mouse control by selecting Adjust Face Template.",
        flush=True,
    )

    await desk.press_key(client, "BackSpace")
    await asyncio.sleep(1.0)
    restored_screenshot = await desk.capture_full_desktop(client)
    root_heading_present = desk.region_contains_text(
        restored_screenshot,
        nominal_alter_face_and_hair,
        nominal_screen_width,
        nominal_screen_height,
        "Alter Face & Hair",
    )
    submenu_heading_present = desk.region_contains_text(
        restored_screenshot,
        nominal_alter_face_and_hair,
        nominal_screen_width,
        nominal_screen_height,
        "Adjust Face Template",
    )
    if root_heading_present and not submenu_heading_present:
        return (
            True,
            "Verified keyboard control with Backspace; "
            "continuing desktop monitoring.",
            restored_screenshot,
        )
    return (
        False,
        "Backspace did not return to the start menu; exiting.",
        restored_screenshot,
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
        desk.display_desktop(
            config,
            update_seconds,
            state=state,
            startup_screen_check=verify_desktop_display,
            startup_control_check=verify_desktop_controls,
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
