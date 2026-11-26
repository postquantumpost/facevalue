#!/usr/bin/env python3

import argparse
import asyncio
import re
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

alter_adjust_face_template = {
    "Bone Structure" : None
}


alter_facial_balance = {
    "Nose Size" : None,
    "Nose/Forehead Ratio" : None,
    "Face Protrusion" : None,
    "Vert. Face Ratio" : None,
    "Facial Feature Slant" : None,
    "Horiz. Face Ratio" : None
}

alter_forehead_glabella = {
    "Forehead Depth" : None,
    "Forehead Protrusion" : None,
    "Nose Bridge Height" : None,
    "Bridge Protrusion 1" : None,
    "Bridge Protrusion 2" : None,
    "Nose Bridge Width" : None
}

alter_brow_ridge = {
    "Brow Ridge Height" : None,
    "Inner Brow Ridge" : None,
    "Outer Brow Ridge" : None
}

alter_eyes = {
    "Eye Position" : None,
    "Eye Size" : None,
    "Eye Slant" : None,
    "Eye Spacing" : None
}

alter_nose_ridge = {
    "Nose Ridge Depth" : None,
    "Nose Ridge Length" : None,
    "Nose Position" : None,
    "Nose Tip Height" : None,
    "Nose Protrusion" : None,
    "Nose Height" : None,
    "Nose Slant" : None
}

alter_nostrils = {
    "Nostril Slant" : None,
    "Nostril Size" : None,
    "Nostril Width" : None
}

alter_cheeks = {
    "Cheekbone Height" : None,
    "Cheekbone Depth" : None,
    "Cheekbone Width" : None,
    "Cheekbone Protrusion" : None,
    "Cheeks" : None
}

alter_lips = {
    "Lip Shape" : None,
    "Mouth Expression" : None,
    "Lip Fullness" : None,
    "Lip Size" : None,
    "Lip Protrusion" : None,
    "Lip Thickness" : None
}

alter_mouth = {
    "Mouth Protrusion" : None,
    "Mouth Slant" : None,
    "Occlusion" : None,
    "Mouth Position" : None,
    "Mouth Width" : None,
    "Mouth-Chin Distance" : None
}

alter_chin = {
    "Chin Tip Protrusion" : None,
    "Chin Length" : None,
    "Chin Protrusion" : None,
    "Chin Depth" : None,
    "Chin Size" : None,
    "Chin Height" : None,
    "Chin Width" : None
}

alter_jaw = {
    "Jaw Protrusion" : None,
    "Jaw Width" : None,
    "Lower Jaw" : None,
    "Jaw Contour" : None
}

alter_face_structure = {
    "Facial Balance" : alter_facial_balance,
    "Forehead/Glabella" : alter_forehead_glabella,
    "Brow Ridge" : alter_brow_ridge,
    "Eyes" : alter_eyes,
    "Nose Ridge" : alter_nose_ridge,
    "Nostrils" : alter_nostrils,
    "Cheeks" : alter_cheeks,
    "Lips" : alter_lips,
    "Mouth" : alter_mouth,
    "Chin" : alter_chin,
    "Jaw" : alter_jaw
}

alter_face_and_hair_names = {
    "Adjust Face Template" : alter_adjust_face_template,
    "Face Structure" : alter_face_structure,
    "empty0" : None,
    "Hair" : None,
    "Eyebrows" : None,
    "Facial Hair" : None,
    "Eyelashes" : None,
    "Eyes" : None,
    "Skin Features" : None,
    "Cosmetics" : None,
    "Tattoo/Mark/Eyepatch" : None,
    "empty1" : None,
    "Similar Face" : None
}





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


def menu_item_index(menu: dict[str, Any], item_name: str) -> int:
    try:
        return list(menu).index(item_name)
    except ValueError as error:
        raise RuntimeError(f"{item_name} is missing from the menu definition") from error


def menu_item_center(
    screenshot: Any,
    menu: dict[str, Any],
    item_name: str,
) -> tuple[int, int]:
    screen_height, screen_width = screenshot.shape[:2]
    item_index = menu_item_index(menu, item_name)
    nominal_x = nominal_menu_anchor[0] + nominal_slider_start / 2
    nominal_y = (
        nominal_menu_anchor[1]
        + nominal_item_height * item_index
        + nominal_item_height / 2
    )
    return (
        round(
            nominal_x
            * screen_width
            / nominal_screen_width
            * desktop_pointer_scale
        ),
        round(
            nominal_y
            * screen_height
            / nominal_screen_height
            * desktop_pointer_scale
        ),
    )


async def select_menu_item(
    client: Any,
    screenshot: Any,
    menu: dict[str, Any],
    item_name: str,
    expected_heading: str,
) -> tuple[bool, str, Any]:
    x, y = menu_item_center(screenshot, menu, item_name)
    await desk.click_at(client, x, y)
    await asyncio.sleep(1.0)
    updated_screenshot = await desk.capture_full_desktop(client)
    found = desk.region_contains_text(
        updated_screenshot,
        nominal_alter_face_and_hair,
        nominal_screen_width,
        nominal_screen_height,
        expected_heading,
    )
    if not found:
        return (
            False,
            f"Unable to select {item_name}; exiting.",
            updated_screenshot,
        )
    return True, f"Verified menu {expected_heading}.", updated_screenshot


async def return_to_menu(
    client: Any,
    expected_heading: str,
    excluded_heading: str,
) -> tuple[bool, str, Any]:
    await desk.press_key(client, "BackSpace")
    await asyncio.sleep(1.0)
    screenshot = await desk.capture_full_desktop(client)
    expected_found = desk.region_contains_text(
        screenshot,
        nominal_alter_face_and_hair,
        nominal_screen_width,
        nominal_screen_height,
        expected_heading,
    )
    excluded_found = desk.region_contains_text(
        screenshot,
        nominal_alter_face_and_hair,
        nominal_screen_width,
        nominal_screen_height,
        excluded_heading,
    )
    if expected_found and not excluded_found:
        return True, f"Returned to {expected_heading}.", screenshot
    return False, f"Backspace did not return to {expected_heading}; exiting.", screenshot


async def verify_desktop_controls(
    client: Any,
    screenshot: Any,
) -> tuple[bool, str, Any]:
    selected, message, updated_screenshot = await select_menu_item(
        client,
        screenshot,
        alter_face_and_hair_names,
        "Adjust Face Template",
        "Alter Face & Hair > Adjust Face Template",
    )
    if not selected:
        return False, message, updated_screenshot
    print(
        "Verified mouse control by selecting Adjust Face Template.",
        flush=True,
    )

    returned, message, restored_screenshot = await return_to_menu(
        client,
        "Alter Face & Hair",
        "Adjust Face Template",
    )
    if not returned:
        return False, message, restored_screenshot
    print("Verified keyboard control with Backspace.", flush=True)

    selected, message, face_structure_screenshot = await select_menu_item(
        client,
        restored_screenshot,
        alter_face_and_hair_names,
        "Face Structure",
        "Alter Face & Hair > Face Structure",
    )
    if not selected:
        return False, message, face_structure_screenshot
    print(message, flush=True)

    selected, message, balance_screenshot = await select_menu_item(
        client,
        face_structure_screenshot,
        alter_face_structure,
        "Facial Balance",
        "Alter Face & Hair > Face Structure > Facial Balance",
    )
    if not selected:
        return False, message, balance_screenshot
    print(message, flush=True)

    if alter_facial_balance.get("Nose Size", object()) is not None:
        return False, "Nose Size is not a leaf menu item; exiting.", balance_screenshot
    nose_index = menu_item_index(alter_facial_balance, "Nose Size")
    item_top = nominal_menu_anchor[1] + nominal_item_height * nose_index
    label_region = [
        nominal_menu_anchor[0],
        item_top,
        nominal_menu_anchor[0] + nominal_slider_start,
        item_top + nominal_item_height,
    ]
    if not desk.region_contains_text(
        balance_screenshot,
        label_region,
        nominal_screen_width,
        nominal_screen_height,
        "Nose Size",
    ):
        return False, "Nose Size is not displayed; exiting.", balance_screenshot

    number_region = [
        nominal_menu_anchor[0] + nominal_number_start,
        item_top,
        nominal_menu_anchor[0] + nominal_item_width,
        item_top + nominal_item_height,
    ]
    number_text = desk.ocr_region_text(
        balance_screenshot,
        number_region,
        nominal_screen_width,
        nominal_screen_height,
        "-+0123456789",
    )
    match = re.search(r"[+-]?\d+", number_text)
    if match is None:
        return False, f"Unable to read Nose Size as an integer: {number_text!r}; exiting.", balance_screenshot
    nose_size = int(match.group())
    print(f"Nose Size: {nose_size}", flush=True)

    returned, message, structure_screenshot = await return_to_menu(
        client,
        "Alter Face & Hair > Face Structure",
        "Facial Balance",
    )
    if not returned:
        return False, message, structure_screenshot
    print(message, flush=True)

    returned, message, root_screenshot = await return_to_menu(
        client,
        "Alter Face & Hair",
        "Face Structure",
    )
    if not returned:
        return False, message, root_screenshot
    return True, f"{message} Continuing desktop monitoring.", root_screenshot




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
