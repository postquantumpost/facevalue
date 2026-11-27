#!/usr/bin/env python3

import argparse
import asyncio
import copy
import queue
import random
import re
import tempfile
from pathlib import Path
from typing import Any, Sequence

import desk
import face


update_seconds: float = 10.0
testmode: bool = True
state: dict[str, Any] = {
    "desktop_client": None,
    "desktop_config": None,
}
current_face_value_state: dict[str, int] = {}
desktop_consensus_match: float | None = None
system_random = random.SystemRandom()

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
    "Chin Tip Position" : None,
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

    heading = desk.ocr_region_text(
        screenshot,
        nominal_alter_face_and_hair,
        nominal_screen_width,
        nominal_screen_height,
    )
    alter_face_and_hair_found = (
        re.sub(r"[^a-z0-9]", "", heading.casefold())
        == "alterfacehair"
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


def analyze_desktop_face(
    data: dict[str, Any],
    desktop_face_path: Path,
) -> float:
    face.add_display_only_image(data, desktop_face_path)
    return face.desktop_consensus_similarity(data)


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


def focused_menu_index(screenshot: Any, menu_size: int) -> int:
    import numpy as np

    if menu_size <= 0:
        raise ValueError("menu must contain at least one item")
    height_px, width_px = screenshot.shape[:2]
    brightness_by_row: list[float] = []
    left = round(
        (nominal_menu_anchor[0] + nominal_slider_start)
        * width_px
        / nominal_screen_width
    )
    right = round(
        (nominal_menu_anchor[0] + nominal_number_start)
        * width_px
        / nominal_screen_width
    )
    for index in range(menu_size):
        top = (
            nominal_menu_anchor[1]
            + nominal_item_height * index
        )
        bottom = top + nominal_item_height
        y1 = round((top + nominal_item_height * 0.25) * height_px / nominal_screen_height)
        y2 = round((bottom - nominal_item_height * 0.25) * height_px / nominal_screen_height)
        row = np.asarray(screenshot)[y1:y2, left:right, :3]
        if row.size == 0:
            raise ValueError("menu rows are outside the screenshot")
        luminance = (
            row[..., 0] * 0.2126
            + row[..., 1] * 0.7152
            + row[..., 2] * 0.0722
        )
        brightness_by_row.append(float(np.median(luminance)))
    return max(range(menu_size), key=brightness_by_row.__getitem__)


def navigable_menu_items(menu: dict[str, Any]) -> list[str]:
    return [
        item_name
        for item_name in menu
        if re.fullmatch(r"empty\d+", item_name) is None
    ]


async def focus_menu_item(
    client: Any,
    screenshot: Any,
    menu: dict[str, Any],
    item_name: str,
) -> tuple[bool, str, Any]:
    navigable_items = navigable_menu_items(menu)
    current_screenshot = screenshot
    raw_items = list(menu)
    for _ in range(4):
        current_raw_index = focused_menu_index(current_screenshot, len(menu))
        current_item_name = raw_items[current_raw_index]
        if current_item_name == item_name:
            return True, f"Focused {item_name}.", current_screenshot
        target_raw_index = raw_items.index(item_name)
        if target_raw_index > current_raw_index:
            key = "Down"
            steps = sum(
                item in navigable_items
                for item in raw_items[current_raw_index + 1 : target_raw_index + 1]
            )
        else:
            key = "Up"
            steps = sum(
                item in navigable_items
                for item in raw_items[target_raw_index:current_raw_index]
            )
        for _ in range(steps):
            await desk.press_key(client, key)
            await asyncio.sleep(0.1)
        await asyncio.sleep(0.25)
        current_screenshot = await desk.capture_full_desktop(client)
    return (
        False,
        f"Unable to focus {item_name} with keyboard navigation; exiting.",
        current_screenshot,
    )


def recognized_menu_depth(heading: str) -> int | None:
    headings = {
        "Alter Face & Hair": 0,
        "Alter Face & Hair > Adjust Face Template": 1,
        "Alter Face & Hair > Face Structure": 1,
    }
    for submenu_name in alter_face_structure:
        headings[
            f"Alter Face & Hair > Face Structure > {submenu_name}"
        ] = 2
    normalized = " ".join(heading.split())
    return headings.get(normalized)


async def recover_desktop_menu(
    client: Any,
    screenshot: Any,
) -> Any | None:
    heading = desk.ocr_region_text(
        screenshot,
        nominal_alter_face_and_hair,
        nominal_screen_width,
        nominal_screen_height,
    )
    depth = recognized_menu_depth(heading)
    if depth is None:
        print(f"Desktop menu is unrecognized: {heading!r}", flush=True)
        return None
    for level in range(depth, 0, -1):
        await desk.press_key(client, "q")
        await asyncio.sleep(0.5)
        screenshot = await desk.capture_full_desktop(client)
        recovered_heading = desk.ocr_region_text(
            screenshot,
            nominal_alter_face_and_hair,
            nominal_screen_width,
            nominal_screen_height,
        )
        print(
            f"Recovering desktop menu level {level}: {recovered_heading}",
            flush=True,
        )
    final_heading = desk.ocr_region_text(
        screenshot,
        nominal_alter_face_and_hair,
        nominal_screen_width,
        nominal_screen_height,
    )
    if recognized_menu_depth(final_heading) == 0:
        return screenshot
    return None


async def select_menu_item(
    client: Any,
    screenshot: Any,
    menu: dict[str, Any],
    item_name: str,
    expected_heading: str,
) -> tuple[bool, str, Any]:
    updated_screenshot = screenshot
    for _ in range(3):
        focused, _, updated_screenshot = await focus_menu_item(
            client,
            updated_screenshot,
            menu,
            item_name,
        )
        if not focused:
            continue
        await desk.press_key(client, "e")
        await asyncio.sleep(1.0)
        updated_screenshot = await desk.capture_full_desktop(client)
        if desk.region_contains_text(
            updated_screenshot,
            nominal_alter_face_and_hair,
            nominal_screen_width,
            nominal_screen_height,
            expected_heading,
        ):
            return True, f"Verified menu {expected_heading}.", updated_screenshot
    return (
        False,
        f"Unable to select {item_name}; exiting.",
        updated_screenshot,
    )


async def return_to_menu(
    client: Any,
    expected_heading: str,
    excluded_heading: str,
) -> tuple[bool, str, Any]:
    await desk.press_key(client, "q")
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
    return False, f"Q did not return to {expected_heading}; exiting.", screenshot


def read_leaf_integer(
    screenshot: Any,
    menu: dict[str, Any],
    item_name: str,
) -> tuple[int | None, str]:
    item_index = menu_item_index(menu, item_name)
    item_top = nominal_menu_anchor[1] + nominal_item_height * item_index
    label_region = [
        nominal_menu_anchor[0],
        item_top,
        nominal_menu_anchor[0] + nominal_slider_start,
        item_top + nominal_item_height,
    ]
    if not desk.region_contains_text(
        screenshot,
        label_region,
        nominal_screen_width,
        nominal_screen_height,
        item_name,
    ):
        return None, f"{item_name} is not displayed; exiting."

    number_region = [
        nominal_menu_anchor[0] + nominal_number_start,
        item_top,
        nominal_menu_anchor[0] + nominal_item_width,
        item_top + nominal_item_height,
    ]
    number_text = desk.ocr_region_text(
        screenshot,
        number_region,
        nominal_screen_width,
        nominal_screen_height,
        "-+0123456789",
    )
    match = re.search(r"[+-]?\d+", number_text)
    if match is None:
        return (
            None,
            f"Unable to read {item_name} as an integer: {number_text!r}; exiting.",
        )
    return int(match.group()), ""


async def read_all_face_structure_values(
    client: Any,
    root_screenshot: Any,
) -> tuple[bool, str, Any]:
    selected, message, structure_screenshot = await select_menu_item(
        client,
        root_screenshot,
        alter_face_and_hair_names,
        "Face Structure",
        "Alter Face & Hair > Face Structure",
    )
    if not selected:
        return False, message, structure_screenshot

    values: dict[str, int] = {}
    submenus = (
        {"Facial Balance": alter_facial_balance}
        if testmode
        else alter_face_structure
    )
    for submenu_name, submenu in submenus.items():
        if not isinstance(submenu, dict):
            return (
                False,
                f"{submenu_name} is not a Face Structure submenu; exiting.",
                structure_screenshot,
            )
        selected, message, submenu_screenshot = await select_menu_item(
            client,
            structure_screenshot,
            alter_face_structure,
            submenu_name,
            f"Alter Face & Hair > Face Structure > {submenu_name}",
        )
        if not selected:
            return False, message, submenu_screenshot

        for item_name, child in submenu.items():
            if child is not None:
                return (
                    False,
                    f"{item_name} unexpectedly contains another menu; exiting.",
                    submenu_screenshot,
                )
            value, error_message = read_leaf_integer(
                submenu_screenshot,
                submenu,
                item_name,
            )
            if value is None:
                await asyncio.sleep(0.5)
                retry_screenshot = await desk.capture_full_desktop(client)
                retry_value, retry_error = read_leaf_integer(
                    retry_screenshot,
                    submenu,
                    item_name,
                )
                if retry_value is not None:
                    submenu_screenshot = retry_screenshot
                    value = retry_value
                    error_message = retry_error
            if value is None:
                return False, error_message, submenu_screenshot
            values[item_name] = value

        returned, message, structure_screenshot = await return_to_menu(
            client,
            "Alter Face & Hair > Face Structure",
            submenu_name,
        )
        if not returned:
            return False, message, structure_screenshot

    returned, message, root_screenshot = await return_to_menu(
        client,
        "Alter Face & Hair",
        "Face Structure",
    )
    if not returned:
        return False, message, root_screenshot

    current_face_value_state.clear()
    current_face_value_state.update(values)
    print(current_face_value_state, flush=True)
    return True, "Read all Face Structure values.", root_screenshot


def face_structure_leaf_paths() -> dict[str, tuple[str, dict[str, Any]]]:
    paths: dict[str, tuple[str, dict[str, Any]]] = {}
    submenus = (
        {"Facial Balance": alter_facial_balance}
        if testmode
        else alter_face_structure
    )
    for submenu_name, submenu in submenus.items():
        if not isinstance(submenu, dict):
            continue
        for item_name, child in submenu.items():
            if child is None:
                paths[item_name] = (submenu_name, submenu)
    return paths


async def modify_random_face_values(
    client: Any,
    root_screenshot: Any,
    modification_count: int = 1,
) -> tuple[bool, str, Any]:
    if not current_face_value_state:
        return False, "No face values are available to modify; exiting.", root_screenshot
    leaf_paths = face_structure_leaf_paths()
    item_names = tuple(leaf_paths)
    if not set(item_names).issubset(current_face_value_state):
        scope = "Facial Balance" if testmode else "Face Structure"
        return (
            False,
            f"{scope} values have not all been read; exiting.",
            root_screenshot,
        )
    current_screenshot = root_screenshot

    for iteration in range(modification_count):
        item_name = system_random.choice(item_names)
        delta = system_random.choice((-1, 1))
        changed, message, current_screenshot, actual_delta = (
            await modify_face_value(
                client,
                current_screenshot,
                item_name,
                delta,
            )
        )
        if not changed and message:
            return False, message, current_screenshot
        old_value = current_face_value_state[item_name] - actual_delta
        print(
            f"Modified {item_name}: previous value {old_value}, "
            f"new value {current_face_value_state[item_name]} "
            f"[{iteration + 1}/{modification_count}]",
            flush=True,
        )

    return (
        True,
        f"Completed {modification_count} random face value modifications.",
        current_screenshot,
    )


async def modify_face_value(
    client: Any,
    root_screenshot: Any,
    item_name: str,
    requested_delta: int,
) -> tuple[bool, str, Any, int]:
    paths = face_structure_leaf_paths()
    try:
        submenu_name, submenu = paths[item_name]
    except KeyError:
        return (
            False,
            f"{item_name} is missing from the Face Structure tree; exiting.",
            root_screenshot,
            0,
        )

    selected, message, structure_screenshot = await select_menu_item(
        client,
        root_screenshot,
        alter_face_and_hair_names,
        "Face Structure",
        "Alter Face & Hair > Face Structure",
    )
    if not selected:
        return False, message, structure_screenshot, 0
    selected, message, submenu_screenshot = await select_menu_item(
        client,
        structure_screenshot,
        alter_face_structure,
        submenu_name,
        f"Alter Face & Hair > Face Structure > {submenu_name}",
    )
    if not selected:
        return False, message, submenu_screenshot, 0

    focused, message, submenu_screenshot = await focus_menu_item(
        client,
        submenu_screenshot,
        submenu,
        item_name,
    )
    if not focused:
        return False, message, submenu_screenshot, 0
    await desk.press_key(client, "e")
    await desk.press_key(
        client,
        "Right" if requested_delta > 0 else "Left",
    )
    await desk.press_key(client, "e")
    await asyncio.sleep(1.0)
    modified_screenshot = await desk.capture_full_desktop(client)
    new_value, error_message = read_leaf_integer(
        modified_screenshot,
        submenu,
        item_name,
    )
    if new_value is None:
        return False, error_message, modified_screenshot, 0
    old_value = current_face_value_state[item_name]
    actual_delta = new_value - old_value
    current_face_value_state[item_name] = new_value

    returned, message, structure_screenshot = await return_to_menu(
        client,
        "Alter Face & Hair > Face Structure",
        submenu_name,
    )
    if not returned:
        return False, message, structure_screenshot, actual_delta
    returned, message, root_screenshot = await return_to_menu(
        client,
        "Alter Face & Hair",
        "Face Structure",
    )
    if not returned:
        return False, message, root_screenshot, actual_delta
    return True, "", root_screenshot, actual_delta


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
        "Verified keyboard control by selecting Adjust Face Template.",
        flush=True,
    )

    returned, message, restored_screenshot = await return_to_menu(
        client,
        "Alter Face & Hair",
        "Adjust Face Template",
    )
    if not returned:
        return False, message, restored_screenshot
    print("Verified keyboard control with Q.", flush=True)

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
    print(message, flush=True)

    read, message, final_screenshot = await read_all_face_structure_values(
        client,
        root_screenshot,
    )
    if not read:
        return False, message, final_screenshot

    modified, message, modified_screenshot = await modify_random_face_values(
        client,
        final_screenshot,
        1,
    )
    if not modified:
        return False, message, modified_screenshot
    print(current_face_value_state, flush=True)
    return True, message, modified_screenshot




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
    parser.add_argument(
        "--testmode",
        type=parse_boolean,
        default=True,
        metavar="true|false",
        help="limit desktop operations to Facial Balance (default: true)",
    )
    arguments = parser.parse_args(argv)
    for option_name in ("input", "output", "desktop"):
        path = getattr(arguments, option_name)
        if path is not None and path.suffix.lower() != ".json":
            parser.error(f"--{option_name} must name a file with a .json extension")

    if arguments.input is not None and arguments.fdir is not None:
        parser.error("--input and --fdir are mutually exclusive")
    if arguments.desktop is not None and arguments.fdir is not None:
        parser.error("--desktop and --fdir cannot be used together")
    if (
        arguments.desktop is not None
        and arguments.input is not None
        and not arguments.window
    ):
        parser.error("--desktop with --input requires --window true")
    return arguments


def run(arguments: argparse.Namespace) -> None:
    global desktop_consensus_match, testmode
    testmode = getattr(arguments, "testmode", True)
    desktop_consensus_match = None
    if arguments.desktop is not None:
        config = desk.read_config(arguments.desktop)
        state["desktop_config"] = config
        if arguments.testmode and arguments.input is None and not arguments.window:
            try:
                asyncio.run(
                    desk.run_desktop_sequence(
                        config,
                        state=state,
                        startup_screen_check=verify_desktop_display,
                        startup_control_check=verify_desktop_controls,
                        startup_recovery_check=recover_desktop_menu,
                    )
                )
            except RuntimeError as error:
                print(error, flush=True)
            return
        if arguments.input is not None:
            data = face.read_input(arguments.input)
            from PIL import Image

            with tempfile.TemporaryDirectory() as temporary_directory:
                face.add_display_only_placeholder(data)
                face_figure = face.show_window(data, block=False)
                completed_updates: queue.Queue[Any] = queue.Queue(maxsize=1)
                capture_number = 0

                def update_face_feedback(_: Any) -> None:
                    try:
                        updated_data = completed_updates.get_nowait()
                    except queue.Empty:
                        updated_data = None
                    if updated_data is not None:
                        data.clear()
                        data.update(updated_data)
                        if arguments.output is not None:
                            face.write_output(arguments.output, data)
                        face.update_figure(face_figure, data)

                async def process_face(
                    screenshot: Any,
                ) -> float:
                    nonlocal capture_number
                    capture_number += 1
                    capture_path = (
                        Path(temporary_directory)
                        / f"desktop-face-{capture_number}.png"
                    )
                    _, width_px = screenshot.shape[:2]
                    desktop_face = screenshot[
                        :, round(width_px * 0.52) :, :3
                    ]
                    Image.fromarray(desktop_face).save(capture_path)
                    analysis_data = copy.deepcopy(data)
                    match_score = await asyncio.to_thread(
                        analyze_desktop_face,
                        analysis_data,
                        capture_path,
                    )
                    print(
                        f"Desktop face captured and processed: {capture_path.name}",
                        flush=True,
                    )
                    try:
                        completed_updates.put_nowait(analysis_data)
                    except queue.Full:
                        completed_updates.get_nowait()
                        completed_updates.put_nowait(analysis_data)
                    return match_score

                async def optimize_once(
                    client: Any,
                    screenshot: Any,
                ) -> Any:
                    global desktop_consensus_match
                    if desktop_consensus_match is None:
                        match_score = await process_face(screenshot)
                        desktop_consensus_match = match_score
                        print(
                            f"Initial desktop/consensus match: {match_score:.6f}",
                            flush=True,
                        )
                        return screenshot

                    item_names = tuple(face_structure_leaf_paths())
                    item_name = system_random.choice(item_names)
                    requested_delta = system_random.choice((-1, 1))
                    ok, message, changed_screenshot, actual_delta = (
                        await modify_face_value(
                            client,
                            screenshot,
                            item_name,
                            requested_delta,
                        )
                    )
                    if not ok:
                        raise RuntimeError(message)
                    previous_value = (
                        current_face_value_state[item_name] - actual_delta
                    )
                    changed_value = current_face_value_state[item_name]
                    print(
                        f"{item_name} changed: previous value "
                        f"{previous_value}, new value {changed_value}",
                        flush=True,
                    )
                    if actual_delta == 0:
                        print(
                            f"No change to {item_name}: kept value "
                            f"{changed_value} and match value "
                            f"{desktop_consensus_match:.6f}",
                            flush=True,
                        )
                        return changed_screenshot
                    candidate_score = await process_face(changed_screenshot)
                    previous_match = desktop_consensus_match
                    print(
                        f"Face match: previous value {previous_match:.6f}, "
                        f"new value {candidate_score:.6f}",
                        flush=True,
                    )
                    if candidate_score < previous_match:
                        reverted, revert_message, reverted_screenshot, revert_delta = (
                            await modify_face_value(
                                client,
                                changed_screenshot,
                                item_name,
                                -actual_delta,
                            )
                        )
                        if not reverted:
                            raise RuntimeError(revert_message)
                        reverted_value = current_face_value_state[item_name]
                        print(
                            f"{item_name} reverted to value {reverted_value} "
                            f"(change {revert_delta:+d}); "
                            f"reverted to match value "
                            f"{desktop_consensus_match:.6f}",
                            flush=True,
                        )
                        return reverted_screenshot

                    desktop_consensus_match = candidate_score
                    print(
                        f"Kept {item_name}: value {changed_value}, "
                        f"kept match value {desktop_consensus_match:.6f}",
                        flush=True,
                    )
                    return changed_screenshot

                def close_all_windows() -> None:
                    stop_event = state.get("desktop_stop_event")
                    if stop_event is not None:
                        stop_event.set()
                    try:
                        face_figure._facevalue_tk_window.destroy()
                    except Exception:
                        pass

                face_figure._facevalue_tk_window.protocol(
                    "WM_DELETE_WINDOW",
                    close_all_windows,
                )

                try:
                    desk.display_desktop(
                        config,
                        update_seconds,
                        state=state,
                        startup_screen_check=verify_desktop_display,
                        startup_control_check=verify_desktop_controls,
                        on_screenshot=update_face_feedback,
                        continuing_control_check=optimize_once,
                        startup_recovery_check=recover_desktop_menu,
                    )
                finally:
                    try:
                        face_figure._facevalue_tk_window.destroy()
                    except Exception:
                        pass
            return
        if arguments.window and arguments.testmode:
            desk.display_desktop(
                config,
                update_seconds,
                state=state,
                startup_screen_check=verify_desktop_display,
                startup_control_check=verify_desktop_controls,
                stop_after_startup=True,
                startup_recovery_check=recover_desktop_menu,
            )
            return
        desk.display_desktop(
            config,
            update_seconds,
            state=state,
            startup_screen_check=verify_desktop_display,
            startup_control_check=verify_desktop_controls,
            startup_recovery_check=recover_desktop_menu,
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
