# -*- coding: utf-8 -*-
# --- Imports and Setup ---
import os
import tempfile
import uuid
import zipfile
import shutil
from lxml import etree
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Tuple, Optional, Generator, List, Dict, Any, Callable, Set
import copy # Needed for deep copying elements
import re # For parsing adjustment values
from pathlib import Path # For easier path manipulation


# --- Global Namespaces and Constants ---
NSMAP = {
    "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "c": "http://schemas.openxmlformats.org/drawingml/2006/chart",          # Chart namespace
    "dgm": "http://schemas.openxmlformats.org/drawingml/2006/diagram",      # Diagram (SmartArt) namespace
    "dsp": "http://schemas.microsoft.com/office/drawing/2008/diagram",      # Diagram Shapes (SmartArt) namespace
    "rel": "http://schemas.openxmlformats.org/package/2006/relationships"   # Relationships namespace
}

# URIs for identifying graphic types and relationships
CHART_URI = "http://schemas.openxmlformats.org/drawingml/2006/chart"
DIAGRAM_URI = "http://schemas.openxmlformats.org/drawingml/2006/diagram"
CHART_REL_TYPE = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/chart"

DEFAULT_SLIDE_WIDTH_EMU = 9144000  # Default width in EMUs (English Metric Units)
DEFAULT_SLIDE_HEIGHT_EMU = 6858000 # Default height in EMUs

# Define known drawable element tags to process at the top level of spTree
DRAWABLE_ELEMENT_TAGS = {
    etree.QName(NSMAP['p'], 'sp'),           # Shape (including text boxes)
    etree.QName(NSMAP['p'], 'pic'),          # Picture
    etree.QName(NSMAP['p'], 'cxnSp'),        # Connector Shape
    etree.QName(NSMAP['p'], 'graphicFrame'), # Graphic Frame (charts, tables, diagrams/SmartArt)
    etree.QName(NSMAP['p'], 'grpSp'),        # Group Shape
    etree.QName(NSMAP['p'], 'contentPart')   # Content Part (e.g., embedded objects)
}

# Default processing options for the UI checkboxes
DEFAULT_OPTIONS = {
    "mirror_shapes_pos": True,           # Mirror position of shapes/text boxes
    "mirror_pictures_pos": True,         # Mirror position of pictures
    "mirror_connectors_pos": True,       # Mirror position of connectors
    "mirror_groups_pos": True,           # Mirror position of groups
    "mirror_frames_pos": True,           # Mirror position of graphic frames (charts, tables, diagrams)
    "apply_flipH_to_shapes_no_text": True, # Apply horizontal flip to shapes (CAUTION: affects text)
    "apply_flipH_to_pictures": False,      # Apply horizontal flip to pictures
    "apply_flipH_to_connectors": True,     # Apply horizontal flip to connectors (Experimental)
    "apply_flipH_to_groups": True,         # Apply horizontal flip to the group container
    "flip_table_columns": True,           # Flip table columns (Experimental - Improved)
    "flip_chart_internals": True,         # NEW: Attempt to flip internal chart elements (Experimental)
    "apply_flipH_to_smartart_frame": True,# NEW: Apply flipH to SmartArt frame (Experimental - High Risk)
    "right_align_text": True,              # Attempt to right-align text
    "set_rtl_paragraphs": True,           # Set the RTL attribute on paragraphs
    "process_masters_layouts": True,       # Process slide masters and layouts as well
    "aggressive_paragraph_rebuild": False, # Optional aggressive rebuild (Risky - Default OFF)
    "experimental_preset_flip": False,     # Experimental flip for specific presets
    "aggressive_shape_rebuild": False,      # Aggressive rebuild for shapes
    "replace_text": True,                 # Replace all text content
    "set_arabic_proofing_lang": True      # Set proofing language to Arabic
}


# --- Core Layout Flipping Logic ---

def get_slide_size_from_presentation(xml_root: etree._Element) -> Tuple[int, int]:
    """
    Extracts slide width and height in EMUs from the presentation.xml root element.
    This version ensures width and height are returned in the correct order.
    """
    try:
        sldSz_elems = xml_root.xpath("./p:sldSz", namespaces=NSMAP)
        if not sldSz_elems:
            sldSz_elems = xml_root.xpath("//p:presentation/p:sldSz", namespaces=NSMAP)

        if sldSz_elems:
            sldSz = sldSz_elems[0]
            cx_attr = sldSz.get("cx") # Width attribute
            cy_attr = sldSz.get("cy") # Height attribute
            if cx_attr and cy_attr:
                slide_width_emu = int(cx_attr)
                slide_height_emu = int(cy_attr)
                logging.info(f"Detected slide size: Width={slide_width_emu}, Height={slide_height_emu} EMU")
                # *** CRITICAL FIX: Ensure width is returned first, then height ***
                return slide_width_emu, slide_height_emu
            else:
                logging.warning("Slide size 'cx' or 'cy' attributes missing in <p:sldSz>.")
        else:
            logging.warning("Slide size element <p:sldSz> not found in presentation.xml.")

    except (ValueError, TypeError, IndexError, etree.XPathEvalError) as e:
        logging.warning(f"Could not parse slide size from presentation.xml, using defaults. Error: {e}")

    logging.warning(f"Using default slide size: Width={DEFAULT_SLIDE_WIDTH_EMU}, Height={DEFAULT_SLIDE_HEIGHT_EMU} EMU")
    return DEFAULT_SLIDE_WIDTH_EMU, DEFAULT_SLIDE_HEIGHT_EMU

def mirror_x_around_center(off_x: int, ext_cx: int, parent_width: int) -> int:
    """
    Calculates the mirrored X-offset for an element based on its current offset,
    width (extent), and the width of its parent container (slide width).
    Mirrors the element's center point relative to the parent's center point.
    """
    # Basic validation
    if parent_width <= 0:
        logging.warning(f"Cannot mirror: parent_width={parent_width} is invalid. Returning original offset.")
        return off_x
    if ext_cx < 0:
        # Handle cases where element width might be negative (shouldn't happen but be safe)
        logging.warning(f"Element width is negative ({ext_cx}), treating as 0 for mirroring calculation.")
        ext_cx = 0

    # Calculate current center X of the shape relative to the parent's left edge
    shape_center_x = off_x + (ext_cx / 2.0)
    # Calculate center X of the parent (slide)
    parent_center_x = parent_width / 2.0

    # Calculate the displacement of the shape's center from the parent's center
    displacement = shape_center_x - parent_center_x

    # The mirrored center is the same displacement but on the opposite side of the parent center
    mirrored_center_x = parent_center_x - displacement

    # Calculate the new X-offset (top-left corner) based on the mirrored center and shape width
    new_off_x = mirrored_center_x - (ext_cx / 2.0)

    logging.debug(f"Mirror calc: off_x={off_x}, ext_cx={ext_cx}, parent_w={parent_width} -> new_off_x={int(new_off_x)}")
    # Return the integer part of the new offset
    return int(new_off_x)

def _get_transform_properties(xfrm_element: etree._Element) -> Optional[Dict[str, Any]]:
    """
    Helper function to extract offset (x, y), extent (cx, cy), and rotation (rot)
    properties from various transform elements (<a:xfrm>, <p:xfrm>).
    Returns a dictionary with parsed integer values and references to the original elements,
    or None if properties cannot be reliably extracted.
    """
    if xfrm_element is None:
        logging.debug("Transform element provided to _get_transform_properties was None.")
        return None

    off_elem = xfrm_element.find("a:off", NSMAP) # Offset element (<a:off x="..." y="...">)
    ext_elem = xfrm_element.find("a:ext", NSMAP) # Extent element (<a:ext cx="..." cy="...">)
    x_str, y_str, cx_str, cy_str = None, None, None, None
    rot_str = xfrm_element.get("rot") # Rotation attribute directly on xfrm
    flipH_str = xfrm_element.get("flipH") # Horizontal flip attribute
    flipV_str = xfrm_element.get("flipV") # Vertical flip attribute
    source_info = "unknown" # For logging where the properties were found

    # Case 1: Standard <a:xfrm> with <a:off> and <a:ext> children
    if off_elem is not None and ext_elem is not None:
        x_str, y_str = off_elem.get("x"), off_elem.get("y")
        cx_str, cy_str = ext_elem.get("cx"), ext_elem.get("cy")
        source_info = "<a:off>, <a:ext>"
    # Case 2: Properties directly on the transform element (e.g., <p:xfrm x="..." y="..." cx="..." cy="...">)
    # Check for x attribute specifically, as rot/flipH/flipV can exist without position/extent
    elif xfrm_element.get("x") is not None:
        x_str, y_str = xfrm_element.get("x"), xfrm_element.get("y")
        cx_str, cy_str = xfrm_element.get("cx"), xfrm_element.get("cy")
        source_info = f"{etree.QName(xfrm_element.tag).localname} attributes"
        # In this case, off_elem and ext_elem are None, properties are on xfrm_element itself
        off_elem, ext_elem = None, None
    # Case 3: Fallback for less common structures (e.g., nested transforms, though unlikely for top-level)
    else:
        child_off = xfrm_element.find(".//a:off", NSMAP) # Search descendants
        child_ext = xfrm_element.find(".//a:ext", NSMAP) # Search descendants
        if child_off is not None and child_ext is not None:
            off_elem, ext_elem = child_off, child_ext
            x_str, y_str = off_elem.get("x"), off_elem.get("y")
            cx_str, cy_str = ext_elem.get("cx"), ext_elem.get("cy")
            source_info = "grandchild <a:off>, <a:ext>"
        else:
            # If no common structure is found, log and return None
            logging.debug(f"Could not find offset/extent for transform element {etree.QName(xfrm_element.tag).localname} using common methods.")
            return None

    logging.debug(f"Read transform props (from {source_info}): x='{x_str}', y='{y_str}', cx='{cx_str}', cy='{cy_str}', rot='{rot_str}', flipH='{flipH_str}', flipV='{flipV_str}'")

    # Try parsing the extracted string values into integers
    try:
        x = int(x_str) if x_str is not None else 0
        y = int(y_str) if y_str is not None else 0
        cx = int(cx_str) if cx_str is not None else 0
        cy = int(cy_str) if cy_str is not None else 0
        rot = int(rot_str) if rot_str is not None else 0 # Rotation in 60,000ths of a degree
        flipH = flipH_str == "1" # Convert "1" to True, otherwise False
        flipV = flipV_str == "1" # Convert "1" to True, otherwise False

        # Return dictionary including parsed values and references to the XML elements
        props = {
            "x": x, "y": y, "cx": cx, "cy": cy, "rot": rot, "flipH": flipH, "flipV": flipV,
            "_off_elem": off_elem,       # Reference to <a:off> or None
            "_ext_elem": ext_elem,       # Reference to <a:ext> or None
            "_xfrm_elem": xfrm_element   # Reference to the main transform element
        }
        logging.debug(f"Parsed transform props: x={x}, y={y}, cx={cx}, cy={cy}, rot={rot}, flipH={flipH}, flipV={flipV}")
        return props
    except (ValueError, TypeError) as e:
        # Handle errors during integer conversion
        logging.warning(f"Could not parse transform properties (from {source_info}): x='{x_str}', y='{y_str}', cx='{cx_str}', cy='{cy_str}', rot='{rot_str}'. Error: {e}")
        return None

def _set_transform_properties(props: Dict[str, Any], set_flip_h: bool = False) -> bool:
    """
    Helper function to set the 'x' attribute (position) and optionally the 'flipH'
    attribute (horizontal flip) and adjust rotation ('rot') on the appropriate transform element.
    Uses the references stored in the 'props' dictionary by _get_transform_properties.
    For rotated elements, temporarily removes rotation, applies flipH, then reapplies negated rotation.

    Args:
        props (Dict[str, Any]): Dictionary containing transform properties and element references.
                                Must include 'x', 'rot', '_xfrm_elem', and optionally '_off_elem'.
                                Must include 'x_should_change' boolean flag.
        set_flip_h (bool, optional): Whether to attempt to apply the standard horizontal flip (flipH="1")
                                     and adjust rotation. Defaults to False.

    Returns:
        bool: True if any changes were made, False otherwise.
    """
    off_elem = props.get("_off_elem")      # Reference to <a:off> or None
    xfrm_elem = props.get("_xfrm_elem")    # Reference to the main transform element (<a:xfrm> or <p:xfrm>)
    new_x = props['x']                     # The calculated new X position
    original_rotation = props.get('rot', 0) # Original rotation angle (default 0)
    new_x_str = str(new_x)                 # Convert to string for setting attribute
    changed = False                        # Flag to track if any modification occurs
    x_set_success = False                  # Flag to track if 'x' was successfully set
    original_x_str = "N/A"                 # For logging purposes

    # Ensure the main transform element reference exists
    if xfrm_elem is None:
        logging.error("Cannot set properties: _xfrm_elem reference is missing in props.")
        return False

    # --- Handle Position Mirroring (Attribute 'x') ---
    target_x_elem = None  # The element where 'x' attribute will be set
    target_tag_name = "N/A" # For logging purposes

    # Determine where to set the 'x' attribute based on how it was read
    if off_elem is not None:
        target_x_elem = off_elem
        original_x_str = target_x_elem.get("x", "Not Found")
        target_tag_name = "a:off"
    elif xfrm_elem.get("x") is not None:
        target_x_elem = xfrm_elem
        original_x_str = xfrm_elem.get("x")
        target_tag_name = f"{etree.QName(xfrm_elem.tag).localname} attributes"

    # Set the 'x' attribute if a target element was found and the value needs changing
    if target_x_elem is not None:
        if props.get('x_should_change', False) and original_x_str != new_x_str: # Check x_should_change flag
            logging.debug(f"Attempting to set x='{new_x_str}' on {target_tag_name} (original: '{original_x_str}', original_rot: {original_rotation})")
            try:
                target_x_elem.set("x", new_x_str)
                changed = True
                x_set_success = True
                logging.debug(f" -> x set successfully on {target_tag_name}")
            except Exception as e:
                logging.error(f" -> Failed to set x on {target_tag_name}: {e}")
        elif not props.get('x_should_change', False):
            logging.debug(f"Skipping x attribute setting on {target_tag_name} as 'x_should_change' is False.")
        elif original_x_str == new_x_str:
            logging.debug(f"Skipping x attribute setting on {target_tag_name} as value '{new_x_str}' is unchanged.")
    else:
        logging.warning(f"Could not determine where to set 'x' property for element referenced by {etree.QName(xfrm_elem.tag).localname}.")

    # --- Handle Standard Visual Flip (Attribute 'flipH') and Rotation Adjustment ---
    if set_flip_h:
        original_flipH_val = xfrm_elem.get("flipH") # Get current flipH value ("1" or "0" or None)
        # Determine the new flip state: if it's currently "1", make it "0", otherwise make it "1"
        # This toggles the flip state, which is generally what's needed for mirroring.
        new_flipH_val = "0" if original_flipH_val == "1" else "1"

        logging.debug(f"Attempting to set flipH='{new_flipH_val}' on {etree.QName(xfrm_elem.tag).localname} (original flipH: '{original_flipH_val}', original_rot: {original_rotation})")
        try:
            if original_rotation != 0:
                # *** Rotated Element Flip Logic ***
                logging.debug(f" -> Element is rotated ({original_rotation}). Removing rot, setting flipH={new_flipH_val}, re-applying negated rot.")
                original_rot_str = xfrm_elem.get("rot") # Keep original string just in case

                # 1. Remove rotation attribute
                if "rot" in xfrm_elem.attrib:
                    del xfrm_elem.attrib["rot"]
                    logging.debug(" -> Temporarily removed 'rot' attribute.")
                else:
                    logging.debug(" -> 'rot' attribute was not present to remove.")

                # 2. Set flipH attribute to the new value
                xfrm_elem.set("flipH", new_flipH_val)
                logging.debug(f" -> Set flipH='{new_flipH_val}'.")

                # 3. Re-apply negated rotation
                new_rotation = -original_rotation
                new_rot_str = str(new_rotation)
                xfrm_elem.set("rot", new_rot_str)
                logging.debug(f" -> Re-applied rotation as rot='{new_rot_str}'.")

                # Update the rotation in the props dict for consistency if needed later
                props['rot'] = new_rotation
                changed = True

            else:
                # *** Non-Rotated Element Flip Logic ***
                xfrm_elem.set("flipH", new_flipH_val)
                logging.debug(f" -> Set flipH='{new_flipH_val}' (element was not rotated).")
                changed = True

        except Exception as e_flip_rot:
            logging.error(f" -> Failed during flipH/rotation adjustment on {etree.QName(xfrm_elem.tag).localname}: {e_flip_rot}")

    # Log a warning if the position was calculated to change but setting 'x' failed
    if props.get('x_should_change', False) and not x_set_success:
        logging.warning(f"Calculated new X={new_x} for {etree.QName(xfrm_elem.tag).localname} but failed to set it (original was '{original_x_str}', rotation: {original_rotation}).")

    return changed


# --- Table Column Flipping ---

def _mirror_table_columns(table_element: etree._Element, filename: str, options: Dict[str, bool]) -> bool:
    """
    Reverses the order of table columns (<a:gridCol>) and attempts to swap the *content*
    of cells (<a:tc>) within each row (<a:tr>). Also attempts to set cell anchor.
    WARNING: This is experimental. It currently SKIPS rows containing any merged cells
             (hMerge, vMerge, gridSpan > 1) to avoid table corruption.

    Args:
        table_element: The <a:tbl> element.
        filename: The name of the XML file being processed (for logging).
        options: Dictionary of processing options (used for 'right_align_text').

    Returns:
        True if any changes were made, False otherwise.
    """
    if table_element is None or table_element.tag != etree.QName(NSMAP['a'], 'tbl'):
        logging.error(f"[{filename}] Invalid element passed to _mirror_table_columns (expected <a:tbl>).")
        return False

    changed = False
    logging.warning(f"[{filename}] Attempting EXPERIMENTAL table column flip (Content Swap & Anchor).")

    # 1. Reverse the column definitions in <a:tblGrid>
    grid = table_element.find("a:tblGrid", NSMAP)
    if grid is not None:
        grid_cols = grid.xpath("./a:gridCol", namespaces=NSMAP)
        if len(grid_cols) > 1:
            logging.debug(f"[{filename}] Reversing {len(grid_cols)} grid columns definitions (<a:gridCol>).")
            reversed_grid_cols = list(reversed(grid_cols))
            # Remove existing cols
            for col in grid_cols: grid.remove(col)
            # Add reversed cols back
            for col in reversed_grid_cols: grid.append(col)
            changed = True
            logging.info(f"[{filename}] Reversed order of <a:gridCol> elements in table grid.")
        else:
            logging.debug(f"[{filename}] Table has 0 or 1 grid columns, no grid definition reversal needed.")
    else:
        logging.warning(f"[{filename}] Table found but required <a:tblGrid> element is missing. Cannot safely reverse columns.")
        return False # Cannot proceed without grid definition

    # 2. Reverse the *content* of cells within each row (<a:tr>)
    rows = table_element.xpath("./a:tr", namespaces=NSMAP)
    logging.debug(f"[{filename}] Found {len(rows)} rows (<a:tr>) in table for cell content swap.")
    rows_swapped = 0
    rows_skipped_merge = 0

    for r_idx, row in enumerate(rows):
        cells = row.xpath("./a:tc", namespaces=NSMAP)
        num_cells = len(cells)
        if num_cells <= 1:
            logging.debug(f"[{filename}] Row {r_idx}: Skipping content swap (0 or 1 cell).")
            continue

        # --- Check for Merged Cells (Horizontal or Vertical) ---
        # If any cell in the row has vMerge or hMerge or gridSpan > 1, skip the row.
        has_merge = any(
            c.get('gridSpan', '1') != '1' or c.get('hMerge') == '1' or c.get('vMerge') == '1'
            for c in cells
        )
        if has_merge:
            logging.error(f"[{filename}] Row {r_idx}: SKIPPING content swap because row contains merged/spanned cells (gridSpan, hMerge, or vMerge). This feature is not supported.")
            rows_skipped_merge += 1
            continue

        # --- Proceed with Content Swap if No Merges Found ---
        logging.debug(f"[{filename}] Row {r_idx}: Attempting to swap content of {num_cells} cells.")
        row_content_swapped = False
        try:
            # Extract content (all children) from each cell
            cell_contents = []
            for cell in cells:
                # Make a deep copy of children to avoid issues when clearing/appending
                content = [copy.deepcopy(child) for child in cell]
                cell_contents.append(content)

            # Reverse the list of contents
            reversed_contents = list(reversed(cell_contents))

            # Clear original cells and append reversed content
            for i, cell in enumerate(cells):
                # Remove all existing children from the cell
                for child in cell:
                    cell.remove(child)
                # Append the content from the corresponding reversed list
                for content_element in reversed_contents[i]:
                    cell.append(content_element)

            changed = True
            rows_swapped += 1
            row_content_swapped = True # Mark that this row's content was swapped
            logging.debug(f"[{filename}] Row {r_idx}: Successfully swapped content of {num_cells} cells.")

        except Exception as e:
            logging.error(f"[{filename}] Row {r_idx}: Error during cell content swap: {e}")
            # Attempting to swap content failed, potentially leaving the row in a bad state.

        # --- Set Cell Properties (e.g., anchor) if content was swapped ---
        # Attempt to set anchor='t' in tcPr to potentially help with text alignment rendering
        if row_content_swapped and options.get('right_align_text', False):
            logging.debug(f"[{filename}] Row {r_idx}: Attempting to set cell properties (anchor='t') for swapped cells.")
            cell_props_set_count = 0
            for cell in cells:
                try:
                    tcPr = cell.find("a:tcPr", NSMAP)
                    if tcPr is None:
                        # Create tcPr if missing, insert it before any content (like txBody)
                        tcPr = etree.Element(etree.QName(NSMAP['a'], 'tcPr'))
                        first_content = cell.find("*") # Find first child element
                        if first_content is not None:
                            first_content.addprevious(tcPr)
                        else:
                            cell.append(tcPr)
                        logging.debug(f"[{filename}] Row {r_idx}, Cell: Created missing <a:tcPr>.")

                    # Set anchor attribute to 't' (top)
                    if tcPr.get("anchor") != "t":
                        tcPr.set("anchor", "t")
                        logging.debug(f"[{filename}] Row {r_idx}, Cell: Set anchor='t' in <a:tcPr>.")
                        changed = True # Setting property is a change
                        cell_props_set_count += 1
                    # Optionally remove anchorCtr if needed, but let's try just setting anchor first
                    # if tcPr.get("anchorCtr") is not None:
                    #     del tcPr.attrib["anchorCtr"]
                    #     logging.debug(f"[{filename}] Row {r_idx}, Cell: Removed anchorCtr attribute.")
                    #     changed = True

                except Exception as e_tcpr:
                    logging.error(f"[{filename}] Row {r_idx}, Cell: Error setting cell properties: {e_tcpr}")
            if cell_props_set_count > 0:
                logging.info(f"[{filename}] Row {r_idx}: Set anchor='t' for {cell_props_set_count} cells.")


    if rows_swapped > 0:
        logging.info(f"[{filename}] Successfully swapped content in {rows_swapped} table rows.")
    if rows_skipped_merge > 0:
        logging.warning(f"[{filename}] Skipped content swap for {rows_skipped_merge} table rows due to merged/spanned cells.")
    if not changed and rows_skipped_merge == 0: # Only log 'no changes' if nothing was skipped either
        logging.info(f"[{filename}] No changes applied during table content swap (no reversible rows found or needed).")

    return changed


# --- Chart Flipping (NEW - Experimental) ---

def _flip_chart_elements(chart_space: etree._Element, filename: str, elem_idx: Optional[int] = None) -> bool:
    """
    EXPERIMENTAL: Attempts to flip internal elements of a chart.
    Focuses on:
    - Reversing category/date axis order using c:scaling/c:orientation.
    - Changing value axis position (c:valAx/c:axPos).
    - Mirroring pie/doughnut chart first slice angle (c:firstSliceAng).
    Further revised to correctly use c:orientation for category/date axes.

    Args:
        chart_space: The <c:chartSpace> element (root of chart XML).
        filename: The name of the XML file being processed (for logging).
        elem_idx: The index of the parent graphicFrame element (for logging, if applicable).

    Returns:
        True if any changes were made, False otherwise.
    """
    if chart_space is None or chart_space.tag != etree.QName(NSMAP['c'], 'chartSpace'):
        logging.error(f"[{filename}] Invalid element passed to _flip_chart_elements (expected <c:chartSpace>).")
        return False

    changed = False
    log_context = f"[{filename}] Chart" + (f" (from Frame #{elem_idx})" if elem_idx is not None else "")
    logging.info(f"{log_context}: Attempting chart internal flip (Further Revised Logic for Axis Order).")

    try:
        plot_area = chart_space.find(".//c:plotArea", namespaces=NSMAP)
        if plot_area is None:
            logging.warning(f"{log_context}: Chart has no <c:plotArea>, cannot perform internal flip.")
            return False

        # 1. Reverse Category and Date Axis (c:catAx, c:dateAx) Order
        # This uses c:scaling/c:orientation@val which should be "minMax" (normal) or "maxMin" (reversed).
        category_axes_tags = ["c:catAx", "c:dateAx"]
        for ax_tag_name in category_axes_tags:
            axes = plot_area.xpath(f"./{ax_tag_name}", namespaces=NSMAP)
            for ax_idx, axis_elem in enumerate(axes):
                ax_local_name = etree.QName(axis_elem.tag).localname
                log_ax_context = f"{log_context} {ax_local_name} #{ax_idx}"

                scaling_elem = axis_elem.find("c:scaling", namespaces=NSMAP)
                if scaling_elem is None:
                    # c:scaling is mandatory for c:catAx and c:dateAx. If missing, the chart XML is likely malformed.
                    # However, we'll try to create it to be robust, though this is a recovery attempt.
                    ax_id_elem = axis_elem.find("c:axId", namespaces=NSMAP)
                    scaling_elem = etree.Element(etree.QName(NSMAP['c'], 'scaling'))
                    if ax_id_elem is not None: # axId is mandatory, should exist
                        ax_id_elem.addnext(scaling_elem)
                    else: # Fallback if axId is unexpectedly missing
                        axis_elem.insert(0, scaling_elem)
                    logging.warning(f"{log_ax_context}: Created missing mandatory <c:scaling> element.")
                    changed = True

                orientation_elem = scaling_elem.find("c:orientation", namespaces=NSMAP)
                if orientation_elem is None:
                    # c:orientation is optional within c:scaling. Default is "minMax".
                    # Insert c:orientation as the first child of c:scaling if creating.
                    orientation_elem = etree.Element(etree.QName(NSMAP['c'], 'orientation'))
                    scaling_elem.insert(0, orientation_elem) # Insert at the beginning of c:scaling
                    orientation_elem.set("val", "minMax") # Set initial default
                    logging.debug(f"{log_ax_context}: Created missing <c:orientation> in <c:scaling> with val='minMax'.")
                    changed = True

                current_orientation_val = orientation_elem.get("val", "minMax") # Default to "minMax" if val attribute is missing
                new_orientation_val = "maxMin" if current_orientation_val == "minMax" else "minMax"

                if current_orientation_val != new_orientation_val:
                    orientation_elem.set("val", new_orientation_val)
                    logging.info(f"{log_ax_context}: Toggled axis orientation in <c:scaling> from '{current_orientation_val}' to '{new_orientation_val}'.")
                    changed = True
                else:
                    # If it's already maxMin, toggling would make it minMax.
                    # For a single LTR -> RTL pass, this is the desired state if it was already maxMin.
                    logging.debug(f"{log_ax_context}: Axis orientation in <c:scaling> is already '{new_orientation_val}'.")

        # 2. Change Value Axis (c:valAx) Position
        val_axes = plot_area.xpath("./c:valAx", namespaces=NSMAP)
        for ax_idx, val_ax_elem in enumerate(val_axes):
            log_ax_context = f"{log_context} ValAx #{ax_idx}"
            ax_pos_elem = val_ax_elem.find("c:axPos", namespaces=NSMAP)
            if ax_pos_elem is not None:
                current_pos = ax_pos_elem.get("val", "l")
                new_pos = "r" if current_pos == "l" else "l"
                if current_pos != new_pos:
                    ax_pos_elem.set("val", new_pos)
                    logging.info(f"{log_ax_context}: Changed value axis position from '{current_pos}' to '{new_pos}'.")
                    changed = True

            tick_lbl_pos_elem = val_ax_elem.find("c:tickLblPos", namespaces=NSMAP)
            if tick_lbl_pos_elem is not None:
                 current_lbl_pos = tick_lbl_pos_elem.get("val", "nextTo")
                 new_lbl_pos = "high" if current_lbl_pos == "nextTo" and ax_pos_elem is not None and ax_pos_elem.get("val") == "r" else current_lbl_pos
                 if tick_lbl_pos_elem.get("val") != new_lbl_pos:
                     tick_lbl_pos_elem.set("val", new_lbl_pos)
                     logging.info(f"{log_ax_context}: Changed tick label position from '{current_lbl_pos}' to '{new_lbl_pos}'.")
                     changed = True

        # 3. Adjust Pie/Doughnut Chart First Slice Angle (c:firstSliceAng)
        pie_chart_tags = ["c:pieChart", "c:doughnutChart", "c:ofPieChart"] # .//c:ofPieChart was too broad
        for pie_tag_name in pie_chart_tags:
            # Search for these chart types directly under plotArea or plotArea/layout
            # Using .// might find nested charts if they exist, which could be complex.
            # Sticking to direct children of plotArea for now.
            pie_chart_elems = plot_area.xpath(f"./{pie_tag_name} | ./c:layout/c:plotArea/{pie_tag_name}", namespaces=NSMAP)

            for pie_idx, pie_chart_elem in enumerate(pie_chart_elems):
                pie_local_name = etree.QName(pie_chart_elem.tag).localname
                log_pie_context = f"{log_context} {pie_local_name} #{pie_idx}"
                first_slice_ang_elem = pie_chart_elem.find("c:firstSliceAng", namespaces=NSMAP)

                if first_slice_ang_elem is None:
                    first_slice_ang_elem = etree.Element(etree.QName(NSMAP['c'], 'firstSliceAng'))
                    first_slice_ang_elem.set("val", "0")

                    # Try to insert before c:extLst if it exists, else append.
                    # Schema: (<c:varyColors/>)? (<c:ser/>)+ (<c:dLbls/>)? (<c:firstSliceAng/>)? (<c:holeSize/>)? (<c:extLst/>)?
                    # So, it can come after dLbls, or after ser if dLbls is not present.
                    # A safe bet is to append if extLst is not there, or insert before extLst.
                    ext_lst_elem = pie_chart_elem.find("c:extLst", namespaces=NSMAP)
                    if ext_lst_elem is not None:
                        ext_lst_elem.addprevious(first_slice_ang_elem)
                        logging.debug(f"{log_pie_context}: Created missing <c:firstSliceAng> and inserted before <c:extLst>.")
                    else:
                        # If no extLst, try to find last c:ser or c:dLbls to append after, or just append.
                        # For simplicity and robustness against various pie chart structures, appending is often safest if specific prior siblings aren't guaranteed.
                        pie_chart_elem.append(first_slice_ang_elem)
                        logging.debug(f"{log_pie_context}: Created missing <c:firstSliceAng> and appended to chart element.")
                    changed = True

                current_angle_str = first_slice_ang_elem.get("val", "0")
                try:
                    current_angle = int(current_angle_str)
                    # Mirror the angle around the vertical axis (12 o'clock).
                    new_angle = (360 - current_angle) % 360

                    if current_angle != new_angle: # Only set if the value actually changes
                        first_slice_ang_elem.set("val", str(new_angle))
                        logging.info(f"{log_pie_context}: Mirrored first slice angle from {current_angle}° to {new_angle}°.")
                        changed = True
                    else:
                        logging.debug(f"{log_pie_context}: First slice angle {current_angle}° is symmetric or already 0; no change needed.")
                except ValueError:
                    logging.warning(f"{log_pie_context}: Could not parse firstSliceAng value '{current_angle_str}' as integer.")

    except Exception as e:
        logging.error(f"{log_context}: CRITICAL Error during chart internal flip: {e}")
        # Potentially re-raise or return a specific error indicator if this function's failure should halt further processing of this file.
        # For now, it will just log and return 'changed' which might be True even if an error occurred after some changes.

    if changed:
        logging.info(f"{log_context}: Applied one or more internal chart flips/structural changes.")
    else:
        logging.info(f"{log_context}: No internal chart flips applied or needed based on current values for this chart type/state.")

    return changed


# --- Arabic Proofing Language ---
def _set_arabic_proofing_language(text_container_elem: etree._Element, options: Dict[str, bool], filename: str, context: str) -> bool:
    """
    Helper function to set Arabic proofing language (lang="ar-SA") on all text runs within a text container.
    This affects spell checking, grammar checking, and other proofing tools in PowerPoint.
    
    Args:
        text_container_elem: The element containing text (p:sp, a:tc, etc.)
        options: Dictionary of processing options
        filename: The name of the XML file being processed (for logging)
        context: Context string for logging
        
    Returns:
        True if any changes were made, False otherwise.
    """
    if not options.get('set_arabic_proofing_lang', False):
        return False
        
    if text_container_elem is None:
        logging.warning(f"[{filename}] {context}: Attempted to set Arabic proofing language on None element.")
        return False
    
    changed = False
    arabic_lang_code = "ar-SA"  # Arabic (Saudi Arabia) - most common Arabic locale
    
    # Find the text body element
    txBody = None
    if text_container_elem.tag == etree.QName(NSMAP['p'], 'sp'):
        txBody = text_container_elem.find("p:txBody", NSMAP)
    elif text_container_elem.tag == etree.QName(NSMAP['a'], 'tc'):
        txBody = text_container_elem.find("a:txBody", NSMAP)
    
    if txBody is None:
        logging.debug(f"[{filename}] {context}: No txBody found, skipping Arabic proofing language setting.")
        return False
    
    # Find all text runs (<a:r>) within the text body
    text_runs = txBody.xpath(".//a:r", namespaces=NSMAP)
    runs_modified = 0
    
    for run_idx, run_elem in enumerate(text_runs):
        # Find or create run properties (<a:rPr>)
        rPr = run_elem.find("a:rPr", NSMAP)
        if rPr is None:
            # Create rPr if missing - insert before the text element
            rPr = etree.Element(etree.QName(NSMAP['a'], 'rPr'))
            text_elem = run_elem.find("a:t", NSMAP)
            if text_elem is not None:
                text_elem.addprevious(rPr)
            else:
                run_elem.insert(0, rPr)  # Insert at beginning if no text element found
            logging.debug(f"[{filename}] {context} Run #{run_idx}: Created missing <a:rPr>.")
        
        # Set the language attribute
        current_lang = rPr.get("lang")
        if current_lang != arabic_lang_code:
            rPr.set("lang", arabic_lang_code)
            logging.debug(f"[{filename}] {context} Run #{run_idx}: Set lang='{arabic_lang_code}' (was '{current_lang}')")
            runs_modified += 1
            changed = True
    
    if runs_modified > 0:
        logging.info(f"[{filename}] {context}: Set Arabic proofing language on {runs_modified} text runs.")
    
    return changed


def _set_arabic_proofing_language_on_master_styles(txStyles: etree._Element, options: Dict[str, bool], filename: str) -> bool:
    """
    Helper function to set Arabic proofing language on master text styles.
    Master styles have a different structure with lvlXpPr elements containing defRPr.
    
    Args:
        txStyles: The <p:txStyles> element from a slide master
        options: Dictionary of processing options
        filename: The name of the XML file being processed (for logging)
        
    Returns:
        True if any changes were made, False otherwise.
    """
    if not options.get('set_arabic_proofing_lang', False):
        return False
        
    if txStyles is None:
        return False
    
    changed = False
    arabic_lang_code = "ar-SA"
    styles_modified = 0
    
    # Process each style type (title, body, other)
    for style_tag in ["p:titleStyle", "p:bodyStyle", "p:otherStyle"]:
        style_elem = txStyles.find(style_tag, namespaces=NSMAP)
        if style_elem is not None:
            style_name = etree.QName(style_elem.tag).localname
            
            # Find all level properties within the style element
            lvl_props_list = []
            a_ns = NSMAP['a']
            for child in style_elem.iterdescendants():
                if child.tag.startswith(f"{{{a_ns}}}lvl") and child.tag.endswith("pPr"):
                    lvl_props_list.append(child)
            
            for lvl_idx, lvl_pPr in enumerate(lvl_props_list):
                lvl_name = etree.QName(lvl_pPr.tag).localname
                
                # Find or create defRPr (default run properties) within the level
                defRPr = lvl_pPr.find("a:defRPr", NSMAP)
                if defRPr is None:
                    # Create defRPr if missing
                    defRPr = etree.Element(etree.QName(NSMAP['a'], 'defRPr'))
                    lvl_pPr.append(defRPr)
                    logging.debug(f"[{filename}] Master {style_name}/{lvl_name}: Created missing <a:defRPr>.")
                    changed = True
                
                # Set the language attribute
                current_lang = defRPr.get("lang")
                if current_lang != arabic_lang_code:
                    defRPr.set("lang", arabic_lang_code)
                    logging.debug(f"[{filename}] Master {style_name}/{lvl_name}: Set lang='{arabic_lang_code}' on defRPr (was '{current_lang}')")
                    styles_modified += 1
                    changed = True
    
    if styles_modified > 0:
        logging.info(f"[{filename}] Set Arabic proofing language on {styles_modified} master style levels.")
    
    return changed


# --- Paragraph Indentation Flipping Helper ---
def _flip_paragraph_indentation(pPr_element: etree._Element, filename: str, context: str) -> bool:
    """
    Adjusts marL, marR, and indent attributes for RTL within a pPr-like element
    (<a:pPr>, <a:defPPr>, <a:lvlXpPr>).
    Returns True if changes were made.
    """
    if pPr_element is None:
        return False
    
    changed = False
    
    original_marL_str = pPr_element.get("marL")
    original_marR_str = pPr_element.get("marR")
    original_indent_str = pPr_element.get("indent")

    # Convert to int, defaulting to 0 if attribute is missing or invalid
    marL_val = 0
    marR_val = 0
    
    try:
        if original_marL_str is not None: marL_val = int(original_marL_str)
    except ValueError: logging.warning(f"[{filename}] {context}: Invalid marL value '{original_marL_str}', treating as 0.")
    try:
        if original_marR_str is not None: marR_val = int(original_marR_str)
    except ValueError: logging.warning(f"[{filename}] {context}: Invalid marR value '{original_marR_str}', treating as 0.")
    
    # --- Swap marL and marR ---
    # New marL becomes old marR
    if str(marR_val) != original_marL_str: # Check if change is needed
        pPr_element.set("marL", str(marR_val))
        changed = True
        logging.debug(f"[{filename}] {context}: Set marL='{marR_val}' (was '{original_marL_str}')")
    elif marR_val == 0 and original_marL_str is not None and original_marL_str != "0": # Explicitly set to 0 if old marL was non-zero
        pPr_element.set("marL", "0")
        changed = True
        logging.debug(f"[{filename}] {context}: Set marL='0' (was '{original_marL_str}')")
    elif marR_val != 0 and original_marL_str is None: # Set if new value is non-zero and old was absent
        pPr_element.set("marL", str(marR_val))
        changed = True
        logging.debug(f"[{filename}] {context}: Set marL='{marR_val}' (was absent)")


    # New marR becomes old marL
    if str(marL_val) != original_marR_str: # Check if change is needed
        pPr_element.set("marR", str(marL_val))
        changed = True
        logging.debug(f"[{filename}] {context}: Set marR='{marL_val}' (was '{original_marR_str}')")
    elif marL_val == 0 and original_marR_str is not None and original_marR_str != "0": # Explicitly set to 0 if old marR was non-zero
        pPr_element.set("marR", "0")
        changed = True
        logging.debug(f"[{filename}] {context}: Set marR='0' (was '{original_marR_str}')")
    elif marL_val != 0 and original_marR_str is None:  # Set if new value is non-zero and old was absent
        pPr_element.set("marR", str(marL_val))
        changed = True
        logging.debug(f"[{filename}] {context}: Set marR='{marL_val}' (was absent)")
        
    # --- Handle indent attribute ---
    # The 'indent' attribute specifies the indentation of the first line of text in a paragraph.
    # For RTL, a positive LTR indent (text pushed right) should become a negative indent (text pushed left from the right margin).
    # A negative LTR indent (hanging indent, text pulled left) should become a positive indent for RTL (hanging from right).
    if original_indent_str is not None:
        try:
            indent_val = int(original_indent_str)
            if indent_val != 0: # Only process if indent is not zero
                new_indent_val = -indent_val # Simple negation
                new_indent_str = str(new_indent_val)
                pPr_element.set("indent", new_indent_str)
                changed = True
                logging.debug(f"[{filename}] {context}: Flipped indent from '{original_indent_str}' to '{new_indent_str}'.")
            else:
                logging.debug(f"[{filename}] {context}: Indent is '0', no change needed.")
        except ValueError:
            logging.warning(f"[{filename}] {context}: Invalid indent value '{original_indent_str}', not flipping.")
            
    return changed


 #--- Text Alignment Helpers ---

def _apply_alignment_to_defPPr(defPPr: etree._Element, options: Dict[str, bool], filename: str, context: str) -> bool:
    """
    Helper function to apply right alignment ('algn="r"') and RTL direction ('rtl="1"')
    to a given default paragraph properties element (<a:defPPr>). Also flips indentation.
    Modifications are based on the 'right_align_text' and 'set_rtl_paragraphs' options.
    Logs changes made with context information.
    Returns True if any changes were made, False otherwise.
    """
    if defPPr is None:
        logging.warning(f"[{filename}] {context}: Attempted to apply alignment to a None defPPr element.")
        return False

    changed_by_this_func = False 
    pPr_was_modified_for_rtl = False # Track if RTL was set here

    current_def_algn = defPPr.get("algn") 
    logging.debug(f"[{filename}] {context}: Checking defPPr alignment. Current algn='{current_def_algn}'")

    if options.get('right_align_text', False):
        if current_def_algn in [None, "l", "just", "dist"]: 
            defPPr.set("algn", "r") 
            logging.debug(f"[{filename}] {context}: Set default algn='r' (was '{current_def_algn}')")
            changed_by_this_func = True

    final_algn_for_rtl_check = defPPr.get("algn") # Re-check alignment after potential change
    apply_rtl_on_def = options.get('set_rtl_paragraphs', False) and final_algn_for_rtl_check != "ctr"

    if apply_rtl_on_def:
        if defPPr.get("rtl") != "1": 
            defPPr.set("rtl", "1") 
            logging.debug(f"[{filename}] {context}: Set default rtl='1' (defPPr algn is now '{final_algn_for_rtl_check}')")
            changed_by_this_func = True 
            pPr_was_modified_for_rtl = True
    elif options.get('set_rtl_paragraphs', False) and final_algn_for_rtl_check == "ctr":
        logging.debug(f"[{filename}] {context}: Skipping default rtl='1' setting because defPPr is centered.")

    # If RTL was set (or was already set and option is true), flip indentation
    if (pPr_was_modified_for_rtl or (defPPr.get("rtl") == "1" and options.get('set_rtl_paragraphs', False))):
        if _flip_paragraph_indentation(defPPr, filename, f"{context} indentation"):
            changed_by_this_func = True
            
    return changed_by_this_func

# --- Experimental Geometry Flip ---
def _try_flip_preset_geometry(spPr: etree._Element, xfrm_element: etree._Element, filename: str, elem_idx: int) -> bool:
    """
    EXPERIMENTAL: Attempts to visually flip specific preset geometries.
    - For 'triangle': Modifies adjustment values (<a:avLst>).
    - For 'rtTriangle': Sets flipV="1" and flipH="0" on the transform.
    This is intended as a fallback when standard flipH on the transform doesn't work visually.

    Args:
        spPr: The shape properties element (<p:spPr>) containing the geometry.
        xfrm_element: The transform element (<a:xfrm>) associated with the shape.
        filename: The name of the XML file being processed (for logging).
        elem_idx: The index of the element being processed (for logging).

    Returns:
        True if an experimental flip was successfully applied, False otherwise.
    """
    if spPr is None or xfrm_element is None: return False

    prstGeom = spPr.find("a:prstGeom", NSMAP)
    if prstGeom is None: return False # Not a preset geometry

    preset_type = prstGeom.get("prst")
    logging.debug(f"[{filename}] Element #{elem_idx}: Found preset geometry '{preset_type}'. Checking for experimental flip.")

    # --- Target 'triangle' (Isosceles) ---
    if preset_type == "triangle":
        avLst = prstGeom.find("a:avLst", NSMAP)
        if avLst is None:
            logging.debug(f"[{filename}] Element #{elem_idx} (triangle): No <a:avLst> found, cannot apply experimental adjustment flip.")
            return False

        # Find the first adjustment guide named "adj" (heuristic)
        adj_guide = avLst.find("a:gd[@name='adj']", NSMAP)
        if adj_guide is None:
            logging.warning(f"[{filename}] Element #{elem_idx} (triangle): Could not find adjustment guide <a:gd name='adj'>.")
            return False

        fmla = adj_guide.get("fmla")
        if not fmla or not fmla.startswith("val "):
            logging.warning(f"[{filename}] Element #{elem_idx} (triangle): 'adj' guide formula '{fmla}' is not in expected 'val <number>' format.")
            return False

        try:
            # Extract the numeric value (assuming it's an integer, often 0-100000)
            match = re.search(r'\bval\s+(\d+)\b', fmla)
            if not match:
                logging.warning(f"[{filename}] Element #{elem_idx} (triangle): Could not extract numeric value from formula '{fmla}'.")
                return False

            current_val = int(match.group(1))
            # Assume range 0-100000 for standard percentage adjustments
            # Flip the value: 0 becomes 100000, 100000 becomes 0, 50000 stays 50000
            flipped_val = 100000 - current_val
            new_fmla = f"val {flipped_val}"

            if fmla != new_fmla:
                logging.warning(f"[{filename}] Element #{elem_idx} (triangle): EXPERIMENTALLY flipping 'adj' value from '{fmla}' to '{new_fmla}'.")
                adj_guide.set("fmla", new_fmla)
                return True # Indicate change was made
            else:
                logging.debug(f"[{filename}] Element #{elem_idx} (triangle): 'adj' value '{fmla}' is already centered (50000), no experimental flip needed.")
                return False

        except (ValueError, TypeError) as e:
            logging.error(f"[{filename}] Element #{elem_idx} (triangle): Error processing adjustment formula '{fmla}': {e}")
            return False

    # --- Target 'rtTriangle' (Right Triangle) ---
    elif preset_type == "rtTriangle":
        try:
            original_flipH = xfrm_element.get("flipH", "0")
            original_flipV = xfrm_element.get("flipV", "0")
            # Apply flipV=1 and ensure flipH=0
            if original_flipV != "1" or original_flipH != "0":
                logging.warning(f"[{filename}] Element #{elem_idx} (rtTriangle): EXPERIMENTALLY setting flipV='1' and flipH='0' on transform.")
                xfrm_element.set("flipV", "1")
                xfrm_element.set("flipH", "0") # Ensure horizontal flip is off
                return True # Indicate change was made
            else:
                logging.debug(f"[{filename}] Element #{elem_idx} (rtTriangle): Transform already has flipV='1' and flipH='0', no experimental flip needed.")
                return False
        except Exception as e:
            logging.error(f"[{filename}] Element #{elem_idx} (rtTriangle): Error applying experimental flipV: {e}")
            return False

    else:
        logging.debug(f"[{filename}] Element #{elem_idx}: Preset '{preset_type}' not targeted by experimental flip.")
        return False


# --- Main Processing Function ---

def process_slide_xml_for_flipping(
    xml_path: str,
    slide_width_emu: int,
    options: Dict[str, bool], # Pass options dictionary
    slide_replacement_data: Optional[List[Dict[str, Any]]] = None # Keep None as default
) -> bool:
    """
    Processes a single slide/master/layout/chart XML file for layout flipping
    and text alignment adjustments based on the provided options.

    Args:
        xml_path: Path to the XML file (slide, master, layout, or chart).
        slide_width_emu: The width of the slide in EMUs (used for position mirroring).
        options: Dictionary of processing options.
        slide_replacement_data: Optional list of dicts for text replacement on this slide.

    Returns:
        True if the XML file was modified and saved, False otherwise.
    """
    try:
        # Parse the XML file
        tree = etree.parse(xml_path)
        root_elem = tree.getroot()
        overall_changed = False # Flag to track if any change occurs in the file
        filename = os.path.basename(xml_path) # For logging context
        text_id_counter = 1 # Counter for replaced text IDs in this file

        # Define correct XPath expressions once at the top of the function
        xpath_lvl_pPr = "./*[self::a:lvl1pPr or self::a:lvl2pPr or self::a:lvl3pPr or self::a:lvl4pPr or self::a:lvl5pPr or self::a:lvl6pPr or self::a:lvl7pPr or self::a:lvl8pPr or self::a:lvl9pPr]"
        xpath_lvl_pPr_descendant = ".//*[self::a:lvl1pPr or self::a:lvl2pPr or self::a:lvl3pPr or self::a:lvl4pPr or self::a:lvl5pPr or self::a:lvl6pPr or self::a:lvl7pPr or self::a:lvl8pPr or self::a:lvl9pPr]"

        # --- Check if slide is hidden ---
        # Slides have a root <p:sld> tag and are located in ppt/slides/
        # The 'show' attribute is "0" if hidden, non-existent or "1" if shown.
        is_slide_file = "ppt/slides/slide" in xml_path and xml_path.endswith(".xml")
        if is_slide_file and root_elem.tag == etree.QName(NSMAP['p'], 'sld') and root_elem.get("show") == "0":
            logging.info(f"[{filename}] Skipping hidden slide.")
            return False # Indicate no changes needed for this file, not an error

        # --- Part 0: Handle Chart XML Files Directly ---
        if root_elem.tag == etree.QName(NSMAP['c'], 'chartSpace'):
            logging.info(f"[{filename}] Detected Chart XML file. Processing for internal flip.")
            if options.get('flip_chart_internals', False):
                if _flip_chart_elements(root_elem, filename):
                    overall_changed = True
            else:
                logging.debug(f"[{filename}] Skipping internal chart flip (option disabled).")

            # Save if changed
            if overall_changed:
                logging.info(f"[{filename}] Chart modifications were made, attempting to save.")
                try:
                    tree.write(xml_path, xml_declaration=True, encoding="UTF-8", standalone=True, pretty_print=False)
                    logging.info(f"[{filename}] Successfully saved chart modifications.")
                    return True
                except Exception as e:
                    logging.error(f"[{filename}] Error writing modified chart XML file: {e}")
                    return False
            else:
                logging.info(f"[{filename}] No chart modifications needed.")
                return False
        # --- End Chart XML Handling ---


        # --- Process Slide/Layout/Master XML ---
        # Determine if the file is a master or layout (influences some processing steps)
        is_layout_file = "slideLayouts" in xml_path
        is_master_file = "slideMasters" in xml_path
        logging.debug(f"[{filename}] Processing XML file. Type: {'Master' if is_master_file else 'Layout' if is_layout_file else 'Slide'}. Options: {options}")

        # --- Part 1: Process Top-Level Drawable Elements ---
        elements_xpath = "/p:sld/p:cSld/p:spTree/* | /p:sldLayout/p:cSld/p:spTree/* | /p:sldMaster/p:cSld/p:spTree/*"
        all_top_level_children = root_elem.xpath(elements_xpath, namespaces=NSMAP)
        logging.debug(f"[{filename}] Found {len(all_top_level_children)} direct children in spTree.")
        top_level_elements = [elem for elem in all_top_level_children if elem.tag in DRAWABLE_ELEMENT_TAGS]
        logging.debug(f"[{filename}] Processing {len(top_level_elements)} drawable elements for position/flip.")
        processed_elements = set() # Keep track of elements already handled (esp. inside groups)

        for elem_idx, element in enumerate(top_level_elements):
            elem_qname = etree.QName(element.tag)
            elem_tag_name = elem_qname.localname
            elem_ns_prefix = elem_qname.namespace.split('/')[-1] # Simplified prefix extraction
            logging.debug(f"[{filename}] --- Processing top-level element #{elem_idx}: {elem_ns_prefix}:{elem_tag_name} ---")

            if element in processed_elements:
                logging.debug(f"[{filename}] Skipping element #{elem_idx} ({elem_tag_name}) - already processed (likely inside a group).")
                continue

            # --- Handle Group Shapes (p:grpSp) ---
            if element.tag == etree.QName(NSMAP['p'], 'grpSp'):
                group = element
                grp_xfrm = group.find("p:grpSpPr/a:xfrm", NSMAP)
                if grp_xfrm is None:
                    logging.warning(f"[{filename}] Group #{elem_idx} missing required transform <p:grpSpPr/a:xfrm>, skipping.")
                    processed_elements.add(group) # Mark group itself as processed
                    continue

                grp_props = _get_transform_properties(grp_xfrm)
                if grp_props is None or grp_props.get('cx', -1) < 0: # Ensure cx is valid
                    logging.warning(f"[{filename}] Skipping Group #{elem_idx} due to invalid or negative-width transform props.")
                    processed_elements.add(group)
                    continue

                group_container_changed = False
                rotation = grp_props.get('rot', 0) # Get group container rotation

                # 1. Mirror Group Container Position (if option enabled)
                if options.get('mirror_groups_pos', False):
                    original_x = grp_props['x']
                    new_x = mirror_x_around_center(original_x, grp_props['cx'], slide_width_emu)
                    grp_props['x_should_change'] = (new_x != original_x)
                    if grp_props['x_should_change']:
                        grp_props['x'] = new_x
                        logging.debug(f"[{filename}] Group #{elem_idx} position mirror calc: new x={new_x} (rotation: {rotation})")
                    # Apply position change ONLY, flip handled separately or on children
                    if _set_transform_properties(grp_props, set_flip_h=False):
                        group_container_changed = True
                        logging.info(f"[{filename}] Mirrored position for Group container #{elem_idx} (rotation: {rotation})")
                    elif grp_props['x_should_change']:
                        logging.warning(f"[{filename}] Position calculation changed for Group container #{elem_idx} but setting failed.")
                else:
                    grp_props['x_should_change'] = False # Ensure flag is false if not mirroring

                # 2. Apply FlipH and Rotation Adjustment to Group Container (if option enabled)
                apply_group_container_flip = options.get('apply_flipH_to_groups', False) # Check if group container should be flipped
                if apply_group_container_flip:
                    # Use a copy to avoid modifying original props if only flip is needed
                    temp_props_for_flip = grp_props.copy()
                    temp_props_for_flip['x'] = grp_props['x'] # Use current X
                    temp_props_for_flip['x_should_change'] = False # Only set flipH/rot
                    if _set_transform_properties(temp_props_for_flip, set_flip_h=True):
                        group_container_changed = True
                        log_msg = f"[{filename}] Applied flipH"
                        # Use the potentially updated rotation from temp_props_for_flip
                        final_rotation = temp_props_for_flip.get('rot', 0)
                        if final_rotation != 0: log_msg += f" and rotation adjustment ({final_rotation})"
                        log_msg += f" to Group container #{elem_idx}"
                        logging.info(log_msg)
                        # Update the main props dict with the potentially changed rotation
                        grp_props['rot'] = final_rotation
                else:
                    logging.debug(f"[{filename}] Group container #{elem_idx} flip disabled by option.")


                if group_container_changed: overall_changed = True
                processed_elements.add(group) # Mark group container as processed

                # 3. Process Descendants Individually (Connectors, Shapes, Pics) for Flip/Rotation
                #    *** ONLY if the group container itself was NOT flipped ***
                #    NOTE: Text alignment for descendants is handled later in Part 2, regardless of this flip logic.
                logging.debug(f"[{filename}] Group #{elem_idx}: Iterating descendants for conditional visual flip. Group container flipped={apply_group_container_flip}")
                descendant_changed_count = 0
                for descendant in group.xpath(".//*[self::p:sp or self::p:pic or self::p:cxnSp or self::p:graphicFrame]", namespaces=NSMAP): # More specific XPath
                    if descendant in processed_elements: continue # Skip if already processed (e.g., nested group)

                    desc_tag = etree.QName(descendant.tag)
                    desc_changed = False

                    # --- Conditional Visual Flip Logic for Descendants---
                    # Only attempt to visually flip descendant if the group container was NOT flipped
                    if not apply_group_container_flip:
                        apply_desc_flip = False
                        desc_xfrm = None
                        desc_spPr = None # Properties element for sp, pic, cxnSp
                        # desc_has_text = False # Check for text in descendant shapes - Handled by option name
                        is_chart_frame_in_group = False 

                        # Check if descendant is a type we might flip individually based on options
                        if desc_tag == etree.QName(NSMAP['p'], 'graphicFrame'):
                            graphic_in_group = descendant.find("a:graphic", NSMAP)
                            graphic_data_in_group = graphic_in_group.find("a:graphicData", NSMAP) if graphic_in_group is not None else None
                            if graphic_data_in_group is not None and graphic_data_in_group.get("uri") == CHART_URI:
                                is_chart_frame_in_group = True
                                logging.debug(f"[{filename}] Group #{elem_idx} Descendant ({desc_tag.localname}): Identified as Chart Frame. Skipping individual flip.")
                            # SmartArt in group - also skip individual flip if group is flipped
                            elif graphic_data_in_group is not None and graphic_data_in_group.get("uri") == DIAGRAM_URI and options.get('apply_flipH_to_smartart_frame', False):
                                logging.debug(f"[{filename}] Group #{elem_idx} Descendant ({desc_tag.localname}): SmartArt Frame. Skipping individual flip if group container is flipped.")
                                # No specific action here, relies on group flip or later SmartArt frame flip logic if not in group.

                        # Only proceed if it's not a chart frame (or other specifically handled graphic frame)
                        if not is_chart_frame_in_group : # Add other conditions if needed
                            if desc_tag == etree.QName(NSMAP['p'], 'cxnSp') and options.get('apply_flipH_to_connectors', False):
                                apply_desc_flip = True
                                desc_spPr = descendant.find("p:spPr", NSMAP)
                                if desc_spPr is not None: desc_xfrm = desc_spPr.find("a:xfrm", NSMAP)
                                logging.debug(f"[{filename}] Group #{elem_idx} Descendant ({desc_tag.localname}): Connector flip enabled (Group NOT flipped).")
                            elif desc_tag == etree.QName(NSMAP['p'], 'sp') and options.get('apply_flipH_to_shapes_no_text', False):
                                # Option name implies check for text, but current logic applies flipH regardless if option is true
                                # For shapes in groups, if group is NOT flipped, and this option is on, shape will be flipped.
                                txBody_in_group_shape = descendant.find("p:txBody", NSMAP)
                                if txBody_in_group_shape is not None:
                                     logging.warning(f"[{filename}] Group #{elem_idx} Descendant ({desc_tag.localname}): Shape contains text, but 'apply_flipH_to_shapes_no_text' is True. Text will be visually flipped.")
                                apply_desc_flip = True
                                desc_spPr = descendant.find("p:spPr", NSMAP)
                                if desc_spPr is not None: desc_xfrm = desc_spPr.find("a:xfrm", NSMAP)
                                logging.debug(f"[{filename}] Group #{elem_idx} Descendant ({desc_tag.localname}): Shape flip enabled (Group NOT flipped).")
                            elif desc_tag == etree.QName(NSMAP['p'], 'pic') and options.get('apply_flipH_to_pictures', False):
                                apply_desc_flip = True
                                desc_spPr = descendant.find("p:spPr", NSMAP)
                                if desc_spPr is not None:
                                    desc_xfrm = desc_spPr.find("a:xfrm", NSMAP)
                                    if desc_xfrm is None: logging.warning(f"[{filename}] Group #{elem_idx} Descendant ({desc_tag.localname}): <a:xfrm> not found within <p:spPr>.")
                                else: logging.warning(f"[{filename}] Group #{elem_idx} Descendant ({desc_tag.localname}): <p:spPr> not found.")
                                logging.debug(f"[{filename}] Group #{elem_idx} Descendant ({desc_tag.localname}): Picture flip enabled (Group NOT flipped).")
                            
                            # Apply visual flip if conditions met
                            if apply_desc_flip and desc_xfrm is not None:
                                desc_props = _get_transform_properties(desc_xfrm)
                                if desc_props:
                                    # IMPORTANT: Do NOT mirror position relative to group
                                    desc_props['x_should_change'] = False
                                    # Apply flipH and rotation adjustment
                                    if _set_transform_properties(desc_props, set_flip_h=True):
                                        desc_changed = True
                                        descendant_changed_count += 1
                                        logging.info(f"[{filename}] Applied visual flipH/rot adjustment to {desc_tag.localname} inside Group #{elem_idx} (Group NOT flipped).")
                                else:
                                    logging.warning(f"[{filename}] Could not get transform properties for {desc_tag.localname} inside Group #{elem_idx}.")
                            elif apply_desc_flip:
                                logging.warning(f"[{filename}] Could not find transform for {desc_tag.localname} inside Group #{elem_idx} to apply visual flip.")
                        # else:
                            # if desc_tag.localname in ['sp', 'pic', 'cxnSp']: # Only log for relevant types
                                # logging.debug(f"[{filename}] Group #{elem_idx} Descendant ({desc_tag.localname}): Skipping individual visual flip because Group container was flipped or descendant is handled differently (e.g. chart).")

                        if desc_changed: overall_changed = True
                        # Mark descendant as processed *for visual flip*, text alignment is separate
                        processed_elements.add(descendant)

                if descendant_changed_count > 0:
                    logging.info(f"[{filename}] Group #{elem_idx}: Applied individual visual flips to {descendant_changed_count} descendants (because Group container was NOT flipped).")

                logging.debug(f"[{filename}] Finished processing visual flips for descendants of Group #{elem_idx}.")
                continue # Finished processing this group and its children for Part 1

            # --- Handle Graphic Frames (p:graphicFrame) - Tables, Charts, SmartArt ---
            if element.tag == etree.QName(NSMAP['p'], 'graphicFrame'):
                frame_changed = False
                graphic = element.find("a:graphic", NSMAP)
                graphic_data = graphic.find("a:graphicData", NSMAP) if graphic is not None else None
                frame_xfrm = element.find("p:xfrm", NSMAP) # Transform for the frame itself
                frame_props = _get_transform_properties(frame_xfrm) if frame_xfrm is not None else None

                # Identify content type (Table, Chart, SmartArt)
                content_type = "Unknown"
                content_elem = None # Specific content element (e.g., a:tbl) 
                uri = None
                if graphic_data is not None:
                    uri = graphic_data.get("uri")
                    if uri == CHART_URI:
                        content_type = "Chart"
                        # Chart internals are processed separately when the chart XML is handled
                        logging.debug(f"[{filename}] Frame #{elem_idx}: Identified as Chart (processing handled separately).")
                    elif uri == DIAGRAM_URI:
                        content_type = "SmartArt"
                        logging.debug(f"[{filename}] Frame #{elem_idx}: Identified as SmartArt.")
                    else:
                        # Check for table within graphicData
                        table_check = graphic_data.find(f".//{{{NSMAP['a']}}}tbl", namespaces=NSMAP)
                        if table_check is not None:
                            content_type = "Table"
                            content_elem = table_check # Assign table element for processing
                            logging.debug(f"[{filename}] Frame #{elem_idx}: Identified as Table (within graphicData).")
                        else:
                             logging.debug(f"[{filename}] Frame #{elem_idx}: Identified as Other Graphic (URI: {uri}).")
                else:
                     logging.warning(f"[{filename}] Frame #{elem_idx}: Missing <a:graphic> or <a:graphicData>.")


                # --- Process Specific Content Types ---

                # A. Table Flipping (if it's a table and option enabled)
                if content_type == "Table" and options.get('flip_table_columns', False):
                    logging.debug(f"[{filename}] >>> Processing Table within graphicFrame #{elem_idx} for column flip <<<")
                    if content_elem is not None and _mirror_table_columns(content_elem, filename, options):
                        frame_changed = True

                    # Also attempt to set default text style for the table
                    if content_elem is not None and (options.get('right_align_text', False) or options.get('set_rtl_paragraphs', False)):
                        tblPr = content_elem.find("a:tblPr", NSMAP)
                        if tblPr is None:
                            tblPr = etree.Element(etree.QName(NSMAP['a'], 'tblPr'))
                            grid_elem = content_elem.find("a:tblGrid", NSMAP) # Corrected variable name
                            if grid_elem is not None: grid_elem.addnext(tblPr)
                            else: content_elem.insert(0, tblPr) # Fallback
                            logging.debug(f"[{filename}] Table in Frame #{elem_idx}: Created missing <a:tblPr>.")

                        defTxStyle = tblPr.find("a:defTxStyle", NSMAP)
                        if defTxStyle is None:
                            defTxStyle = etree.SubElement(tblPr, etree.QName(NSMAP['a'], 'defTxStyle'))
                            logging.debug(f"[{filename}] Table in Frame #{elem_idx}: Created missing <a:defTxStyle>.")

                        defPPr_table = defTxStyle.find("a:defPPr", NSMAP) # Renamed to avoid conflict
                        if defPPr_table is None:
                            defPPr_table = etree.SubElement(defTxStyle, etree.QName(NSMAP['a'], 'defPPr'))
                            logging.debug(f"[{filename}] Table in Frame #{elem_idx}: Created missing <a:defPPr> in defTxStyle.")

                        context_table_style = f"Table in Frame #{elem_idx} default text style"
                        if _apply_alignment_to_defPPr(defPPr_table, options, filename, context_table_style):
                            logging.info(f"[{filename}] Applied alignment/RTL to default text style for Table in Frame #{elem_idx}.")
                            frame_changed = True

                # B. Chart Flipping - Handled separately by processing the chart XML file

                # C. SmartArt Flipping (if it's SmartArt and option enabled)
                elif content_type == "SmartArt" and options.get('apply_flipH_to_smartart_frame', False):
                    logging.warning(f"[{filename}] >>> Applying EXPERIMENTAL flipH to SmartArt Frame #{elem_idx} (High Risk!) <<<")
                    if frame_props:
                        # Use a copy to only apply flipH/rotation
                        temp_props_for_flip = frame_props.copy()
                        temp_props_for_flip['x_should_change'] = False # Don't change position here
                        if _set_transform_properties(temp_props_for_flip, set_flip_h=True):
                            frame_changed = True
                            final_rotation = temp_props_for_flip.get('rot', 0)
                            log_msg = f"[{filename}] Applied flipH"
                            if final_rotation != 0: log_msg += f" and rotation adjustment ({final_rotation})"
                            log_msg += f" to SmartArt Frame #{elem_idx}"
                            logging.info(log_msg)
                            # Update main props if rotation changed
                            frame_props['rot'] = final_rotation
                        else:
                             logging.warning(f"[{filename}] Frame #{elem_idx} (SmartArt): Failed to apply flipH/rotation to frame transform.")
                    else:
                        logging.warning(f"[{filename}] Frame #{elem_idx} (SmartArt): Cannot apply flipH, missing transform properties.")


                # --- Process Frame Position Mirroring (Always check if option enabled) ---
                if options.get('mirror_frames_pos', False):
                    if frame_props and frame_props.get('cx', -1) >= 0: # Ensure cx is valid
                        rotation = frame_props.get('rot', 0)
                        original_x = frame_props['x']
                        new_x = mirror_x_around_center(original_x, frame_props['cx'], slide_width_emu)
                        frame_props['x_should_change'] = (new_x != original_x)
                        if frame_props['x_should_change']:
                            frame_props['x'] = new_x
                            logging.debug(f"[{filename}] Frame #{elem_idx} ({content_type}) position mirror calc: new x={new_x} (rotation: {rotation})")
                        # Apply position change ONLY. Flip handled above for specific types if enabled.
                        if _set_transform_properties(frame_props, set_flip_h=False):
                            frame_changed = True
                            logging.info(f"[{filename}] Mirrored position for Frame #{elem_idx} ({content_type}, rotation: {rotation})")
                        elif frame_props['x_should_change']:
                            logging.warning(f"[{filename}] Position calc changed for Frame #{elem_idx} ({content_type}) but setting failed.")
                    elif frame_xfrm is not None: # Log only if transform existed but props were bad
                        logging.warning(f"[{filename}] Skipping position mirror for Frame #{elem_idx} ({content_type}) (invalid transform props).")
                    else: # Log if transform itself was missing
                         logging.warning(f"[{filename}] Skipping position mirror for Frame #{elem_idx} ({content_type}) (missing transform <p:xfrm>).")
                else:
                    # Ensure flag is false if not mirroring
                    if frame_props: frame_props['x_should_change'] = False

                if frame_changed: overall_changed = True

                # Mark frame and its potential content as processed
                processed_elements.add(element)
                if content_elem is not None and content_type == "Table": # Only mark table content here
                    processed_elements.add(content_elem)
                    for desc in content_elem.xpath(".//*"): processed_elements.add(desc)

                logging.debug(f"[{filename}] Marked Frame #{elem_idx} ({content_type}) as processed.")
                continue # Skip rest of loop for this graphic frame

            # --- Handle Other Top-Level Shapes (sp, pic, cxnSp) ---
            xfrm = None
            spPr = None # Specific properties element for sp, pic, cxnSp
            has_text_body = False # Flag to check for text body
            is_chart_shape = False # Flag to check if shape contains a chart graphic

            # Find the shape properties and transform element
            if element.tag in [etree.QName(NSMAP['p'], 'sp'), etree.QName(NSMAP['p'], 'cxnSp')]:
                spPr = element.find("p:spPr", NSMAP)
                if spPr is not None:
                    xfrm = spPr.find("a:xfrm", NSMAP)
                    if xfrm is None:
                         logging.debug(f"[{filename}] Element #{elem_idx} ({elem_tag_name}): <a:xfrm> not found within <p:spPr>.")
                else:
                    logging.warning(f"[{filename}] Element #{elem_idx} ({elem_tag_name}) missing <p:spPr>.")
                
                if element.tag == etree.QName(NSMAP['p'], 'sp'):
                    txBody = element.find("p:txBody", NSMAP)
                    if txBody is not None: has_text_body = True
            elif element.tag == etree.QName(NSMAP['p'], 'pic'):
                spPr = element.find("p:spPr", NSMAP) 
                if spPr is not None:
                    xfrm = spPr.find("a:xfrm", NSMAP) 
                    if xfrm is None: logging.warning(f"[{filename}] Element #{elem_idx} ({elem_tag_name}): <a:xfrm> not found within <p:spPr> for picture.")
                else: logging.warning(f"[{filename}] Element #{elem_idx} ({elem_tag_name}) missing <p:spPr> for picture.")


            if xfrm is None:
                logging.warning(f"[{filename}] Could not locate transform for Element #{elem_idx} ({elem_tag_name}), skipping position/flip.")
                processed_elements.add(element) 
                continue 

            props = _get_transform_properties(xfrm)
            if props is None or props.get('cx', -1) < 0: # Ensure cx is valid
                logging.warning(f"[{filename}] Skipping Element #{elem_idx} ({elem_tag_name}) (invalid/negative-width transform props).")
                processed_elements.add(element)
                continue

            element_changed = False
            rotation = props.get('rot', 0) 

            # Determine if position should be mirrored
            should_mirror_pos = (
                (element.tag == etree.QName(NSMAP['p'], 'sp') and options.get('mirror_shapes_pos', False)) or
                (element.tag == etree.QName(NSMAP['p'], 'pic') and options.get('mirror_pictures_pos', False)) or
                (element.tag == etree.QName(NSMAP['p'], 'cxnSp') and options.get('mirror_connectors_pos', False))
            )

            if should_mirror_pos:
                original_x = props['x']
                element_width = props.get('cx', 0)
                new_x = mirror_x_around_center(original_x, element_width, slide_width_emu)
                props['x_should_change'] = (new_x != original_x)
                if props['x_should_change']:
                    props['x'] = new_x
                    logging.debug(f"[{filename}] Element #{elem_idx} ({elem_tag_name}) position mirror calc: new x={new_x} (rotation: {rotation})")
                else:
                    logging.debug(f"[{filename}] Element #{elem_idx} ({elem_tag_name}) position mirror calc: x unchanged.")
            else:
                props['x_should_change'] = False

            # Determine if standard visual flipH should be attempted
            apply_standard_flip_h = False
            if element.tag == etree.QName(NSMAP['p'], 'sp') and options.get('apply_flipH_to_shapes_no_text', False):
                graphic_check = element.find(".//a:graphic", NSMAP) # Check for nested graphic (e.g. chart in shape)
                if graphic_check is not None and graphic_check.find("a:graphicData[@uri='{CHART_URI}']", NSMAP) is not None:
                    is_chart_shape = True
                    logging.debug(f"[{filename}] Element #{elem_idx} ({elem_tag_name}): Shape contains Chart. Skipping flipH based on this check.")
                
                if has_text_body and not is_chart_shape : # Only warn if it has text AND is not a chart shape
                     logging.warning(f"[{filename}] Element #{elem_idx} ({elem_tag_name}): Applying flipH even though shape contains text (option 'apply_flipH_to_shapes_no_text' is True). Text may be visually flipped.")
                
                if not is_chart_shape: # Only apply if not a chart shape
                    apply_standard_flip_h = True

            elif element.tag == etree.QName(NSMAP['p'], 'pic') and options.get('apply_flipH_to_pictures', False):
                apply_standard_flip_h = True
            elif element.tag == etree.QName(NSMAP['p'], 'cxnSp') and options.get('apply_flipH_to_connectors', False):
                apply_standard_flip_h = True

            # --- AGGRESSIVE REBUILD OPTION ---
            should_aggressively_rebuild = (
                apply_standard_flip_h and
                options.get('aggressive_shape_rebuild', False) and
                element.tag == etree.QName(NSMAP['p'], 'sp') 
            )

            if should_aggressively_rebuild:
                logging.warning(f"[{filename}] Element #{elem_idx} ({elem_tag_name}): Attempting AGGRESSIVE REBUILD (Highest Risk!). Rotation={rotation}")
                rebuild_success = False
                try:
                    new_element = copy.deepcopy(element)
                    new_spPr = new_element.find("p:spPr", NSMAP)
                    new_xfrm = None
                    if new_spPr is not None: new_xfrm = new_spPr.find("a:xfrm", NSMAP)

                    if new_xfrm is not None:
                        new_props = _get_transform_properties(new_xfrm)
                        if new_props:
                            # Apply mirrored position
                            new_props['x_should_change'] = props['x_should_change']
                            if new_props['x_should_change']: new_props['x'] = props['x']

                            # Apply flipH and rotation adjustment
                            if _set_transform_properties(new_props, set_flip_h=True):
                                element_changed = True

                            parent = element.getparent()
                            if parent is not None:
                                parent.replace(element, new_element)
                                rebuild_success = True
                                logging.info(f"[{filename}] Element #{elem_idx} ({elem_tag_name}): Aggressive rebuild successful.")
                            else: logging.error(f"[{filename}] Element #{elem_idx} ({elem_tag_name}): Cannot rebuild, parent not found.")
                        else: logging.error(f"[{filename}] Element #{elem_idx} ({elem_tag_name}): Failed to get props from copied element during rebuild.")
                    else: logging.error(f"[{filename}] Element #{elem_idx} ({elem_tag_name}): Failed to find transform in copied element during rebuild.")
                except Exception as rebuild_err:
                    logging.error(f"[{filename}] Element #{elem_idx} ({elem_tag_name}): Error during aggressive rebuild: {rebuild_err}")

                if rebuild_success:
                    overall_changed = True
                    processed_elements.add(new_element) 
                    continue 

                logging.warning(f"[{filename}] Element #{elem_idx} ({elem_tag_name}): Aggressive rebuild failed, falling back to standard processing.")


            # --- Standard Processing / Fallback ---
            experimental_flip_applied = False
            if apply_standard_flip_h and options.get('experimental_preset_flip', False) and element.tag == etree.QName(NSMAP['p'], 'sp'):
                if _try_flip_preset_geometry(spPr, xfrm, filename, elem_idx):
                    experimental_flip_applied = True
                    element_changed = True

            set_standard_flip = apply_standard_flip_h and not experimental_flip_applied
            if _set_transform_properties(props, set_flip_h=set_standard_flip):
                element_changed = True
                log_pos_status = props.get('x_should_change', False)
                log_flip_status = set_standard_flip 
                logging.info(f"[{filename}] Applied standard changes to Element #{elem_idx} ({elem_tag_name}): Pos changed={log_pos_status}, Flip attempted={log_flip_status}. Check preceding logs for flip/rotation details.")
            elif props.get('x_should_change', False): # Position changed but setting failed
                logging.warning(f"[{filename}] Position calculation changed for Element #{elem_idx} ({elem_tag_name}) but setting failed.")

            if element_changed: overall_changed = True
            processed_elements.add(element) 
        # --- End of Part 1 ---

        # --- Part 1.5: Replace Text Content (Optional) ---
        if options.get('replace_text', False) and slide_replacement_data: # Ensure data is present
            logging.info(f"[{filename}] Starting text replacement")
            text_replaced_count = 0
            # XPath targets text bodies in shapes (p:sp) and table cells (a:tc)
            # Excludes shapes within SmartArt diagrams (ancestor::dgm:*)
            target_text_bodies = root_elem.xpath(
                ".//p:sp[p:txBody and not(ancestor::dgm:*)]/p:txBody | .//a:tc/a:txBody",
                namespaces=NSMAP
            )
            logging.debug(f"[{filename}] Found {len(target_text_bodies)} text bodies for potential replacement.")

            replacement_map = {item['id']: item['Arabic'] for item in slide_replacement_data if isinstance(item.get('id'), int) and isinstance(item.get('Arabic'), str)}
            if not replacement_map:
                 logging.warning(f"[{filename}] Text replacement enabled, but replacement map for this slide is empty or invalid. Skipping replacement for this file.")
            
            current_text_id_in_file = 1 # Use a file-local counter
            
            if replacement_map: # Proceed only if map has entries
                for txBody in target_text_bodies:
                    paragraphs = txBody.findall("./a:p", namespaces=NSMAP)
                    for p_elem in paragraphs:
                        runs = p_elem.findall("./a:r", namespaces=NSMAP)
                        paragraph_text_content = "".join(p_elem.xpath(".//a:t/text()", namespaces=NSMAP)).strip()

                        if runs and paragraph_text_content: 
                            pPr = p_elem.find("a:pPr", NSMAP)
                            preserved_pPr = copy.deepcopy(pPr) if pPr is not None else None
                            
                            first_rPr = None
                            if runs[0].find("a:rPr", NSMAP) is not None:
                                first_rPr = copy.deepcopy(runs[0].find("a:rPr", NSMAP))
                            
                            endParaRPr = p_elem.find("a:endParaRPr", NSMAP)
                            preserved_endParaRPr = copy.deepcopy(endParaRPr) if endParaRPr is not None else None

                            for child in list(p_elem): p_elem.remove(child)

                            if preserved_pPr is not None: p_elem.append(preserved_pPr)
                            
                            new_run = etree.Element(etree.QName(NSMAP['a'], 'r'))
                            if first_rPr is not None:
                                new_run.append(first_rPr) # Apply first run's style
                                # Set Arabic proofing language if option is enabled
                                if options.get('set_arabic_proofing_lang', False):
                                    first_rPr.set("lang", "ar-SA")
                                    logging.debug(f"[{filename}] Set Arabic proofing language on replaced text run.")
                                elif options.get('set_arabic_proofing_lang', False):
                                    # Create rPr with Arabic language if no existing rPr
                                    new_rPr = etree.Element(etree.QName(NSMAP['a'], 'rPr'))
                                    new_rPr.set("lang", "ar-SA")
                                    new_run.append(new_rPr)
                                    logging.debug(f"[{filename}] Created rPr with Arabic proofing language for replaced text run.")
                            new_text_elem = etree.SubElement(new_run, etree.QName(NSMAP['a'], 't'))
                            
                            replacement_text = replacement_map.get(current_text_id_in_file)
                            if replacement_text is not None:
                                new_text_elem.text = replacement_text
                                logging.debug(f"[{filename}] Replaced text for ID {current_text_id_in_file} with: '{replacement_text[:30]}...'")
                            else:
                                new_text_elem.text = paragraph_text_content 
                                logging.warning(f"[{filename}] No replacement text found for ID {current_text_id_in_file}. Using original text: '{paragraph_text_content[:30]}...'")
                            
                            # Ensure xml:space="preserve" if original text had leading/trailing spaces or multiple spaces
                            if paragraph_text_content != paragraph_text_content.strip() or "  " in paragraph_text_content:
                                from lxml.etree import XML_NS
                                new_text_elem.set(etree.QName(XML_NS, 'space'), 'preserve')

                            p_elem.append(new_run)
                            if preserved_endParaRPr is not None: p_elem.append(preserved_endParaRPr)
                            
                            current_text_id_in_file += 1
                            text_replaced_count += 1
                            overall_changed = True
            if text_replaced_count > 0:
                logging.info(f"[{filename}] Replaced text content in {text_replaced_count} paragraphs.")
        elif options.get('replace_text', False) and not slide_replacement_data:
             logging.info(f"[{filename}] Text replacement option is ON, but no replacement data was provided for this slide. Skipping text replacement.")
        # --- End of Part 1.5 ---


        # --- Part 2: Text Alignment & RTL Processing ---
        if options.get('right_align_text', False) or options.get('set_rtl_paragraphs', False):
            logging.info(f"[{filename}] Starting text alignment/RTL/padding/indentation processing...")
            text_alignment_changed_count = 0
            text_rtl_set_count = 0
            padding_swapped_count = 0 
            indent_flipped_count = 0 
            rebuilt_para_count = 0
            cell_props_changed_count = 0 

            target_shapes_xpath = ".//p:sp[p:txBody and not(ancestor::dgm:*)] | .//a:tc[a:txBody]"
            target_elements_with_text = root_elem.xpath(target_shapes_xpath, namespaces=NSMAP)
            logging.info(f"[{filename}] Text Processing: Found {len(target_elements_with_text)} elements for text alignment/RTL/padding/indentation.")

            for elem_idx, text_container_elem in enumerate(target_elements_with_text):
                is_shape = text_container_elem.tag == etree.QName(NSMAP['p'], 'sp')
                is_table_cell = text_container_elem.tag == etree.QName(NSMAP['a'], 'tc')
                is_in_group = is_shape and any(ancestor.tag == etree.QName(NSMAP['p'], 'grpSp') for ancestor in text_container_elem.iterancestors())
                context_log_prefix = f"Element #{elem_idx}{' (Shape' + (' in Group' if is_in_group else '') + ')' if is_shape else ' (TableCell)' if is_table_cell else ''}"

                txBody = text_container_elem.find("p:txBody", NSMAP) if is_shape else text_container_elem.find("a:txBody", NSMAP)
                if txBody is None:
                    logging.debug(f"[{filename}] {context_log_prefix}: Skipping, no direct txBody found.")
                    continue

                element_text_processing_changed = False 
                logging.debug(f"[{filename}] Processing text in {context_log_prefix}.")

                # --- 2a. Process Text Body Padding (<a:bodyPr>) ---
                bodyPr = txBody.find("a:bodyPr", NSMAP)
                if bodyPr is not None:
                    if options.get('right_align_text', False) or options.get('set_rtl_paragraphs', False):
                        original_lIns_str = bodyPr.get("lIns")
                        original_rIns_str = bodyPr.get("rIns")
                        lIns_val, rIns_val = 0, 0
                        try:
                            if original_lIns_str is not None: lIns_val = int(original_lIns_str)
                        except ValueError: logging.warning(f"[{filename}] {context_log_prefix}: Invalid lIns value '{original_lIns_str}', using 0.")
                        try:
                            if original_rIns_str is not None: rIns_val = int(original_rIns_str)
                        except ValueError: logging.warning(f"[{filename}] {context_log_prefix}: Invalid rIns value '{original_rIns_str}', using 0.")
                        
                        new_lIns_str, new_rIns_str = str(rIns_val), str(lIns_val)
                        made_body_pr_change = False
                        if bodyPr.get("lIns", "0") != new_lIns_str: bodyPr.set("lIns", new_lIns_str); made_body_pr_change = True
                        if bodyPr.get("rIns", "0") != new_rIns_str: bodyPr.set("rIns", new_rIns_str); made_body_pr_change = True
                        
                        if made_body_pr_change:
                            logging.info(f"[{filename}] {context_log_prefix}: Swapped bodyPr lIns/rIns. Original: lIns='{original_lIns_str}', rIns='{original_rIns_str}' -> New: lIns='{new_lIns_str}', rIns='{new_rIns_str}'.")
                            element_text_processing_changed = True
                            padding_swapped_count +=1
                else:
                    logging.debug(f"[{filename}] {context_log_prefix}: No <a:bodyPr> found.")


                # --- 2b. Process individual paragraphs (<a:p>) ---
                paragraph_elems = txBody.findall("./a:p", namespaces=NSMAP)
                p_idx_loop = 0
                while p_idx_loop < len(paragraph_elems): # Use while for potential aggressive rebuild
                    p_elem = paragraph_elems[p_idx_loop]
                    pPr_modified_flag = False # Tracks if pPr was modified for this paragraph
                    pPr_rtl_set_here = False  # Tracks if RTL was set specifically in this iteration for this pPr

                    pPr = p_elem.find("a:pPr", NSMAP)
                    if pPr is None:
                        # Create pPr if options require it and it's missing
                        if options.get('right_align_text', False) or options.get('set_rtl_paragraphs', False):
                            first_content_child = next((child for child in p_elem if child.tag != etree.Comment), None)
                            pPr = etree.Element(etree.QName(NSMAP['a'], 'pPr'))
                            if first_content_child is not None: first_content_child.addprevious(pPr)
                            else: p_elem.append(pPr)
                            logging.debug(f"[{filename}] {context_log_prefix} Paragraph #{p_idx_loop}: Created missing <a:pPr>.")
                            pPr_modified_flag = True # Creating pPr is a change
                    
                    # Ensure pPr exists before proceeding with alignment/RTL/indent
                    if pPr is not None:
                        initial_algn_p = pPr.get("algn")
                        forced_right_alignment_p = False

                        if options.get('right_align_text', False):
                            if initial_algn_p in [None, "l", "just", "dist"]:
                                pPr.set("algn", "r")
                                logging.debug(f"[{filename}] {context_log_prefix} Paragraph #{p_idx_loop}: Forced algn='r' (was '{initial_algn_p}')")
                                pPr_modified_flag = True
                                forced_right_alignment_p = True
                                text_alignment_changed_count +=1
                        
                        final_algn_p_for_rtl = pPr.get("algn") # Re-check after potential algn change
                        apply_rtl_p = options.get('set_rtl_paragraphs', False) and final_algn_p_for_rtl != "ctr"
                        if apply_rtl_p:
                            if pPr.get("rtl") != "1":
                                pPr.set("rtl", "1")
                                logging.debug(f"[{filename}] {context_log_prefix} Paragraph #{p_idx_loop}: Forced rtl='1' (algn is '{final_algn_p_for_rtl}')")
                                pPr_modified_flag = True
                                pPr_rtl_set_here = True
                                text_rtl_set_count +=1
                        
                        # Flip indentation if RTL was set (or already set and option is true)
                        if (pPr_rtl_set_here or (pPr.get("rtl") == "1" and options.get('set_rtl_paragraphs', False))):
                            if _flip_paragraph_indentation(pPr, filename, f"{context_log_prefix} Paragraph #{p_idx_loop}"):
                                pPr_modified_flag = True
                                indent_flipped_count +=1
                        
                        # Aggressive Paragraph Rebuild (Optional)
                        if forced_right_alignment_p and options.get('aggressive_paragraph_rebuild', False):
                            logging.warning(f"[{filename}] {context_log_prefix} Paragraph #{p_idx_loop}: AGGRESSIVE rebuild (High Risk!).")
                            try:
                                new_p_elem_rebuild = etree.Element(p_elem.tag, nsmap=p_elem.nsmap)
                                new_p_elem_rebuild.append(copy.deepcopy(pPr)) 
                                for child_rebuild in p_elem:
                                    if child_rebuild.tag != etree.QName(NSMAP['a'], 'pPr'): new_p_elem_rebuild.append(copy.deepcopy(child_rebuild))
                                parent_rebuild = p_elem.getparent()
                                if parent_rebuild is not None:
                                    parent_rebuild.replace(p_elem, new_p_elem_rebuild)
                                    paragraph_elems[p_idx_loop] = new_p_elem_rebuild 
                                    p_elem = new_p_elem_rebuild 
                                    logging.info(f"[{filename}] {context_log_prefix} Paragraph #{p_idx_loop}: Aggressive rebuild completed.")
                                    rebuilt_para_count += 1
                                    pPr_modified_flag = True 
                            except Exception as rebuild_error_p: logging.error(f"[{filename}] {context_log_prefix} Paragraph #{p_idx_loop}: Error during aggressive rebuild: {rebuild_error_p}")
                    
                    if pPr_modified_flag: element_text_processing_changed = True
                    p_idx_loop += 1

                # --- 2c. Process List Styles (<a:lstStyle>) ---
                lstStyle = txBody.find("a:lstStyle", NSMAP)
                if lstStyle is not None and (options.get('right_align_text', False) or options.get('set_rtl_paragraphs', False)):
                    logging.debug(f"[{filename}] {context_log_prefix}: Processing <a:lstStyle>.")
                    list_style_levels_changed_count = 0
                    # Process direct level properties (lvl1pPr etc.) and their defPPr
                    for lvl_pPr_elem in lstStyle.xpath(xpath_lvl_pPr, namespaces=NSMAP):
                        lvl_name = etree.QName(lvl_pPr_elem.tag).localname
                        lvl_context = f"{context_log_prefix} ListStyle {lvl_name}"
                        lvl_prop_changed_here = False
                        lvl_rtl_set_here = False

                        current_lvl_algn = lvl_pPr_elem.get("algn")
                        if options.get('right_align_text', False):
                            if current_lvl_algn in [None, "l", "just", "dist"]:
                                lvl_pPr_elem.set("algn", "r")
                                logging.debug(f"[{filename}] {lvl_context}: Forced algn='r' (was '{current_lvl_algn}')")
                                lvl_prop_changed_here = True
                        
                        final_lvl_algn_for_rtl = lvl_pPr_elem.get("algn")
                        apply_rtl_lvl = options.get('set_rtl_paragraphs', False) and final_lvl_algn_for_rtl != "ctr"
                        if apply_rtl_lvl:
                            if lvl_pPr_elem.get("rtl") != "1":
                                lvl_pPr_elem.set("rtl", "1")
                                logging.debug(f"[{filename}] {lvl_context}: Forced rtl='1' (algn is '{final_lvl_algn_for_rtl}')")
                                lvl_prop_changed_here = True
                                lvl_rtl_set_here = True
                        
                        # Flip indentation for the lvlXpPr itself if RTL was set/is true
                        if (lvl_rtl_set_here or (lvl_pPr_elem.get("rtl") == "1" and options.get('set_rtl_paragraphs', False))):
                             if _flip_paragraph_indentation(lvl_pPr_elem, filename, lvl_context):
                                lvl_prop_changed_here = True
                                # indent_flipped_count is incremented by the helper

                        # Process defPPr within this lvlXpPr
                        defPPr_in_lvl = lvl_pPr_elem.find("a:defPPr", NSMAP)
                        if defPPr_in_lvl is None and (options.get('right_align_text', False) or options.get('set_rtl_paragraphs', False)): # Create if needed
                            defPPr_in_lvl = etree.SubElement(lvl_pPr_elem, etree.QName(NSMAP['a'], 'defPPr'))
                            logging.debug(f"[{filename}] {lvl_context}: Created missing <a:defPPr>.")
                            lvl_prop_changed_here = True # Creating is a change
                        
                        if defPPr_in_lvl is not None: # Now process it (either existing or newly created)
                           if _apply_alignment_to_defPPr(defPPr_in_lvl, options, filename, f"{lvl_context} defPPr"):
                               lvl_prop_changed_here = True
                        
                        if lvl_prop_changed_here: list_style_levels_changed_count +=1
                    
                    if list_style_levels_changed_count > 0:
                        element_text_processing_changed = True
                        # list_style_override_count is incremented in main summary

                # --- 2d. Process Parent Table Cell Properties (if applicable) ---
                if is_table_cell: # This part is now handled by _mirror_table_columns directly
                    pass # tcPr changes are done within _mirror_table_columns

                # --- 2e. Set Arabic Proofing Language (if applicable) ---
                if _set_arabic_proofing_language(text_container_elem, options, filename, context_log_prefix):
                    element_text_processing_changed = True

                if element_text_processing_changed: overall_changed = True

            # Log summary counts for text processing (moved outside the loop)
            if text_alignment_changed_count > 0: logging.info(f"[{filename}] Summary: Applied direct right-alignment to {text_alignment_changed_count} paragraphs.")
            if text_rtl_set_count > 0: logging.info(f"[{filename}] Summary: Set direct rtl='1' attribute for {text_rtl_set_count} non-centered paragraphs.")
            if padding_swapped_count > 0: logging.info(f"[{filename}] Summary: Swapped lIns/rIns padding for {padding_swapped_count} text bodies.")
            if indent_flipped_count > 0: logging.info(f"[{filename}] Summary: Flipped/adjusted paragraph indent for {indent_flipped_count} instances.")
            # list_style_override_count is handled by summing up list_style_levels_changed_count from each element
            if cell_props_changed_count > 0: logging.info(f"[{filename}] Summary: Applied anchor/rtl changes to {cell_props_changed_count} table cell properties (<a:tcPr>).") # This might be redundant if table function handles it
            if rebuilt_para_count > 0: logging.warning(f"[{filename}] Summary: Aggressively rebuilt {rebuilt_para_count} paragraphs (High Risk Option Enabled).")


        # --- Part 3: Process Masters & Layouts (Default Alignment for Placeholders/Styles) ---
        if (is_layout_file or is_master_file) and (options.get('right_align_text', False) or options.get('set_rtl_paragraphs', False)):
            logging.debug(f"[{filename}] Processing default styles/placeholders for alignment/RTL/indentation...")
            master_layout_defpr_changed_count = 0
            master_layout_padding_changed_count = 0
            master_layout_indent_changed_count = 0

            # A. Layout Placeholders & Master Text Styles (p:txStyles -> titleStyle, bodyStyle, otherStyle)
            # Common structure: style_element -> lvlXpPr -> defPPr
            # Also: style_element -> defPPr (for overall default)
            
            style_elements_to_check = []
            if is_layout_file:
                # For layouts, check txBody of placeholder shapes directly
                placeholder_txBodies = root_elem.xpath(".//p:sp[p:nvSpPr/p:nvPr/p:ph]/p:txBody", namespaces=NSMAP)
                for ph_txBody_idx, ph_txBody in enumerate(placeholder_txBodies):
                    ph_context = f"Layout Placeholder txBody #{ph_txBody_idx}"
                    # Process bodyPr for padding
                    bodyPr_ph = ph_txBody.find("a:bodyPr", NSMAP)
                    if bodyPr_ph is not None:
                        # (Same bodyPr swapping logic as in Part 2a)
                        original_lIns_str_ph = bodyPr_ph.get("lIns")
                        original_rIns_str_ph = bodyPr_ph.get("rIns")
                        lIns_val_ph, rIns_val_ph = 0,0
                        try:
                            if original_lIns_str_ph is not None: lIns_val_ph = int(original_lIns_str_ph)
                        except ValueError: pass
                        try:
                            if original_rIns_str_ph is not None: rIns_val_ph = int(original_rIns_str_ph)
                        except ValueError: pass
                        
                        new_lIns_str_ph, new_rIns_str_ph = str(rIns_val_ph), str(lIns_val_ph)
                        made_ph_bodypr_change = False
                        if bodyPr_ph.get("lIns", "0") != new_lIns_str_ph: bodyPr_ph.set("lIns", new_lIns_str_ph); made_ph_bodypr_change = True
                        if bodyPr_ph.get("rIns", "0") != new_rIns_str_ph: bodyPr_ph.set("rIns", new_rIns_str_ph); made_ph_bodypr_change = True
                        if made_ph_bodypr_change:
                            logging.info(f"[{filename}] {ph_context}: Swapped bodyPr lIns/rIns.")
                            master_layout_padding_changed_count +=1
                            overall_changed = True
                    
                    # Process lstStyle within placeholder txBody
                    lstStyle_ph = ph_txBody.find("a:lstStyle", NSMAP)
                    if lstStyle_ph is None: # Ensure lstStyle exists if needed for modifications
                        if bodyPr_ph is not None: lstStyle_ph = etree.Element(etree.QName(NSMAP['a'], 'lstStyle')); bodyPr_ph.addnext(lstStyle_ph)
                        else: lstStyle_ph = etree.SubElement(ph_txBody, etree.QName(NSMAP['a'], 'lstStyle'))
                        logging.debug(f"[{filename}] {ph_context}: Created missing <a:lstStyle>.")
                        overall_changed = True
                    
                    # Process lvlXpPr and their defPPr within this lstStyle
                    for lvl_pPr_ph in lstStyle_ph.xpath(xpath_lvl_pPr, namespaces=NSMAP):
                        lvl_name_ph = etree.QName(lvl_pPr_ph.tag).localname
                        lvl_ph_context = f"{ph_context} ListStyle {lvl_name_ph}"
                        lvl_ph_changed = False
                        lvl_ph_rtl_set = False
                        # Apply to lvlXpPr itself
                        if _apply_alignment_to_defPPr(lvl_pPr_ph, options, filename, lvl_ph_context): # Reuses defPPr logic for lvlXpPr
                             lvl_ph_changed = True
                        if lvl_pPr_ph.get("rtl") == "1": lvl_ph_rtl_set = True # Check if RTL was set

                        # And its defPPr child
                        defPPr_lvl_ph = lvl_pPr_ph.find("a:defPPr", NSMAP)
                        if defPPr_lvl_ph is None : # Create if needed
                            defPPr_lvl_ph = etree.SubElement(lvl_pPr_ph, etree.QName(NSMAP['a'], 'defPPr'))
                            lvl_ph_changed = True
                        if _apply_alignment_to_defPPr(defPPr_lvl_ph, options, filename, f"{lvl_ph_context} defPPr"):
                            lvl_ph_changed = True
                        if defPPr_lvl_ph.get("rtl") == "1": lvl_ph_rtl_set = True
                        
                        if lvl_ph_changed: master_layout_defpr_changed_count +=1; overall_changed = True
                        if lvl_ph_rtl_set: master_layout_indent_changed_count +=1 # Count if RTL implies indent flip
                    
                    # --- Set Arabic proofing language on placeholder if option enabled ---
                    if _set_arabic_proofing_language(ph_txBody, options, filename, f"Layout Placeholder #{ph_txBody_idx}"):
                        overall_changed = True

            elif is_master_file:
                txStyles = root_elem.find(".//p:txStyles", namespaces=NSMAP)
                if txStyles is not None:
                    style_elements_to_check.extend(txStyles.findall("./p:titleStyle", NSMAP))
                    style_elements_to_check.extend(txStyles.findall("./p:bodyStyle", NSMAP))
                    style_elements_to_check.extend(txStyles.findall("./p:otherStyle", NSMAP))
                
                for style_elem_idx, style_elem in enumerate(style_elements_to_check):
                    style_name = etree.QName(style_elem.tag).localname
                    master_style_context = f"Master {style_name} #{style_elem_idx}"
                    
                    # Process bodyPr for padding (if exists directly under style element)
                    bodyPr_master = style_elem.find("a:bodyPr", NSMAP)
                    if bodyPr_master is not None:
                         # (Same bodyPr swapping logic as in Part 2a)
                        original_lIns_str_ms = bodyPr_master.get("lIns")
                        original_rIns_str_ms = bodyPr_master.get("rIns")
                        lIns_val_ms, rIns_val_ms = 0,0
                        try:
                            if original_lIns_str_ms is not None: lIns_val_ms = int(original_lIns_str_ms)
                        except ValueError: pass
                        try:
                            if original_rIns_str_ms is not None: rIns_val_ms = int(original_rIns_str_ms)
                        except ValueError: pass
                        
                        new_lIns_str_ms, new_rIns_str_ms = str(rIns_val_ms), str(lIns_val_ms)
                        made_ms_bodypr_change = False
                        if bodyPr_master.get("lIns", "0") != new_lIns_str_ms: bodyPr_master.set("lIns", new_lIns_str_ms); made_ms_bodypr_change = True
                        if bodyPr_master.get("rIns", "0") != new_rIns_str_ms: bodyPr_master.set("rIns", new_rIns_str_ms); made_ms_bodypr_change = True
                        if made_ms_bodypr_change:
                            logging.info(f"[{filename}] {master_style_context}: Swapped bodyPr lIns/rIns.")
                            master_layout_padding_changed_count +=1
                            overall_changed = True

                    # Process lvlXpPr and their defPPr within this master style
                    for lvl_pPr_master in style_elem.xpath(xpath_lvl_pPr_descendant, namespaces=NSMAP):
                        lvl_name_master = etree.QName(lvl_pPr_master.tag).localname
                        lvl_master_context = f"{master_style_context}/{lvl_name_master}"
                        lvl_master_changed = False
                        lvl_master_rtl_set = False

                        if _apply_alignment_to_defPPr(lvl_pPr_master, options, filename, lvl_master_context):
                             lvl_master_changed = True
                        if lvl_pPr_master.get("rtl") == "1": lvl_master_rtl_set = True

                        defPPr_lvl_master = lvl_pPr_master.find("a:defPPr", NSMAP)
                        if defPPr_lvl_master is None :
                            defPPr_lvl_master = etree.SubElement(lvl_pPr_master, etree.QName(NSMAP['a'], 'defPPr'))
                            lvl_master_changed = True
                        if _apply_alignment_to_defPPr(defPPr_lvl_master, options, filename, f"{lvl_master_context} defPPr"):
                            lvl_master_changed = True
                        if defPPr_lvl_master.get("rtl") == "1": lvl_master_rtl_set = True

                        if lvl_master_changed: master_layout_defpr_changed_count +=1; overall_changed = True
                        if lvl_master_rtl_set: master_layout_indent_changed_count +=1
                
                if txStyles is not None:
                    # --- Set Arabic proofing language on master styles if option enabled ---
                    if _set_arabic_proofing_language_on_master_styles(txStyles, options, filename):
                        overall_changed = True
            
            if master_layout_defpr_changed_count > 0: logging.info(f"[{filename}] Applied default style changes (algn/rtl) to {master_layout_defpr_changed_count} master/layout pPr elements.")
            if master_layout_padding_changed_count > 0: logging.info(f"[{filename}] Swapped padding in {master_layout_padding_changed_count} master/layout <a:bodyPr> elements.")
            if master_layout_indent_changed_count > 0: logging.info(f"[{filename}] Flipped/adjusted indentation for {master_layout_indent_changed_count} master/layout paragraph property sets due to RTL.")


        # --- Part 4: Save Changes ---
        logging.info(f"[{filename}] Reached end of processing. Final 'overall_changed' flag is: {overall_changed}")

        if overall_changed:
            logging.info(f"[{filename}] Modifications were made, attempting to save.")
            try:
                tree.write(xml_path, xml_declaration=True, encoding="UTF-8", standalone=True, pretty_print=False) # standalone="yes" is also common
                logging.info(f"[{filename}] Successfully saved modifications.")
                return True 
            except Exception as e:
                logging.error(f"[{filename}] Error writing modified XML file: {e}")
                return False 
        else:
            logging.info(f"[{filename}] No modifications needed (overall_changed=False).")
            return False 

    except etree.XMLSyntaxError as e:
        logging.error(f"Error parsing XML file {os.path.basename(xml_path)}: {e}")
    except Exception as e:
        logging.error(f"Unexpected error processing file {os.path.basename(xml_path)}: {e}")

    return False # Return False if any exception occurred


# --- PPTX Processing Orchestration ---

def find_chart_xml_files(extract_dir: str, slide_layout_master_files: List[str]) -> Set[str]:
    """
    Finds chart XML file paths referenced from slide/layout/master relationship files.
    CORRECTED to properly resolve chart file paths.

    Args:
        extract_dir: The root directory where the PPTX was extracted.
        slide_layout_master_files: List of paths to slide/layout/master XML files.

    Returns:
        A set of unique absolute paths to the chart XML files found.
    """
    chart_files = set()
    # NSMAP and CHART_REL_TYPE should be accessible here (e.g., defined globally)

    for xml_file_path_str in slide_layout_master_files:
        try:
            xml_file = Path(xml_file_path_str) # Path to the source part (e.g., slideX.xml)
            # Relationship file is in a subdirectory "_rels" next to the source part,
            # and named like "slideX.xml.rels"
            rels_file = xml_file.parent / "_rels" / (xml_file.name + ".rels")

            if rels_file.exists():
                logging.debug(f"Scanning relationships file: {rels_file}")
                rels_tree = etree.parse(str(rels_file))
                # Find relationships pointing to charts
                chart_rels = rels_tree.xpath(
                    f"//rel:Relationship[@Type='{CHART_REL_TYPE}']",
                    namespaces=NSMAP # Ensure NSMAP includes 'rel'
                )
                for rel in chart_rels:
                    target = rel.get("Target")
                    if target:
                        # *** THIS IS THE CORRECTED PATH RESOLUTION ***
                        # The Target URI is relative to the source part's directory.
                        # For example, if xml_file is "ppt/slides/slide1.xml" and Target is "../charts/chart1.xml",
                        # the resulting path should be "ppt/charts/chart1.xml".
                        source_part_dir = xml_file.parent # Directory of the source part (e.g., .../ppt/slides/)
                        abs_target_path = (source_part_dir / target).resolve()
                        # *** END OF CORRECTION HIGHLIGHT ***

                        # Ensure the path is within the extract_dir for safety and is a file
                        if abs_target_path.is_file() and str(abs_target_path).startswith(str(Path(extract_dir).resolve())):
                            chart_files.add(str(abs_target_path))
                            logging.debug(f"  Found chart relationship target: {target} -> {abs_target_path}")
                        else:
                             logging.warning(f"  Skipping relationship target: '{target}'. Resolved path '{abs_target_path}' is not a file or is outside extract_dir '{Path(extract_dir).resolve()}'. Source .rels: {rels_file}")
            else:
                logging.debug(f"No relationships file found for {xml_file.name} at {rels_file}")

        except etree.XMLSyntaxError as e:
            logging.error(f"Error parsing relationships file {str(rels_file) if 'rels_file' in locals() else xml_file_path_str + '.rels'}: {e}")
        except Exception as e:
            logging.error(f"Error processing relationships for {xml_file_path_str}: {e}")

    if chart_files:
        logging.info(f"Found {len(chart_files)} unique chart XML files referenced: {[Path(f).name for f in chart_files]}")
    else:
        logging.warning("Found 0 unique chart XML files referenced. Chart internals will not be flipped.")
    return chart_files


def find_all_xml_files_to_process(extract_dir: str, process_masters_layouts: bool) -> List[str]:
    """
    Finds all relevant XML files (slides, layouts, masters, and charts)
    within the extracted PPTX directory structure.
    """
    xml_files = []
    ppt_dir = os.path.join(extract_dir, "ppt")

    # 1. Find Slides, Layouts, Masters
    slides_dir = os.path.join(ppt_dir, "slides")
    if os.path.isdir(slides_dir):
        for fname in os.listdir(slides_dir):
            # Ensure it's slideX.xml, not slideX.xml.rels
            if fname.lower().startswith("slide") and fname.lower().endswith(".xml") and ".xml.rels" not in fname.lower():
                xml_files.append(os.path.join(slides_dir, fname))

    if process_masters_layouts:
        logging.info("Processing scope includes: Slides, Slide Masters, and Slide Layouts.")
        for sub_dir in ["slideMasters", "slideLayouts"]:
            dir_path = os.path.join(ppt_dir, sub_dir)
            if os.path.isdir(dir_path):
                for fname in os.listdir(dir_path):
                    # Ensure it's *.xml, not *.xml.rels
                    if fname.lower().endswith(".xml") and ".xml.rels" not in fname.lower():
                        xml_files.append(os.path.join(dir_path, fname))
    else:
        logging.info("Processing scope includes: Slides only.")

    logging.info(f"Found {len(xml_files)} slide/layout/master XML files.")
    slide_layout_master_files = list(xml_files) # Keep a copy for chart finding

    # 2. Find Chart files referenced by the above
    # Pass the absolute path of the extract_dir
    chart_files = find_chart_xml_files(extract_dir, slide_layout_master_files)

    # 3. Combine and ensure uniqueness
    all_files = set(xml_files)
    all_files.update(chart_files)

    logging.info(f"Found total {len(all_files)} unique XML files to process: {[os.path.basename(f) for f in all_files]}")
    return list(all_files)


def ui_flip_pptx_layout(
    pptx_path: str,
    options: Dict[str, bool]
) -> Generator[Tuple[Optional[str], str, str], None, None]:
    """Main orchestrator function to flip a PPTX file layout."""
    temp_dir = None
    files_changed_count = 0
    
    log_stream = io.StringIO()
    stream_handler = None

    try:
        # Setup in-memory logging handler for Gradio output
        stream_handler = logging.StreamHandler(log_stream)
        stream_handler.setFormatter(log_formatter)
        # Ensure handler captures messages based on root logger level (set later)
        stream_handler.setLevel(root_logger.level)
        root_logger.addHandler(stream_handler)

        logging.info(f"--- Starting PPTX Flip Process for: {os.path.basename(pptx_path)} ---")
        logging.info(f"Options: {options}")

        # Create unique temporary directory
        temp_dir = os.path.join(tempfile.gettempdir(), f"ppt_flip_{uuid.uuid4().hex}")
        os.makedirs(temp_dir, exist_ok=True)
        logging.info(f"Created temporary directory: {temp_dir}")
        log_stream.flush()
        yield (None, "10% - Extracting PPTX contents...", log_stream.getvalue())

        # Extract PPTX contents
        try:
            with zipfile.ZipFile(pptx_path, "r") as z_in:
                z_in.extractall(temp_dir)
            logging.info("PPTX extracted successfully.")
        except zipfile.BadZipFile:
            logging.error("BadZipFile error during extraction. Input file might be corrupted or not a PPTX.")
            log_stream.flush()
            yield (None, "❌ Error: Invalid or corrupted PPTX file.", log_stream.getvalue())
            return
        except Exception as e:
            logging.exception("Error during PPTX extraction:")
            log_stream.flush()
            yield (None, f"❌ Error extracting PPTX: {str(e)}", log_stream.getvalue())
            return

        log_stream.flush()
        yield (None, "20% - Reading presentation properties...", log_stream.getvalue())

        # Get slide dimensions
        presentation_xml_path = os.path.join(temp_dir, "ppt", "presentation.xml")
        slide_width_emu, slide_height_emu = DEFAULT_SLIDE_WIDTH_EMU, DEFAULT_SLIDE_HEIGHT_EMU
        if os.path.exists(presentation_xml_path):
            try:
                pres_tree = etree.parse(presentation_xml_path)
                pres_root = pres_tree.getroot()
                slide_width_emu, slide_height_emu = get_slide_size_from_presentation(pres_root)
            except Exception as e:
                logging.warning(f"Could not parse presentation.xml or get slide size: {e}. Using defaults.")
        else:
            logging.warning("presentation.xml not found. Using default slide size.")

        log_stream.flush()
        yield (None, f"30% - Using slide width {slide_width_emu} EMU. Locating XML files (incl. charts)...", log_stream.getvalue()) # Updated message

        # Find ALL XML files to process based on options (incl. charts)
        process_masters_layouts = options.get('process_masters_layouts', True)
        # Pass the absolute path to the temporary directory
        xml_files_to_process = find_all_xml_files_to_process(temp_dir, process_masters_layouts)
        total_files = len(xml_files_to_process)

        if total_files == 0:
            logging.warning("No XML files found to process based on selected options.")
            log_stream.flush()
            yield (None, "⚠️ Warning: No XML files found to process based on options. Re-zipping.", log_stream.getvalue())
        else:
            log_stream.flush()
            yield (None, f"40% - Found {total_files} XML files (incl. charts). Starting processing...", log_stream.getvalue()) # Updated message

            # Process XML files in parallel
            processed_count = 0
            files_changed_count = 0
            # Limit workers to avoid overwhelming system resources
            max_workers = min(os.cpu_count() or 1, 8)
            logging.info(f"Using up to {max_workers} worker threads for XML processing.")

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Create a dictionary mapping futures to their corresponding file paths
                future_to_path = {
                    executor.submit(process_slide_xml_for_flipping, path, slide_width_emu, options): path
                    for path in xml_files_to_process # Use the combined list
                }

                # Process completed futures
                for future in as_completed(future_to_path):
                    path = future_to_path[future]
                    filename = os.path.basename(path)
                    processed_count += 1

                    try:
                        # Get the result (True if changed, False otherwise)
                        changed_flag = future.result()
                        if changed_flag: files_changed_count += 1
                        logging.debug(f"Future completed for {filename}. Changed: {changed_flag}")
                    except Exception as e:
                        # Log errors from individual file processing tasks
                        logging.error(f"Error processing future for {filename}: {e}")

                    # Update progress periodically
                    progress_percent = 40 + (50 * (processed_count / total_files if total_files > 0 else 0))
                    # Update roughly every 5% or on the last file
                    update_interval = max(1, total_files // 20)
                    if processed_count % update_interval == 0 or processed_count == total_files:
                        log_stream.flush()
                        yield (None, f"{int(progress_percent)}% - Processed {processed_count}/{total_files} XML files ({files_changed_count} modified).", log_stream.getvalue())

        log_stream.flush()
        yield (None, "90% - Re-zipping the PPTX file...", log_stream.getvalue())

        # Create the output PPTX file path
        base_name = os.path.basename(pptx_path)
        name_part, ext_part = os.path.splitext(base_name)
        # Sanitize filename part for safety
        safe_name_part = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in name_part)
        output_filename = f"flipped_{safe_name_part}{ext_part}"
        # Save output in a standard temporary location
        flipped_pptx_path = os.path.join(tempfile.gettempdir(), output_filename)
        files_skipped_zipping = 0

        # Re-zip the modified contents
        try:
            with zipfile.ZipFile(flipped_pptx_path, "w", zipfile.ZIP_DEFLATED) as z_out:
                # Walk through the temporary directory
                for root_dir, _, files_in_dir in os.walk(temp_dir):
                    for filename in files_in_dir:
                        full_path = None
                        try:
                            full_path = os.path.join(root_dir, filename)
                            # Calculate the relative path for the archive
                            relative_path = os.path.relpath(full_path, temp_dir)
                            # Write the file to the zip archive
                            z_out.write(full_path, arcname=relative_path)
                        except OSError as oe:
                            logging.error(f"OS Error writing file to zip: '{filename}' from '{full_path}'. Error: {oe}")
                            files_skipped_zipping += 1
                        except Exception as e_zip:
                            logging.exception(f"Unexpected error writing file to zip: '{filename}' from '{full_path}'.")
                            files_skipped_zipping += 1
        except Exception as e:
            logging.exception("Critical error during re-zipping process:")
            log_stream.flush()
            yield(None, f"❌ Error during re-zipping: {str(e)}", log_stream.getvalue())
            return

        logging.info(f"Successfully created flipped PPTX: {flipped_pptx_path}")
        completion_message = f"✅ Processing complete. {files_changed_count} XML files modified."
        if files_skipped_zipping > 0:
            completion_message += f" ⚠️ Skipped adding {files_skipped_zipping} file(s) to the output due to errors (check logs)."
        log_stream.flush()
        yield (flipped_pptx_path, completion_message, log_stream.getvalue())

    except Exception as e:
        # Catch any top-level errors in the orchestration
        logging.exception("An unexpected error occurred during PPTX processing orchestration.")
        log_stream.flush()
        yield (None, f"❌ An unexpected error occurred: {str(e)}", log_stream.getvalue())
    finally:
        # Cleanup: Remove the log handler and temporary directory
        if stream_handler:
            root_logger.removeHandler(stream_handler)
            stream_handler.close()
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                logging.info(f"Cleaned up temporary directory: {temp_dir}")
            except Exception as e:
                logging.error(f"Error removing temporary directory {temp_dir}: {e}")
        logging.info("--- PPTX Flip Process Finished ---")

def flip_pptx_layout(
    pptx_path: str,
    options: Dict[str, bool],
    output_dir: Optional[str] = None,
    output_filename: Optional[str] = None,
    replacement_data: Optional[List[List[Dict[str, Any]]]] = None # NEW
) -> Optional[str]:
    """Main orchestrator function to flip a PPTX file layout."""
    temp_dir = None
    files_changed_count = 0

    try:
        logging.info(f"--- Starting PPTX Flip Process for: {os.path.basename(pptx_path)} ---")
        logging.info(f"Options: {options}")

        # Create unique temporary directory
        temp_dir = os.path.join(tempfile.gettempdir(), f"ppt_flip_{uuid.uuid4().hex}")
        os.makedirs(temp_dir, exist_ok=True)
        logging.info(f"Created temporary directory: {temp_dir}")
        logging.info("10% - Extracting PPTX contents...")

        # Extract PPTX contents
        try:
            with zipfile.ZipFile(pptx_path, "r") as z_in:
                z_in.extractall(temp_dir)
            logging.info("PPTX extracted successfully.")
        except zipfile.BadZipFile:
            logging.error("BadZipFile error during extraction. Input file might be corrupted or not a PPTX.")
            return None
        except Exception as e:
            logging.exception("Error during PPTX extraction:")
            return None

        logging.info("20% - Reading presentation properties...")

        # Get slide dimensions
        presentation_xml_path = os.path.join(temp_dir, "ppt", "presentation.xml")
        slide_width_emu, slide_height_emu = DEFAULT_SLIDE_WIDTH_EMU, DEFAULT_SLIDE_HEIGHT_EMU
        if os.path.exists(presentation_xml_path):
            try:
                pres_tree = etree.parse(presentation_xml_path)
                pres_root = pres_tree.getroot()
                slide_width_emu, slide_height_emu = get_slide_size_from_presentation(pres_root)
            except Exception as e:
                logging.warning(f"Could not parse presentation.xml or get slide size: {e}. Using defaults.")
        else:
            logging.warning("presentation.xml not found. Using default slide size.")

        logging.info(f"30% - Using slide width {slide_width_emu} EMU. Locating XML files (incl. charts)...")

        # Find ALL XML files to process based on options (incl. charts)
        process_masters_layouts = options.get('process_masters_layouts', True)
        # Pass the absolute path to the temporary directory
        xml_files_to_process = find_all_xml_files_to_process(temp_dir, process_masters_layouts)
        total_files = len(xml_files_to_process)

        # --- Prepare replacement data mapping --- #
        slide_data_map: Dict[str, List[Dict[str, Any]]] = {}
        if replacement_data is not None and options.get('replace_text', False):
            ppt_slides_dir = os.path.join(temp_dir, "ppt", "slides")
            slide_files = []
            if os.path.isdir(ppt_slides_dir):
                for fname in os.listdir(ppt_slides_dir):
                    if fname.lower().startswith("slide") and fname.lower().endswith(".xml") and ".xml.rels" not in fname.lower():
                        match = re.search(r'slide(\d+)\.xml', fname, re.IGNORECASE)
                        if match:
                            slide_num = int(match.group(1))
                            slide_files.append((slide_num, os.path.join(ppt_slides_dir, fname)))
            
            # Sort slides numerically by extracted number
            slide_files.sort()
            sorted_slide_paths = [path for num, path in slide_files]

            if len(sorted_slide_paths) != len(replacement_data):
                logging.warning(f"Mismatch between number of slides found ({len(sorted_slide_paths)}) and replacement data provided ({len(replacement_data)}). Text replacement might be incomplete.")
                # Map based on the shorter length to avoid index errors
                min_len = min(len(sorted_slide_paths), len(replacement_data))
                for i in range(min_len):
                    # Handle None entries in replacement_data
                    if replacement_data[i] is not None:
                        slide_data_map[sorted_slide_paths[i]] = replacement_data[i]
                    else:
                        slide_data_map[sorted_slide_paths[i]] = None # Explicitly mark for skipping
                        logging.info(f"Marked slide {os.path.basename(sorted_slide_paths[i])} for skipping text replacement due to None data.")
            else:
                for i, slide_path in enumerate(sorted_slide_paths):
                    # Handle None entries in replacement_data
                    if replacement_data[i] is not None:
                        slide_data_map[slide_path] = replacement_data[i]
                    else:
                        slide_data_map[slide_path] = None # Explicitly mark for skipping
                        logging.info(f"Marked slide {os.path.basename(slide_path)} for skipping text replacement due to None data.")
                        logging.publish(f"Marked slide #{i + 1} for skipping text replacement due to faulty translation data.")

            logging.info(f"Prepared text replacement data map. {len([v for v in slide_data_map.values() if v is not None])} slides have data, {len([v for v in slide_data_map.values() if v is None])} marked for skipping.")

        if total_files == 0:
            logging.warning("No XML files found to process based on selected options.")
            return None
        else:
            logging.info(f"40% - Found {total_files} XML files (incl. charts). Starting processing...")

            # Process XML files in parallel
            processed_count = 0
            files_changed_count = 0
            # Limit workers to avoid overwhelming system resources
            max_workers = min(os.cpu_count() or 1, 8)
            logging.info(f"Using up to {max_workers} worker threads for XML processing.")

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Create a dictionary mapping futures to their corresponding file paths
                # Pass the specific slide replacement data if available
                future_to_file = {
                    executor.submit(
                        process_slide_xml_for_flipping, xml_file, slide_width_emu, options, slide_data_map.get(xml_file) # Pass specific data or None
                    ): xml_file
                    for xml_file in xml_files_to_process
                }

                # Process completed futures
                for future in as_completed(future_to_file):
                    path = future_to_file[future]
                    filename = os.path.basename(path)
                    processed_count += 1

                    try:
                        # Get the result (True if changed, False otherwise)
                        changed_flag = future.result()
                        if changed_flag: files_changed_count += 1
                        logging.debug(f"Future completed for {filename}. Changed: {changed_flag}")
                    except Exception as e:
                        # Log errors from individual file processing tasks
                        logging.error(f"Error processing future for {filename}: {e}")

                    # Update progress periodically
                    progress_percent = 40 + (50 * (processed_count / total_files if total_files > 0 else 0))
                    # Update roughly every 5% or on the last file
                    update_interval = max(1, total_files // 20)
                    if processed_count % update_interval == 0 or processed_count == total_files:
                        logging.info(f"{int(progress_percent)}% - Processed {processed_count}/{total_files} XML files ({files_changed_count} modified).")
                        logging.publish(f"{int(processed_count/total_files*100)}% processed.")

        logging.info("90% - Re-zipping the PPTX file...")
        logging.publish("Rebuilding the PPTX file...")

        # Create the output PPTX file path
        final_dir = output_dir if output_dir is not None else os.getcwd()

        if output_filename is None:
            base_name = os.path.basename(pptx_path)
            name_part, ext_part = os.path.splitext(base_name)
            # Sanitize filename part for safety
            safe_name_part = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in name_part)
            final_filename = f"flipped_{safe_name_part}{ext_part}"
        else:
            final_filename = output_filename

        # Ensure output directory exists
        os.makedirs(final_dir, exist_ok=True)

        flipped_pptx_path = os.path.join(final_dir, final_filename)

        files_skipped_zipping = 0

        # Re-zip the modified contents
        try:
            with zipfile.ZipFile(flipped_pptx_path, "w", zipfile.ZIP_DEFLATED) as z_out:
                # Walk through the temporary directory
                for root_dir, _, files_in_dir in os.walk(temp_dir):
                    for filename in files_in_dir:
                        full_path = None
                        try:
                            full_path = os.path.join(root_dir, filename)
                            # Calculate the relative path for the archive
                            relative_path = os.path.relpath(full_path, temp_dir)
                            # Write the file to the zip archive
                            z_out.write(full_path, arcname=relative_path)
                        except OSError as oe:
                            logging.error(f"OS Error writing file to zip: '{filename}' from '{full_path}'. Error: {oe}")
                            files_skipped_zipping += 1
                        except Exception as e_zip:
                            logging.exception(f"Unexpected error writing file to zip: '{filename}' from '{full_path}'.")
                            files_skipped_zipping += 1
        except Exception as e:
            logging.exception("Critical error during re-zipping process:")
            return None

        logging.info(f"Successfully created flipped PPTX: {flipped_pptx_path}")
        completion_message = f"✅ Processing complete. {files_changed_count} XML files modified."
        if files_skipped_zipping > 0:
            completion_message += f" ⚠️ Skipped adding {files_skipped_zipping} file(s) to the output due to errors."
        logging.info(completion_message)
        return flipped_pptx_path

    except Exception as e:
        # Catch any top-level errors in the orchestration
        logging.exception(f"An unexpected error occurred during PPTX processing orchestration. Error:\n{e}")
        return None
    finally:
        # Cleanup: Remove the log handler and temporary directory
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                logging.info(f"Cleaned up temporary directory: {temp_dir}")
            except Exception as e:
                logging.error(f"Error removing temporary directory {temp_dir}: {e}")
        logging.info("--- PPTX Flip Process Finished ---")
        logging.publish("PPTX flip and text replacement process complete.")

# --- Function used in server ---
def process_pptx_flip(
    input_pptx_path: str,
    options: Optional[Dict[str, bool]] = None,
    output_dir: Optional[str] = None,
    output_filename: Optional[str] = None,
    replacement_data: Optional[List[List[Dict[str, Any]]]] = None,
    publish_id: Optional[str] = '0'
) -> Optional[str]:
    """
    Processes a PPTX file for layout flipping based on provided options.
    This is the main entry point for integrating the flipping logic.

    Args:
        input_pptx_path: The path to the input PPTX file.
        options: A dictionary of processing options. If None, uses DEFAULT_OPTIONS.
        log_level: The logging level to use (e.g., logging.INFO, logging.DEBUG).
        output_dir: The directory where the output file will be saved. If None, uses the current working directory.
        output_filename: The name of the output file. If None, uses the 'flipped_' pattern.
        replacement_data: Data for text replacement, structured as List[slide_data], where slide_data is List[replacement_dict]. replacement_dict is {'id': int, 'Arabic': str}.

    Returns:
        The path to the flipped PPTX file if successful, otherwise None.
    """

    logging.set_publish_id(publish_id)
    # Use default options if none provided
    if options is None:
        options = DEFAULT_OPTIONS.copy()
        logging.info("Using default processing options.")
    else:
        # Validate provided options if necessary, or merge with defaults
        pass # Assuming the caller provides a valid dictionary for now

    # --- Input Validation ---
    if not input_pptx_path or not isinstance(input_pptx_path, str) or not os.path.exists(input_pptx_path):
        logging.error(f"Invalid input PPTX path: '{input_pptx_path}'")
        return None
    if not input_pptx_path.lower().endswith(".pptx"):
         logging.error(f"Input file does not have a .pptx extension: '{input_pptx_path}'")
         return None

    # --- Run Processing ---
    logging.info(f"Processing '{os.path.basename(input_pptx_path)}' with options: {options}")
    logging.publish("Processing flipping and text replacement...")

    try:
        # Call the layout flipping function, passing the new parameter
        output_path = flip_pptx_layout(input_pptx_path, options, output_dir, output_filename, replacement_data)
        if output_path:
            logging.info(f"Successfully generated flipped file: {output_path}")
            logging.publish(f"Successfully generated output file.")
            return output_path
        else:
            logging.error("PPTX flipping process failed to produce an output file.")
            logging.publish(f"Failed to generate output file.")
            return None
    except Exception as e:
        logging.exception(f"An error occurred during processing of '{input_pptx_path}': {e}")
        return None


# --- Gradio User Interface ---
def launch_ui():
    def run_flipping_interface(
        file_obj,
        # Mirror Position Options
        cb_mirror_shapes_pos, cb_mirror_pictures_pos, cb_mirror_connectors_pos,
        cb_mirror_groups_pos, cb_mirror_frames_pos,
        # Visual Flip Options
        cb_apply_flipH_shapes, cb_apply_flipH_pictures, cb_apply_flipH_groups,
        cb_apply_flipH_connectors, cb_experimental_preset_flip, cb_aggressive_shape_rebuild,
        # Text, Table, Chart, SmartArt Options
        cb_right_align_text, cb_set_rtl_paragraphs, cb_flip_table_columns,
        cb_flip_chart_internals, cb_apply_flipH_to_smartart_frame, # New options
        cb_aggressive_rebuild,
        cb_replace_text, # NEW
        cb_set_arabic_proofing_lang, # NEW
        # Scope & Logging
        cb_process_masters_layouts, log_level_dropdown
        ):
        """Wrapper function called by the Gradio button's click event."""

        # --- Set Log Level ---
        log_level_str = log_level_dropdown if log_level_dropdown else "INFO" # Default to INFO if None
        log_level = getattr(logging, log_level_str.upper(), logging.INFO)
        # Set level for the root logger AND ensure existing handlers respect it
        root_logger.setLevel(log_level)
        for handler in root_logger.handlers:
            # Check if handler has setLevel method before calling
            if hasattr(handler, 'setLevel'):
                handler.setLevel(log_level)
        logging.info(f"--- Log level set to: {log_level_str} ---")


        # --- Collate Options ---
        options = {
            "mirror_shapes_pos": cb_mirror_shapes_pos,
            "mirror_pictures_pos": cb_mirror_pictures_pos,
            "mirror_connectors_pos": cb_mirror_connectors_pos,
            "mirror_groups_pos": cb_mirror_groups_pos,
            "mirror_frames_pos": cb_mirror_frames_pos,
            "apply_flipH_to_shapes_no_text": cb_apply_flipH_shapes,
            "apply_flipH_to_pictures": cb_apply_flipH_pictures,
            "apply_flipH_to_groups": cb_apply_flipH_groups,
            "apply_flipH_to_connectors": cb_apply_flipH_connectors,
            "flip_table_columns": cb_flip_table_columns,
            "flip_chart_internals": cb_flip_chart_internals, # New option
            "apply_flipH_to_smartart_frame": cb_apply_flipH_to_smartart_frame, # New option
            "right_align_text": cb_right_align_text,
            "set_rtl_paragraphs": cb_set_rtl_paragraphs,
            "process_masters_layouts": cb_process_masters_layouts,
            "aggressive_paragraph_rebuild": cb_aggressive_rebuild,
            "experimental_preset_flip": cb_experimental_preset_flip,
            "aggressive_shape_rebuild": cb_aggressive_shape_rebuild,
            "replace_text": cb_replace_text, # NEW
            "set_arabic_proofing_lang": cb_set_arabic_proofing_lang # NEW
        }

        # --- Initial UI Update ---
        # Use dictionary access for Gradio components defined in the UI scope
        yield {
            output_file: None,
            status_box: "Starting...",
            debug_log_output: f"Starting process v2.5 (Arabic Proofing Language) with options:\n{options!r}" # Updated version
        }

        # --- Input Validation ---
        if file_obj is None:
            yield { output_file: None, status_box: "❌ Please upload a PPTX file.", debug_log_output: "Error: No file uploaded." }
            return

        # Get the file path safely
        pptx_path = getattr(file_obj, 'name', None)
        if not pptx_path or not os.path.exists(pptx_path):
            # Handle cases where file_obj might be a string path already (e.g., temp file)
            if isinstance(file_obj, str) and os.path.exists(file_obj):
                pptx_path = file_obj
            else:
                logging.error(f"Invalid file object or path received from Gradio: {file_obj!r}")
                yield { output_file: None, status_box: f"❌ Error: Uploaded file path is invalid or inaccessible.", debug_log_output: f"Error: Invalid path '{pptx_path}'. Object: {file_obj!r}"}
                return

        # --- Run Processing ---
        logging.info(f"Processing '{os.path.basename(pptx_path)}' via Gradio interface.")
        final_output_path, final_message, final_log_content = None, "An unknown issue occurred.", ""

        try:
            # Iterate through the generator returned by flip_pptx_layout
            for output_path, status_message, log_content in ui_flip_pptx_layout(pptx_path, options):
                final_output_path = output_path
                final_message = status_message
                final_log_content = log_content
                # Update UI with progress
                yield { output_file: final_output_path, status_box: status_message, debug_log_output: final_log_content }
        except Exception as e:
            final_message = f"❌ Interface Error: {str(e)}"
            logging.exception("Error in Gradio interface callback (run_flipping_interface)")
            # Ensure log content includes the error
            yield { output_file: None, status_box: final_message, debug_log_output: final_log_content + f"\n\n--- Interface Error ---\n{str(e)}" }

        # Final UI update (redundant if loop finished normally, but good practice)
        yield { output_file: final_output_path, status_box: final_message, debug_log_output: final_log_content }
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            <h1>PPTX Layout Flipper flipped ↔️ v2.5 (Arabic Proofing Language)</h1>
            <p>This tool mirrors the layout of PowerPoint presentations for Right-to-Left (RTL) adaptation. Select the desired options below.</p>
            <p style="color:green;"><strong>🆕 NEW in v2.5:</strong> Added Arabic proofing language setting (ar-SA) for spell check and grammar tools.</p>
            <p style="color:orange;"><strong>⚠️ WARNINGS:</strong>
            <ul>
                <li>Now attempts to swap Left/Right padding in text boxes when alignment options are enabled.</li>
                <li>Table flipping is experimental (swaps content, skips merged rows).</li>
                <li>Chart flipping is experimental (reverses axes, adjusts pie angle). Now attempts to process externally linked charts.</li>
                <li>SmartArt flipping (applying flipH to frame) is HIGHLY experimental and likely to distort visuals/text.</li>
                <li>Applying FlipH to shapes/connectors/groups may flip text or cause inaccuracies with rotated elements.</li>
                <li>AGGRESSIVE REBUILD options are HIGH RISK and may cause data loss or corruption (Defaults OFF).</li>
            </ul>
            Use with caution and always check the output thoroughly. Review logs (DEBUG level for most detail).
            </p>
            """
        )

        with gr.Row():
            ppt_input = gr.File(label="📂 Upload PPTX File", file_types=[".pptx"])
            output_file = gr.File(label="🔄 Download Flipped PPTX", interactive=False)

        gr.Markdown("## Processing Options")
        with gr.Row():
            # Column 1: Position Mirroring Options
            with gr.Column(scale=1):
                gr.Markdown("**Mirror Element Positions:**")
                cb_mirror_shapes_pos = gr.Checkbox(label="Shapes (incl. Text Boxes)", value=DEFAULT_OPTIONS["mirror_shapes_pos"])
                cb_mirror_pictures_pos = gr.Checkbox(label="Pictures", value=DEFAULT_OPTIONS["mirror_pictures_pos"])
                cb_mirror_connectors_pos = gr.Checkbox(label="Connectors", value=DEFAULT_OPTIONS["mirror_connectors_pos"])
                cb_mirror_groups_pos = gr.Checkbox(label="Groups", value=DEFAULT_OPTIONS["mirror_groups_pos"])
                cb_mirror_frames_pos = gr.Checkbox(label="Frames (Charts/Tables/SmartArt)", value=DEFAULT_OPTIONS["mirror_frames_pos"])

            # Column 2: Visual Flip (flipH) Options
            with gr.Column(scale=1):
                gr.Markdown("**Apply Visual Flip (Horizontal):**<br>(⚠️ Adjusts rotation if element rotated)")
                cb_apply_flipH_shapes = gr.Checkbox(label="Shapes (⚠️ Flips text if in shape)", value=DEFAULT_OPTIONS["apply_flipH_to_shapes_no_text"])
                cb_apply_flipH_pictures = gr.Checkbox(label="Pictures", value=DEFAULT_OPTIONS["apply_flipH_to_pictures"])
                cb_apply_flipH_connectors = gr.Checkbox(label="Connectors (⚠️ Experimental)", value=DEFAULT_OPTIONS["apply_flipH_to_connectors"])
                cb_apply_flipH_groups = gr.Checkbox(label="Groups (Flips Container)", value=DEFAULT_OPTIONS["apply_flipH_to_groups"])
                cb_experimental_preset_flip = gr.Checkbox(label="Try experimental preset flip (⚠️ Triangle/rtTriangle!)", value=DEFAULT_OPTIONS["experimental_preset_flip"])
                cb_aggressive_shape_rebuild = gr.Checkbox(label="Aggressively Rebuild Flipped Shapes (⚠️ Highest Risk!)", value=DEFAULT_OPTIONS["aggressive_shape_rebuild"])


            # Column 3: Text, Table, Chart, SmartArt Handling Options
            with gr.Column(scale=1):
                gr.Markdown("**Text & Complex Objects:**")
                cb_right_align_text = gr.Checkbox(label="Right-Align LTR/Justified Text", value=DEFAULT_OPTIONS["right_align_text"])
                cb_set_rtl_paragraphs = gr.Checkbox(label="Set RTL Paragraph Attribute", value=DEFAULT_OPTIONS["set_rtl_paragraphs"])
                cb_flip_table_columns = gr.Checkbox(label="Flip Table Columns (⚠️ Experimental - Skips Merged)", value=DEFAULT_OPTIONS["flip_table_columns"])
                cb_flip_chart_internals = gr.Checkbox(label="Flip Chart Internals (⚠️ Experimental - Axes/Pie)", value=DEFAULT_OPTIONS["flip_chart_internals"]) # New
                cb_apply_flipH_to_smartart_frame = gr.Checkbox(label="Flip SmartArt Frame (⚠️ Experimental - High Risk!)", value=DEFAULT_OPTIONS["apply_flipH_to_smartart_frame"]) # New
                cb_aggressive_rebuild = gr.Checkbox(label="Aggressively Rebuild Paragraphs (⚠️ High Risk - OFF by Default)", value=DEFAULT_OPTIONS["aggressive_paragraph_rebuild"])
                cb_replace_text = gr.Checkbox(label="Replace All Text (Hello world <id>)", value=False) # NEW
                cb_set_arabic_proofing_lang = gr.Checkbox(label="Set Arabic Proofing Language (ar-SA)", value=DEFAULT_OPTIONS["set_arabic_proofing_lang"]) # NEW

            # Column 4: Processing Scope & Logging
            with gr.Column(scale=1):
                gr.Markdown("**Scope & Logging:**")
                cb_process_masters_layouts = gr.Checkbox(label="Include Masters & Layouts", value=DEFAULT_OPTIONS["process_masters_layouts"])
                log_level_dropdown = gr.Dropdown(
                    label="Log Level",
                    choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                    value="INFO", # Default log level
                    info="Set detail level for logs (DEBUG is most verbose)."
                )


        process_button = gr.Button("Process PPTX with Selected Options", variant="primary")
        status_box = gr.Textbox(label="📊 Status", info="Processing progress and messages will appear here.", interactive=False, lines=2)
        debug_log_output = gr.Code(label="Log Output", interactive=False, visible=True, lines=15) # Renamed label

        # Define inputs list including the new checkboxes
        ui_inputs = [
            ppt_input,
            # Mirror Position
            cb_mirror_shapes_pos, cb_mirror_pictures_pos, cb_mirror_connectors_pos,
            cb_mirror_groups_pos, cb_mirror_frames_pos,
            # Visual Flip
            cb_apply_flipH_shapes, cb_apply_flipH_pictures, cb_apply_flipH_groups,
            cb_apply_flipH_connectors, cb_experimental_preset_flip, cb_aggressive_shape_rebuild,
            # Text & Complex
            cb_right_align_text, cb_set_rtl_paragraphs, cb_flip_table_columns,
            cb_flip_chart_internals, cb_apply_flipH_to_smartart_frame, # New inputs
            cb_aggressive_rebuild,
            cb_replace_text, # NEW
            cb_set_arabic_proofing_lang, # NEW
            # Scope & Logging
            cb_process_masters_layouts, log_level_dropdown
        ]
        # Define outputs
        ui_outputs = [output_file, status_box, debug_log_output]

        # Connect button click to the processing function
        process_button.click(fn=run_flipping_interface, inputs=ui_inputs, outputs=ui_outputs, queue=True)

        gr.Markdown(
            """
            <hr>
            <p style='text-align: center; margin-top: 15px; color: grey; font-size: small;'>
                Remember to review the warnings above. Chart and SmartArt flipping are particularly experimental.
            </p>
            """
        )
    # Set share=True to create a public link (optional)
    # Set debug=True for Gradio's internal debugging (can be noisy)
    demo.launch(debug=False, share=True)


if __name__ == "__main__":
    # --- Launch the Gradio Application ---
    import gradio as gr
    import logging
    import io # Required for capturing logs in memory
    # --- Configuration & Setup ---
    # Configure logging: Set level and format. Ensure handler exists.
    if not logging.getLogger().hasHandlers():
        # Basic config if no handlers are present
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d - %(funcName)s] %(message)s')
    else:
        # If handlers exist (e.g., Gradio adds one), set the level for the root logger
        logging.getLogger().setLevel(logging.INFO) # Default INFO

    log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d - %(funcName)s] %(message)s')
    root_logger = logging.getLogger() # Get the root logger

    print("Launching Gradio interface...")
    launch_ui()

else:
    # --- Initialize Server Logger ---
    try:
        from pipeline_utilities import Logger
        logging = Logger()
    except Exception as e:
        print(f"Error importing pipeline_utilities: {e}")
