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
import requests
import json
import firebase_admin
from datetime import timedelta
import traceback
import zipfile
import math
from lxml import etree
from pptx import Presentation
from pptx.shapes.group import GroupShape
from pptx.shapes.graphfrm import GraphicFrame
from pptx.enum.shapes import MSO_SHAPE_TYPE
from firebase_admin import credentials, storage, db

db_url = 'https://snb-ai-translation-agent-default-rtdb.firebaseio.com'
secret = 'nAWmdbcHRL9UGDOP0S1Rp0pZ2c7TEIapUrsEBzHJ'
download_folder = "downloads"
# Path to your service account key
cred = credentials.Certificate("service_account_key.json")


def init_firebase():
    try:
        # Initialize app with storage bucket
        firebase_admin.initialize_app(cred, {
            'storageBucket': 'snb-ai-translation-agent.firebasestorage.app',
            'databaseURL': db_url
        })
    except Exception as e:
        pass


class Logger:
    def __init__(self):
        self.publish_ref = "logs/0"

    def set_publish_id(self, id: str):
        self.publish_ref = f"logs/{id}"

    def info(self, message):
        print(f"INFO: {message}")

    def debug(self, message):
        print(f"DEBUG: {message}")

    def warning(self, message):
        self.print_and_write(f"WARNING: {message}")

    def error(self, message):
        self.print_and_write(f"ERROR: {message}")

    def exception(self, message):
        self.print_and_write(f"EXCEPTION: {message}")

    def print_and_write(self, message):
        print(message)
        if not os.path.exists("logs"):
            os.makedirs("logs")
        with open(f"{self.publish_ref}.txt", "a") as f:
            f.write(message + "\n")

    def publish(self, message):
        init_firebase()
        try:
            ref = db.reference(f"/{self.publish_ref}")
            # Push a new object (auto-generates a unique key)
            ref.push({
                'message': str(message)
            })
        except Exception as e:
            self.exception(e)
        print(message)


def upload_output(output_path: str):
    init_firebase()
    bucket = storage.bucket()
    blob = bucket.blob(output_path)

    # Upload from local file
    blob.upload_from_filename(output_path)

    # Optionally make it publicly accessible
    return blob.generate_signed_url(expiration=timedelta(hours=2))
    blob.make_public()
    return blob.public_url


def download_file(url: str, filename: str):
    response = requests.get(url)
    if not os.path.exists(download_folder):
        os.makedirs(download_folder)
    output_path = f"{download_folder}/{filename}"
    # Save the content to a file
    with open(output_path, 'wb') as file:
        file.write(response.content)
    return output_path


def clear_id(id: str):
    requests.put(
        f"{db_url}/translation_requests/{id}.json?auth={secret}",
        data=json.dumps({})
    )


def extract_text_shapes(pptx_path: str):
    try:
        if not os.path.isfile(pptx_path):
            return f"No english file found at: {pptx_path}"

        eng_pptx = Presentation(pptx_path)
        pptx_rects = [eng_pptx.slide_width / eng_pptx.slide_height]
        pptx_zip = zipfile.ZipFile(pptx_path, "r")

        AVERAGE_CHAR_WIDTH_FACTOR = 0.47
        LINE_SPACING_FACTOR = 1.0

        def emu_to_pt(emu):
            return emu / 12700

        def pt_to_emu(pt):
            return int(pt * 12700)

        def process_table(table_shape: GraphicFrame, group_left=0, group_top=0):
            """Processes a table and extracts bounding boxes for each cell."""
            table = table_shape.table
            num_rows = len(table.rows)
            num_cols = len(table.columns)

            shape_left = table_shape.left + group_left
            shape_top = table_shape.top + group_top
            shape_width = table_shape.width
            shape_height = table_shape.height

            row_heights = [row.height for row in table.rows]
            col_widths = [col.width for col in table.columns]

            total_width = sum(col_widths)
            total_height = sum(row_heights)

            scale_factor = (
                shape_height / total_height if total_height > 0 else 1
            )

            y_cursor = shape_top
            for row_idx, row_height in enumerate(row_heights):
                if row_height == 0:
                    for col_idx, _ in enumerate(col_widths):
                        cell = table.cell(row_idx, col_idx)
                        num_lines = 0
                        font_size = 0
                        for para in cell.text_frame.paragraphs:
                            num_lines += 1
                            for run in para.runs:
                                if run.text.strip() and run.font.size:
                                    line_spacing_factor = para.line_spacing if para.line_spacing else LINE_SPACING_FACTOR
                                    font_size = max(font_size, run.font.size.pt * line_spacing_factor)
                    row_height = max(row_height, num_lines * pt_to_emu(font_size))
                row_height = row_height * scale_factor
                x_cursor = shape_left
                for col_idx, col_width in enumerate(col_widths):
                    cell = table.cell(row_idx, col_idx)

                    if not cell.text.strip():
                        x_cursor += col_width
                        continue  # Skip empty cells

                    x1 = x_cursor / eng_pptx.slide_width
                    x2 = (x_cursor + col_width) / eng_pptx.slide_width
                    y1 = y_cursor / eng_pptx.slide_height
                    y2 = (y_cursor + row_height) / eng_pptx.slide_height

                    yield [x1, x2, y1, y2, cell.text.strip()]
                    x_cursor += col_width

                y_cursor += row_height

        def process_shape(shape, x_scale=1, y_scale=1, x_transform=0, y_transform=0, initial_left=-1, initial_top=-1):
            """ Processes a shape or a grouped shape recursively. """
            if shape.shape_type == MSO_SHAPE_TYPE.GROUP:
                shape: GroupShape = shape

                shape_left = shape.left + x_transform
                shape_top = shape.top + y_transform
                if initial_left != -1 and initial_top != -1:
                    shape_left = shape_left + (shape.left - initial_left) * (x_scale - 1)
                    shape_top = shape_top  + (shape.top - initial_top) * (y_scale - 1)

                minX1 = math.inf
                maxX2 = 0
                minY1 = math.inf
                maxY2 = 0
                for sub_shape in shape.shapes:
                    if sub_shape.left < minX1: minX1 = sub_shape.left
                    if sub_shape.left + sub_shape.width > maxX2: maxX2 = sub_shape.left + sub_shape.width
                    if sub_shape.top < minY1: minY1 = sub_shape.top
                    if sub_shape.top + sub_shape.height > maxY2: maxY2 = sub_shape.top + sub_shape.height
                initial_group_width = maxX2 - minX1
                initial_group_height = maxY2 - minY1
                if initial_group_width == 0: group_scale_x = 1
                else: group_scale_x = shape.width / initial_group_width
                if initial_group_height == 0: group_scale_y = 1
                else: group_scale_y = shape.height / initial_group_height
                group_transform_x = shape_left - minX1
                group_transform_y = shape_top - minY1
                for sub_shape in shape.shapes:
                    yield from process_shape(sub_shape, group_scale_x, group_scale_y, group_transform_x, group_transform_y, minX1, minY1)
                    # yield from process_shape(sub_shape)
                return

            elif shape.shape_type == MSO_SHAPE_TYPE.TABLE:
                # yield from process_table(shape, group_left, group_top)
                yield from process_table(shape)
                return

            elif not shape.has_text_frame:
                return

            text_frame = shape.text_frame
            if not text_frame.paragraphs:
                return

            # Adjust shape dimensions based on group offsets
            shape_width_pt = emu_to_pt(shape.width)
            shape_height_pt = emu_to_pt(shape.height)

            para_info = []
            total_estimated_height = 0
            first = True

            for para in text_frame.paragraphs:
                text = para.text.strip()
                is_empty = not text

                font_size = 0
                for run in para.runs:
                    if run.text.strip() and run.font.size:
                        font_size = max(font_size, run.font.size.pt)
                if font_size == 0:
                    font_size = 12

                line_spacing_factor = para.line_spacing if para.line_spacing else LINE_SPACING_FACTOR

                if is_empty:
                    num_lines = 1
                else:
                    char_width_pt = font_size * AVERAGE_CHAR_WIDTH_FACTOR
                    chars_per_line = max(1, math.floor(shape_width_pt / char_width_pt))
                    num_lines = max(1, math.ceil(len(text) / chars_per_line))

                para_height_pt = (
                    1.3 * num_lines * font_size * line_spacing_factor 
                    - 0.25 * (num_lines - 1) * font_size * line_spacing_factor 
                    + emu_to_pt(para.space_before if para.space_before else 0) 
                    + emu_to_pt(para.space_after if para.space_after else 0)
                )
                para_info.append({'para': para, 'height_pt': para_height_pt, 'is_empty': is_empty})
                total_estimated_height += para_height_pt
                first = False

            scale_factor = (
                (shape_height_pt - emu_to_pt(text_frame.margin_bottom + text_frame.margin_top)) / total_estimated_height
                if total_estimated_height > 0 else 1
            )

            shape_left = shape.left + x_transform
            shape_top = shape.top + text_frame.margin_top + y_transform
            if initial_left != -1 and initial_top != -1:
                shape_left = shape_left + (shape.left - initial_left) * (x_scale - 1)
                shape_top = shape_top  + (shape.top - initial_top) * (y_scale - 1)
            y_cumulative = shape_top

            for info in para_info:
                scaled_height_pt = info['height_pt'] * scale_factor

                if info['is_empty']:
                    y_cumulative += pt_to_emu(scaled_height_pt)
                    continue

                scaled_height_emu = pt_to_emu(scaled_height_pt)

                x1 = shape_left / eng_pptx.slide_width
                x2 = (shape_left + shape.width * x_scale) / eng_pptx.slide_width
                y1 = y_cumulative / eng_pptx.slide_height
                y2 = (y_cumulative + scaled_height_emu * y_scale) / eng_pptx.slide_height

                yield [x1, x2, y1, y2, info['para'].text.strip()]
                y_cumulative += scaled_height_emu

        def is_slide_hidden(slide_number):
            """Checks if a specific slide is hidden based on the 'show' attribute in its XML."""
            slide_path = f"ppt/slides/slide{slide_number}.xml"

            with pptx_zip.open(slide_path) as xml_file:
                tree = etree.parse(xml_file)

            # Get the root element <p:sld>
            root = tree.getroot()

            # Check if 'show' attribute is set to "0"
            return root.get("show") == "0"

        for i, slide in enumerate(eng_pptx.slides):
            if is_slide_hidden(i + 1): continue
            slide_paragraph_rects = []
            for shape in slide.shapes:
                slide_paragraph_rects.extend(process_shape(shape))
            pptx_rects.append(slide_paragraph_rects)

        return pptx_rects

    except Exception as e:
        traceback.print_exc()
        return str(e)