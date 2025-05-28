from paddlex import create_model
from pdf2image import convert_from_path
import os
import traceback
import zipfile
import time
from IPython.display import clear_output
import requests
import json
import ast
import tempfile
import uuid
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Tuple, Optional, Generator, List, Dict, Any, Callable, Set
import copy # Needed for deep copying elements
import re # For parsing adjustment values
from pathlib import Path # For easier path manipulation
import firebase_admin
from firebase_admin import credentials, storage
from datetime import timedelta

from openai import OpenAI
import concurrent.futures
import pandas as pd
import tiktoken
from collections import deque
import threading

from pipeline_utilities import *
from slide_flipping import process_pptx_flip


logger = Logger()
parallelWorkers = 5
model = "gpt-4o"

client = OpenAI(
    api_key='sk-proj-zo8UYHSjQsRDj27x0UgPT3BlbkFJRhKTphRtEu8ITjFjfRBS'
)

paddle_text_types = {
    "doc_title": "Page Title",
    "paragraph_title": "Paragraph Header",
    "text": "Body",
    "page_number": "Formals",
    "abstract": "Body",
    "table_of_contents": "Page Title",
    "references": "Formals",
    "footnotes": "Formals",
    "header": "Formals",
    "footer": "Formals",
    "algorithm": "Unknown",
    "formula": "Unknown",
    "formula_number": "Formals",
    # "image": "Shape",
    "figure_caption": "Paragraph Header",
    "table": "Table",
    "table_caption": "Paragraph Header",
    "seal": "Formals",
    "figure_title": "Paragraph Header",
    # "figure": "Shape",
    "header_image": "Formals",
    "footer_image": "Formals",
    "sidebar_text": "Body"
}

class PipelinePublic:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(PipelinePublic, cls).__new__(cls)
            cls._instance.model = model
            cls._instance.parallelWorkers = parallelWorkers
            cls._instance.logger = Logger()
            cls._instance.token_limiter = PipelinePublic.TokenRateLimiter(tpm_limit=30000, model_name="gpt-4o")
            cls._instance.paddle_model = create_model(model_name="PP-DocLayout-L")
        return cls._instance

    class TokenRateLimiter:
        EXPECTED_COMPLETION_TOKENS_ESTIMATE = 600
        def __init__(self, tpm_limit=30000, model_name="gpt-4o"):
            self.tpm_limit = tpm_limit
            self.token_log = deque()  # Stores (timestamp, tokens_used)
            self.lock = threading.Lock()
            try:
                # gpt-4o uses the same tokenizer as gpt-4 / gpt-4-turbo
                self.tokenizer = tiktoken.encoding_for_model(model_name)
            except KeyError:
                # Fallback for models not yet in tiktoken or if specific name changes
                print(f"Warning: Tokenizer for model '{model_name}' not found directly. Using 'cl100k_base' encoding. Token count accuracy may vary.")
                self.tokenizer = tiktoken.get_encoding("cl100k_base")

        def _cleanup_log_unsafe(self):
            """Removes entries older than 60 seconds. Not thread-safe by itself."""
            current_time = time.time()
            while self.token_log and self.token_log[0][0] < current_time - 60:
                self.token_log.popleft()

        def _get_current_tokens_in_window_unsafe(self):
            """Calculates total tokens used in current 60s window. Not thread-safe."""
            self._cleanup_log_unsafe()
            return sum(tokens for _, tokens in self.token_log)

        def wait_if_needed(self, prompt_tokens_for_this_request):
            """
            Waits if adding estimated_tokens_for_this_request (prompt + default completion) 
            would exceed TPM.
            """
            estimated_total_tokens = prompt_tokens_for_this_request + PipelinePublic.TokenRateLimiter.EXPECTED_COMPLETION_TOKENS_ESTIMATE
            
            with self.lock:
                while True:
                    self._cleanup_log_unsafe()
                    current_tokens_in_window = self._get_current_tokens_in_window_unsafe()
                    available_budget = self.tpm_limit - current_tokens_in_window

                    if estimated_total_tokens <= available_budget:
                        # Enough budget for the estimated tokens of the current request
                        break 
                    else:
                        # Not enough budget. Wait for some tokens to expire from the window.
                        if self.token_log:
                            oldest_entry_time, _ = self.token_log[0]
                            # Time to wait until the oldest entry expires
                            wait_duration = (oldest_entry_time + 60.1) - time.time() # 60.1 for small buffer
                            
                            if wait_duration <= 0:
                                # Oldest entry should have expired. Re-evaluate.
                                self._cleanup_log_unsafe() # Ensure log is clean before re-checking
                                continue 
                            
                            print(f"TPM Check: Window tokens {current_tokens_in_window}, est. next call {estimated_total_tokens} (limit {self.tpm_limit}). Waiting {wait_duration:.2f}s.")
                            # Release lock while sleeping
                            self.lock.release()
                            try:
                                time.sleep(wait_duration)
                            finally:
                                # Reacquire lock before continuing loop
                                self.lock.acquire()
                            # After waking up, loop again to re-evaluate conditions
                        else:
                            # No tokens in log, but estimated request itself is too large
                            if estimated_total_tokens > self.tpm_limit:
                                print(f"Warning: Estimated tokens for a single request ({estimated_total_tokens}, incl. default completion) exceeds TPM limit ({self.tpm_limit}). This request may fail if not handled by API retries.")
                            # Allow to proceed if this is the case.
                            break 

        def add_request_tokens(self, actual_total_tokens_used):
            """Adds the actual token count of a completed request to the log."""
            with self.lock:
                if actual_total_tokens_used > 0:
                    self.token_log.append((time.time(), actual_total_tokens_used))

        def estimate_prompt_tokens(self, prompt_text):
            return len(self.tokenizer.encode(prompt_text))


    def get_prompt(input_json: str, output_json: str = ""):
        instruction = """Translate the following sentences from english to arabic. Return the translations by id in JSON format. Do not translate any acronyms or words that are better left in english.
        Example:
        English: [{"id": 1, "Text Type": "Page Title", "English": "The Sun"}, {"id": 2, "Text Type": "Page Title", "English": "The sun was setting over the horizon, painting the sky in shades of orange and pink."}]
        Arabic: [{"id": 1, "Arabic": "الشمس"}, {"id": 2, "Arabic": "كانت الشمس تغرب عند الأفق، لتملأ السماء بدرجات اللون البرتقالي والوردي."}]
        """
        return f"{instruction}\nEnglish: {input_json}\nArabic: {output_json}"


    def chat_gpt(self, prompt):
        # Estimate prompt tokens for the current request
        prompt_tokens = self.token_limiter.estimate_prompt_tokens(prompt)
        
        # Wait if necessary based on current token usage and estimated tokens for this call
        self.token_limiter.wait_if_needed(prompt_tokens)

        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        response_message = response.choices[0].message.content
        
        # Record actual tokens used (prompt + completion)
        if response.usage:
            actual_tokens = response.usage.total_tokens
            self.token_limiter.add_request_tokens(actual_tokens)

        time.sleep(2)

        return response_message.strip().replace('\\x0c', '\\n').replace('\\x0b', '\\n')


    def fetch_and_parse_output(self, prompt):
        response = self.chat_gpt(prompt)
        response = response.replace("```json", "").replace("```", "")
        response = json.loads(response)
        return response

    def infer(self, input_json: str, index: int = 0):
        generated_text = self.fetch_and_parse_output(PipelinePublic.get_prompt(input_json))

        if isinstance(generated_text, str):
            output_str_list = generated_text
            try:
                out_list_repr = ast.literal_eval(output_str_list)
            except (ValueError, SyntaxError, TypeError) as e:
                out_list_repr = PipelinePublic.reevaluate(output_str_list)

            if out_list_repr is not None:
                if isinstance(out_list_repr, list):
                    for item in out_list_repr:
                        if isinstance(item, dict) and "id" in item and "Arabic" in item:
                            pass
                        else:
                            out_list_repr = None
        else:
            out_list_repr = generated_text

        return index, out_list_repr

    def reevaluate(a: str):
        b = re.sub(r"'", "¨", a)
        b = re.sub(r'\[{"id"', "[{'id'", b)
        b = re.sub(r'"}, {"id"', "'}, {'id'", b)
        b = re.sub(r'"},{"id"', "'},{'id'", b)
        b = re.sub(r'"Arabic": "', "'Arabic': '", b)
        b = re.sub(r'"Arabic":"', "'Arabic':'", b)
        b = re.sub(r'"}]', "'}]", b)
        try:
            c = ast.literal_eval(b)
            for item in c:
                item["Arabic"] = re.sub(r"¨", "'", item["Arabic"])
            return c
        except (ValueError, SyntaxError, TypeError) as e:
            return None

    def convert_pdf_to_images_sm(self, pdf_path, output_folder='pdf2image_output'):
        """Converts a PDF file to image files in the SageMaker environment."""
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        if not os.path.exists(pdf_path):
            self.logger.error("PDF does not exist.")
            return []

        image_paths = []
        self.logger.publish("Converting pdf to images...")
        try:
            # In SageMaker's Linux environment, poppler should be found if installed via apt-get
            images = convert_from_path(pdf_path)
            for i, img in enumerate(images):
                img_path = os.path.join(output_folder, f'page_{i+1}.png')
                img.save(img_path, 'PNG')
                image_paths.append(img_path)
            self.logger.publish(f"Converted {len(image_paths)} pages to images.")
            return image_paths
        except Exception as e:
            self.logger.error(f"Error converting PDF: {e}")
            return []


    def analyze_document_layout(self, pdf_path, number_of_slides, output_save_dir='paddle_output'):
        """
        Converts PDF to images and runs layout analysis using the PPStructure engine.
        Focuses on extracting layout information (type, bbox).
        """
        image_files = self.convert_pdf_to_images_sm(pdf_path)
        if not image_files:
            self.logger.publish("PDF processing failed.")
            return None

        if not os.path.exists(output_save_dir):
            os.makedirs(output_save_dir)

        all_page_layout_results = []

        for i, img_path in enumerate(image_files):
            self.logger.publish(f"Processing page #{i+1} of {number_of_slides}")
            output = self.paddle_model.predict(img_path, batch_size=1, layout_nms=True)
            page_layout_info = []
            size = None
            for res in output:
                if size is None:
                    size = res['input_img'].shape
                page_layout_info.append(res['boxes'])
                # # Print result and save images
                # res.print()
                # res.save_to_img(save_path=output_save_dir)
                # res.save_to_json(save_path=f"{output_save_dir}/res_{img_name}.json")

            all_page_layout_results.append({
                'image_path': img_path,
                'image_size': size[0:2],
                'layout_results': page_layout_info
            })
            os.remove(img_path)

        self.logger.publish("Document layout analysis complete.")
        return all_page_layout_results


    def apply_layout_types(self, extracted_shapes, layout):
        number_of_slides = len(extracted_shapes)
        source = []
        for i in range(number_of_slides):
            self.logger.publish(f"Processing slide #{i + 1} of {number_of_slides}")
            slide_source = []
            slide = extracted_shapes[i]
            for nshape in slide:  # [x1,x2,y1,y2]
                size = layout[i]['image_size']  # (height,width) in pixels
                shape = [size[1]*nshape[0], size[1]*nshape[1], size[0]*nshape[2], size[0]*nshape[3]]
                slide_layout = layout[i]['layout_results'][0]
                center = ((shape[0]+shape[1])/2, (shape[2]+shape[3])/2)  # (x,y)
                text_type = "Unknown"
                for j in range(len(slide_layout)):
                    coordinates = slide_layout[j]['coordinate']  # [x1,y1,x2,y2]
                    if center[0] >= coordinates[0] \
                        and center[0] <= coordinates[2] \
                        and center[1] >= coordinates[1] \
                            and center[1] <= coordinates[3]:
                        # Center is inside layout
                        text_type = slide_layout[j]['label']
                        text_type = paddle_text_types.get(text_type)
                        if text_type is None:
                            text_type = "Unknown"
                        break
                if text_type == "Unknown":
                    # Try seeing if there is any overlap
                    def rectangles_overlap(xa1, xa2, ya1, ya2, xb1, xb2, yb1, yb2):
                        # Make sure the coordinates are ordered correctly
                        xa1, xa2 = min(xa1, xa2), max(xa1, xa2)
                        ya1, ya2 = min(ya1, ya2), max(ya1, ya2)
                        xb1, xb2 = min(xb1, xb2), max(xb1, xb2)
                        yb1, yb2 = min(yb1, yb2), max(yb1, yb2)

                        # Check for non-overlap
                        if xa2 <= xb1 or xb2 <= xa1:
                            return False  # No overlap in x-axis
                        if ya2 <= yb1 or yb2 <= ya1:
                            return False  # No overlap in y-axis

                        return True  # Overlap exists

                    for j in range(len(slide_layout)):
                        coordinates = slide_layout[j]['coordinate']  # [x1,y1,x2,y2]
                        if rectangles_overlap(shape[0], shape[1], shape[2], shape[3],
                                            coordinates[0], coordinates[2], coordinates[1], coordinates[3]):
                            # Center is inside layout
                            text_type = slide_layout[j]['label']
                            text_type = paddle_text_types.get(text_type)
                            if text_type is None:
                                text_type = "Body"
                            break

                slide_source.append({
                    "type": text_type,
                    "text": nshape[4].replace('\'', '’')
                })
            source.append(slide_source)
        return source

    def run_translation(self, request):
        try:
            request_id = request.get("id")
            if request_id is None:
                request_id = "0"
            self.logger.set_publish_id(request_id)

            filename = request.get("filename")
            if filename is None:
                filename = request_id
            self.logger.print_and_write(f"Processing request {request_id}, filename: {filename}, with model id: pro")

            logger.publish("Fetching files...")
            pdfFilename = f"{request_id}.pdf"
            pdfPath = download_file(request['pdfUrl'], pdfFilename)
            pptFilename = f"{request_id}.pptx"
            pptPath = download_file(request['pptUrl'], pptFilename)

            logger.publish("Analyzing text shapes in pptx file...")
            extracted_shapes = extract_text_shapes(pptPath)
            scale_factor = extracted_shapes[0]
            extracted_shapes.pop(0)
            number_of_slides = len(extracted_shapes)

            logger.publish("Processing layout types...")
            layout = self.analyze_document_layout(pdfPath, number_of_slides)
            os.remove(pdfPath)

            logger.publish("Applying layout types to text shapes...")
            source = self.apply_layout_types(extracted_shapes, layout)
            logger.publish("Analyzing shapes complete.")

            logger.publish("Initializing translation...")
            # # Write source to txt file
            # with open("source.txt", 'w') as file:
            #     file.write(str(source))
            inputJson = []
            outputJson = []
            for slide in source:
                slideJson = []
                for i, text in enumerate(slide):
                    slideJson.append({
                        'id': i + 1,
                        'Text Type': text['type'],
                        'English': text['text']
                    })
                inputJson.append(str(slideJson).replace('\'', '"'))
                outputJson.append(None)

            with concurrent.futures.ThreadPoolExecutor(max_workers=parallelWorkers) as executor:
                futures = [executor.submit(self.infer, prompt, index) for index, prompt in enumerate(inputJson)]
                for future in concurrent.futures.as_completed(futures):
                    i, output_str_list = future.result()

                    logger.publish(f"Translated slide #{i + 1} of {number_of_slides}...")
                    if output_str_list is None:
                        logger.publish(f"Translation parsing error in slide #{i + 1}.")
                        outputJson[i] = None
                    else:
                        if len(inputJson[i]) != len(output_str_list):
                            logger.publish(f"Translation length error in slide #{i + 1}.")
                        outputJson[i] = output_str_list

            for slide in outputJson:
                with open(f"outputs/output_{request_id}.txt", 'a') as file:
                    file.write(str(slide))
                    file.write('\n\n')

            # outputJson = []
            # for i, slide in enumerate(inputJson):
            #     logger.publish(f"Translating slide #{i + 1} of {number_of_slides}...")
            #     output_str_list = infer(str(slide).replace('\'', '"'))
            #     with open(f"outputs/output_{first_id}.txt", 'a') as file:
            #         file.write(str(output_str_list))
            #         file.write('\n\n')
            #     if output_str_list is None:
            #         logger.publish(f"Translation parsing error in slide #{i + 1}.")
            #         outputJson.append(None)
            #     else:
            #         if len(slide) != len(output_str_list):
            #             logger.publish(f"Translation length error in slide #{i + 1}.")
            #         outputJson.append(output_str_list)

            # use outputJson to change text
            outputPath = process_pptx_flip(
                input_pptx_path=pptPath,
                output_dir="outputs",
                output_filename=f"{filename}_AR.pptx",
                replacement_data=outputJson,
                publish_id=request_id
            )

            if outputPath is None:
                raise Exception("Output path is None")

            logger.publish("Preparing output file for download...")
            uploadUrl = upload_output(outputPath)
            logger.publish("Output file ready for download.")
            logger.publish("DONE")
            logger.publish(uploadUrl)
            return True
        except Exception as e:
            self.logger.error(e)
            # Clear request from database
            clear_id(request_id)
            self.logger.publish(f"Error occurred. Please provide the following code to the developing team: {request_id}")
            return False
