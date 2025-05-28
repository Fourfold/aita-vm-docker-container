import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer, BitsAndBytesConfig # Added BitsAndBytesConfig
from peft import PeftModel # Added PeftModel
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
from pipeline_utilities import *
from slide_flipping import process_pptx_flip
from pipeline_public import PipelinePublic

model_name = "gemmax2_9b_finetuned"
base_model = "ModelSpace/GemmaX2-28-9B-Pretrain"

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

class PipelinePro:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(PipelinePro, cls).__new__(cls)
            cls._instance.model_name = model_name
            cls._instance.base_model = base_model
            cls._instance.logger = Logger()
            cls._instance.initialize_model()
            cls._instance.paddle_model = create_model(model_name="PP-DocLayout-L")
        return cls._instance


    def initialize_model(self):
        # --- Configuration ---
        lora_adapter_path = self.model_name
        # Optional but Recommended: Add Quantization for lower memory usage and potentially faster inference
        use_quantization = True  # Set to False to load in full precision (if you have enough VRAM)
        quantization_bit = 4  # Or 8

        # --- Check for GPU ---
        if not torch.cuda.is_available():
            raise SystemError("CUDA is not available. Please ensure you have a GPU and CUDA installed.")

        # --- Load Model and Tokenizer ---
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model)

        print("Loading model...")
        bnb_config = None
        model_kwargs = {"device_map": "auto"}

        if use_quantization:
            if quantization_bit == 4:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16  # Use bfloat16 if supported, otherwise float16
                )
                # Add torch_dtype for 4-bit loading
                model_kwargs["torch_dtype"] = torch.bfloat16  # Or torch.float16
            elif quantization_bit == 8:
                bnb_config = BitsAndBytesConfig(load_in_8bit=True)

            model_kwargs["quantization_config"] = bnb_config

        # Load the model with device_map and optional quantization
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model,
            **model_kwargs
        )
        print(f"Base model loaded successfully on device: {self.model.device}")  # Should show cuda if successful

        # --- Load LoRA Adapters ---
        print(f"Loading LoRA adapters from: {lora_adapter_path}...")
        self.model = PeftModel.from_pretrained(self.model, lora_adapter_path)
        print("LoRA adapters loaded successfully.")
        print(f"Final model (with adapters) ready on device: {self.model.device}")


    def get_prompt(input_json: str, output_json: str = ""):
        instruction = "Translate the following sentences from english to arabic. Return the translations by id in JSON format. MAKE SURE THAT THE NUMBER OF ITEMS IN THE ENGLISH AND ARABIC JSON LISTS IS EQUAL."
        return f"{instruction}\nEnglish: {input_json}\nArabic: {output_json}"


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


    def infer(self, input_json: str, use_text_stream: bool = False):
        EOS_TOKEN = self.tokenizer.eos_token
        inputs = self.tokenizer([self.get_prompt(input_json)], return_tensors="pt").to("cuda")
        if not use_text_stream:
            outputs = self.model.generate(**inputs, max_new_tokens=4096, use_cache=True)
            generated_text = '[{"id":' + self.tokenizer.batch_decode(outputs)[0].split('Arabic: [{"id":')[1]
        else:
            print("Starting generation...\n")
            text_streamer = TextStreamer(self.tokenizer, skip_prompt=True)  # skip_prompt=True avoids printing the input again
            outputs = self.model.generate(**inputs, streamer=text_streamer, max_new_tokens=4096)
            # Decode the newly generated tokens
            # Get the length of the input prompt tokens
            input_length = inputs["input_ids"].shape[1]
            # Slice the outputs to get only the tokens generated *after* the prompt
            # We use outputs[0] because generate returns a batch, and we take the first item
            generated_tokens = outputs[0, input_length:]
            # Use the tokenizer to decode these tokens into a string
            generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            print("\nGeneration complete.")

        output_str_list = generated_text[:-len(EOS_TOKEN)].replace('\\x0c', '\\n').replace('\\x0b', '\\n')

        try:
            out_list_repr = ast.literal_eval(output_str_list)
        except (ValueError, SyntaxError, TypeError) as e:
            out_list_repr = PipelinePro.reevaluate(output_str_list)

        if out_list_repr is not None:
            if isinstance(out_list_repr, list):
                for item in out_list_repr:
                    if isinstance(item, dict) and "id" in item and "Arabic" in item:
                        pass
                    else:
                        out_list_repr = None

        return out_list_repr


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

            self.logger.publish("Fetching files...")
            pdfFilename = f"{request_id}.pdf"
            pdfPath = download_file(request['pdfUrl'], pdfFilename)
            pptFilename = f"{request_id}.pptx"
            pptPath = download_file(request['pptUrl'], pptFilename)

            self.logger.publish("Analyzing text shapes in pptx file...")
            extracted_shapes = extract_text_shapes(pptPath)
            scale_factor = extracted_shapes[0]
            extracted_shapes.pop(0)
            number_of_slides = len(extracted_shapes)

            self.logger.publish("Processing layout types...")
            layout = self.analyze_document_layout(pdfPath, number_of_slides)
            os.remove(pdfPath)

            self.logger.publish("Applying layout types to text shapes...")
            source = self.apply_layout_types(extracted_shapes, layout)
            self.logger.publish("Analyzing shapes complete.")

            self.logger.publish("Initializing translation...")
            # # Write source to txt file
            # with open("source.txt", 'w') as file:
            #     file.write(str(source))
            # Use source (list of slides) to send requests to LLaMAX
            inputJson = []
            for slide in source:
                slideJson = []
                for i, text in enumerate(slide):
                    slideJson.append({
                        'id': i + 1,
                        'Text Type': text['type'],
                        'English': text['text']
                    })
                inputJson.append(str(slideJson).replace('\'', '"'))

            outputJson = []
            for i, slide in enumerate(inputJson):
                self.logger.publish(f"Translating slide #{i + 1} of {number_of_slides}...")
                output_str_list = self.infer(slide)
                with open(f"outputs/output_{request_id}.txt", 'a') as file:
                    file.write(str(output_str_list))
                    file.write('\n\n')

                selected_output = None
                try_gpt = False

                if output_str_list is None:
                    self.logger.publish(f"Translation parsing error in slide #{i + 1}.")
                    try_gpt = True
                elif len(slide) != len(output_str_list):
                    self.logger.publish(f"Translation length error in slide #{i + 1}.")
                    try_gpt = True
                    selected_output = output_str_list
                else:
                    selected_output = output_str_list

                if try_gpt:
                    self.logger.publish(f"Retrying translation for slide #{i + 1}.")
                    gpt_pipeline = PipelinePublic()
                    output_str_list = gpt_pipeline.infer(slide)
                    if output_str_list is None:
                        self.logger.publish(f"Translation parsing error in slide #{i + 1}.")
                    elif len(slide) != len(output_str_list):
                        self.logger.publish(f"Translation length error in slide #{i + 1}.")
                        if selected_output is None:
                            selected_output = output_str_list
                    else:
                        selected_output = output_str_list
                
                outputJson.append(selected_output)

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

            self.logger.publish("Preparing output file for download...")
            uploadUrl = self.upload_output(outputPath)
            self.logger.publish("Output file ready for download.")
            self.logger.publish("DONE")
            self.logger.publish(uploadUrl)
            return True
        
        except Exception as e:
            self.logger.error(e)
            # Clear request from database
            clear_id(request_id)
            self.logger.publish(f"Error occurred. Please provide the following code to the developing team: {request_id}")
            return False
