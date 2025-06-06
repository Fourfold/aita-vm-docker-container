import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer, BitsAndBytesConfig # Added BitsAndBytesConfig
from peft import PeftModel # Added PeftModel
from paddlex import create_model
from pdf2image import convert_from_path
import os
import traceback
import zipfile
import time
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
from paddle_classifier import PaddleClassifier


model_name = "gemmax2_9b_finetuned"
base_model = "ModelSpace/GemmaX2-28-9B-Pretrain"


class PipelinePro:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(PipelinePro, cls).__new__(cls)
            cls._instance.model_name = model_name
            cls._instance.base_model = base_model
            cls._instance.initialize_model()
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


    def run_translation(self, request):
        logger = Logger("0")
        try:
            request_id = request.get("id")
            if request_id is None:
                request_id = "0"
            
            logger = Logger(request_id)

            filename = request.get("filename")
            if filename is None:
                filename = request_id
            logger.print_and_write(f"Processing request {request_id}, filename: {filename}, with model id: pro")

            logger.publish("Fetching files...")
            pdfFilename = f"{request_id}.pdf"
            pdfPath = download_file(request['pdfUrl'], pdfFilename)
            pptFilename = f"{request_id}.pptx"
            pptPath = download_file(request['pptUrl'], pptFilename)

            paddle_classifier = PaddleClassifier()
            source, number_of_slides = paddle_classifier.get_source(pptPath, pdfPath, request_id)

            logger.publish("Initializing translation...")
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
                logger.publish(f"Translating slide #{i + 1} of {number_of_slides}...")
                output_str_list = self.infer(slide)
                with open(f"outputs/output_{request_id}.txt", 'a') as file:
                    file.write(str(output_str_list))
                    file.write('\n\n')

                selected_output = None
                try_gpt = False

                if output_str_list is None:
                    logger.publish(f"Translation parsing error in slide #{i + 1}.")
                    try_gpt = True
                elif len(slide) != len(output_str_list):
                    logger.publish(f"Translation length error in slide #{i + 1}.")
                    try_gpt = True
                    selected_output = output_str_list
                else:
                    selected_output = output_str_list

                if try_gpt:
                    logger.publish(f"Retrying translation for slide #{i + 1}.")
                    gpt_pipeline = PipelinePublic()
                    output_str_list = gpt_pipeline.infer(slide)
                    if output_str_list is None:
                        logger.publish(f"Translation parsing error in slide #{i + 1}.")
                    elif len(slide) != len(output_str_list):
                        logger.publish(f"Translation length error in slide #{i + 1}.")
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

            logger.publish("Preparing output file for download...")
            uploadUrl = self.upload_output(outputPath)
            logger.publish("Output file ready for download.")
            logger.publish("DONE")
            logger.publish(uploadUrl)
            return True
        
        except Exception as e:
            logger.error(e)
            # Clear request from database
            clear_id(request_id)
            logger.publish(f"Error occurred. Please provide the following code to the developing team: {request_id}")
            return False
