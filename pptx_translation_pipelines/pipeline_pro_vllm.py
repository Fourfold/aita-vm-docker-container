import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
import os
import ast
import re # For parsing adjustment values
from pipeline_utilities import *
from slide_flipping import process_pptx_flip
from pipeline_public import PipelinePublic
from paddle_classifier import LayoutClassifier


model_name = "gemmax2_9b_finetuned"
base_model = "ModelSpace/GemmaX2-28-9B-Pretrain"


class PipelineProVLLM:
    """
    Enhanced version of PipelinePro using vLLM for faster inference
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(PipelineProVLLM, cls).__new__(cls)
            cls._instance.model_name = model_name
            cls._instance.base_model = base_model
            cls._instance.initialize_model()
        return cls._instance

    def initialize_model(self):
        # --- Configuration ---
        lora_adapter_path = self.model_name
        
        # --- Check for GPU ---
        if not torch.cuda.is_available():
            raise SystemError("CUDA is not available. Please ensure you have a GPU and CUDA installed.")

        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model)

        print("Loading vLLM model with LoRA support...")
        # Initialize vLLM with LoRA support
        self.model = LLM(
            model=self.base_model,
            enable_lora=True,
            max_lora_rank=128,  # Adjust based on your LoRA rank
            max_loras=1,  # Number of LoRA adapters to support
            max_model_len=4096,  # Adjust based on your needs
            gpu_memory_utilization=0.9,  # Use 90% of GPU memory
            tensor_parallel_size=1,  # Adjust if using multiple GPUs
            # Note: vLLM handles quantization differently - you may need to prepare a quantized model
            # quantization="bitsandbytes",  # This might not be directly supported
            load_format="auto"
        )
        
        # Create LoRA request for inference
        self.lora_request = LoRARequest(
            lora_name="translation_adapter",
            lora_int_id=1,
            lora_path=lora_adapter_path
        )
        
        print(f"vLLM model with LoRA adapters loaded successfully.")

    @staticmethod
    def get_prompt(input_json: str, output_json: str = ""):
        instruction = "Translate the following sentences from english to arabic. Return the translations by id in JSON format. MAKE SURE THAT THE NUMBER OF ITEMS IN THE ENGLISH AND ARABIC JSON LISTS IS EQUAL."
        return f"{instruction}\nEnglish: {input_json}\nArabic: {output_json}"

    @staticmethod
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
        except Exception as e:
            return None

    def infer_batch(self, input_json_list: list, use_text_stream: bool = False):
        """
        Process multiple inputs in a single batch using vLLM for better GPU utilization
        """
        if not input_json_list:
            return []
        
        # Prepare batch prompts
        prompts = [self.get_prompt(input_json) for input_json in input_json_list]
        
        # Configure sampling parameters - these are optimized for vLLM
        sampling_params = SamplingParams(
            temperature=0.0,  # Deterministic for consistency
            max_tokens=4096,
            stop_token_ids=[self.tokenizer.eos_token_id] if self.tokenizer.eos_token_id else None,
            skip_special_tokens=True,
            use_beam_search=False,  # Faster than beam search
        )
        
        # Generate for entire batch using vLLM
        try:
            outputs = self.model.generate(
                prompts, 
                sampling_params, 
                lora_request=self.lora_request
            )
            
            # Process batch results
            results = []
            for i, output in enumerate(outputs):
                try:
                    generated_text = output.outputs[0].text
                    
                    # Clean up the output
                    if 'Arabic: [{"id":' in generated_text:
                        generated_text = '[{"id":' + generated_text.split('Arabic: [{"id":')[1]
                    
                    output_str_list = generated_text.replace('\\x0c', '\\n').replace('\\x0b', '\\n')
                    
                    try:
                        out_list_repr = ast.literal_eval(output_str_list)
                    except (ValueError, SyntaxError, TypeError):
                        out_list_repr = self.reevaluate(output_str_list)
                    
                    # Validate output format
                    if out_list_repr is not None and isinstance(out_list_repr, list):
                        valid = all(isinstance(item, dict) and "id" in item and "Arabic" in item 
                                  for item in out_list_repr)
                        if not valid:
                            out_list_repr = None
                            
                    results.append(out_list_repr)
                    
                except Exception as e:
                    print(f"Error processing batch item {i}: {e}")
                    results.append(None)
            
            return results
            
        except Exception as e:
            print(f"Error in vLLM batch inference: {e}")
            return [None] * len(input_json_list)

    def infer(self, input_json: str, use_text_stream: bool = False):
        """
        Single inference using vLLM
        """
        prompts = [self.get_prompt(input_json)]
        
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=4096,
            stop_token_ids=[self.tokenizer.eos_token_id] if self.tokenizer.eos_token_id else None,
            skip_special_tokens=True
        )
        
        try:
            outputs = self.model.generate(prompts, sampling_params, lora_request=self.lora_request)
            generated_text = outputs[0].outputs[0].text
            
            # Clean up the output
            if 'Arabic: [{"id":' in generated_text:
                generated_text = '[{"id":' + generated_text.split('Arabic: [{"id":')[1]
            
            output_str_list = generated_text.replace('\\x0c', '\\n').replace('\\x0b', '\\n')

            try:
                out_list_repr = ast.literal_eval(output_str_list)
            except (ValueError, SyntaxError, TypeError) as e:
                out_list_repr = self.reevaluate(output_str_list)

            if out_list_repr is not None:
                if isinstance(out_list_repr, list):
                    for item in out_list_repr:
                        if isinstance(item, dict) and "id" in item and "Arabic" in item:
                            pass
                        else:
                            out_list_repr = None

            return out_list_repr
            
        except Exception as e:
            print(f"Error in vLLM inference: {e}")
            return None

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
            logger.print_and_write(f"Processing request {request_id}, filename: {filename}, with model id: pro-vllm")

            logger.publish("Fetching files...")
            pdfFilename = f"{request_id}.pdf"
            pdfPath = download_file(request['pdfUrl'], pdfFilename)
            pptFilename = f"{request_id}.pptx"
            pptPath = download_file(request['pptUrl'], pptFilename)

            paddle_classifier = LayoutClassifier()
            source, number_of_slides = paddle_classifier.get_source(pptPath, pdfPath, request_id)

            logger.publish("Initializing translation...")

            inputJson = []
            for slide in source:
                slideJson = []
                for i, text in enumerate(slide):
                    slideJson.append({
                        'id': i + 1,
                        'Text Type': text['type'],
                        'English': text['text']
                    })
                inputJson.append(slideJson)

            if not os.path.exists("outputs"):
                os.makedirs("outputs")
            
            logger.publish("Processing slides with vLLM batch translation...")
            
            # Prepare batch data for slides with content
            batch_slides = []
            slide_indices = []
            
            for i, slide in enumerate(inputJson):
                if len(slide) == 0:
                    logger.publish(f"Found empty slide: #{i + 1} of {number_of_slides}")
                else:
                    batch_slides.append(str(slide).replace('\'', '"'))
                    slide_indices.append(i)
            
            # Process slides in optimal batches for vLLM
            # vLLM can handle larger batches more efficiently than transformers
            batch_size = min(8, len(batch_slides))  # Increased batch size for vLLM
            outputJson = [None] * number_of_slides  # Initialize with None for all slides
            
            if batch_slides:
                logger.publish(f"Processing {len(batch_slides)} slides in batches of {batch_size} using vLLM...")
                
                for batch_start in range(0, len(batch_slides), batch_size):
                    batch_end = min(batch_start + batch_size, len(batch_slides))
                    current_batch = batch_slides[batch_start:batch_end]
                    current_indices = slide_indices[batch_start:batch_end]
                    
                    logger.publish(f"Translating slides {current_indices[0] + 1}-{current_indices[-1] + 1} of {number_of_slides} with vLLM...")
                    
                    # Use vLLM batch inference
                    batch_results = self.infer_batch(current_batch)
                    
                    # Process batch results
                    for local_idx, (slide_idx, output_list) in enumerate(zip(current_indices, batch_results)):
                        selected_output = None
                        try_gpt = False
                        original_slide = inputJson[slide_idx]
                        
                        if output_list is None:
                            logger.publish(f"vLLM translation parsing error in slide #{slide_idx + 1}.")
                            try_gpt = True
                        elif len(original_slide) != len(output_list):
                            logger.publish(f"vLLM translation length error in slide #{slide_idx + 1}.")
                            try_gpt = True
                            selected_output = output_list
                        else:
                            selected_output = output_list

                        if try_gpt:
                            logger.publish(f"Retrying translation for slide #{slide_idx + 1} with GPT...")
                            gpt_pipeline = PipelinePublic()
                            _, gpt_output_list = gpt_pipeline.infer(str(original_slide).replace('\'', '"'))
                            if gpt_output_list is None:
                                logger.publish(f"GPT translation parsing error in slide #{slide_idx + 1}.")
                            elif len(original_slide) != len(gpt_output_list):
                                logger.publish(f"GPT translation length error in slide #{slide_idx + 1}.")
                                if selected_output is None:
                                    selected_output = gpt_output_list
                            else:
                                selected_output = gpt_output_list
                        
                        outputJson[slide_idx] = selected_output

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
            # TODO: Delete output folder after upload
            logger.publish("Output file ready for download.")
            logger.publish("DONE")
            logger.publish(uploadUrl)
            clear_id(request_id)
            return True
        
        except Exception as e:
            logger.error(e)
            # Clear request from database
            clear_id(request_id)
            logger.publish(f"Error occurred. Please provide the following code to the developing team: {request_id}")
            return False 