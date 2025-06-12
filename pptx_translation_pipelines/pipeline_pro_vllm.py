"""
Alternative VLLM implementation without quantization
This version loads the full model and applies LoRA adapter directly,
avoiding quantization format compatibility issues.
"""

import torch
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
import os
import ast
from concurrent.futures import ThreadPoolExecutor
import re
from pipeline_utilities import *
from slide_flipping import process_pptx_flip
from pipeline_public import PipelinePublic
from paddle_classifier import LayoutClassifier


model_name = "gemmax2_9b_finetuned"
base_model = "ModelSpace/GemmaX2-28-9B-Pretrain"
max_tokens = '2048'
batch_size = '16'


class PipelineProVLLM:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(PipelineProVLLM, cls).__new__(cls)
            cls._instance.model_name = model_name
            cls._instance.base_model = base_model
            try:
                cls._instance.max_tokens = int(os.getenv('MAX_TOKENS', max_tokens))
            except (ValueError, TypeError):
                cls._instance.max_tokens = int(max_tokens)
            try:
                cls._instance.batch_size = int(os.getenv('BATCH_SIZE', batch_size))
            except (ValueError, TypeError):
                cls._instance.batch_size = int(batch_size)
            cls._instance.initialize_model()
        return cls._instance

    def initialize_model(self):
        """Initialize VLLM without quantization for maximum compatibility"""
        # --- Configuration ---
        lora_adapter_path = self.model_name
        
        print("Initializing VLLM engine without quantization...")
        print("Note: This will use more GPU memory but avoids quantization compatibility issues")
        
        # VLLM configuration for full precision model
        # Adjust memory settings since we're not using quantization
        self.llm = LLM(
            model=self.base_model,  # Use base model directly
            # No quantization parameter - loads in full precision
            
            # GPU memory optimization - reduce since we're not quantizing
            gpu_memory_utilization=0.9,  # Further reduced to account for no quantization
            max_model_len=4096,  # Significantly reduced to save shared memory
            
            # Shared memory optimization - critical for avoiding shared memory errors
            block_size=16,  # Use smallest block size to minimize shared memory usage
            
            # Performance optimizations
            enforce_eager=True,  # Disable CUDA graphs to save shared memory
            enable_lora=True,  # Enable LoRA support
            max_lora_rank=64,  # Based on your LoRA adapter
            max_loras=1,  # Number of LoRA adapters to cache
            
            # Tensor parallelism (if using multiple GPUs)
            tensor_parallel_size=1,  # Set to number of GPUs if using multiple
            
            # Other optimizations
            disable_log_stats=True,  # Disable logging for production
            trust_remote_code=True,
            
            # Dtype settings - use bfloat16 for better stability
            dtype="bfloat16",  # BFloat16 often works better than Float16
            
            # KV cache optimization - use auto to avoid FP8 kernel compilation issues
            kv_cache_dtype="auto",  # Use model's default dtype to avoid FP8 kernel issues
            
            # Disable prefix caching to avoid potential kernel compilation issues
            enable_prefix_caching=False,  # Disable to avoid potential Triton kernel issues
            
            # Swap space for CPU offloading if needed
            swap_space=4,  # GB of CPU memory to use as swap space
            
            # Seed for reproducibility
            seed=42,
        )
        
        # Load LoRA adapter
        self.lora_request = None
        if os.path.exists(lora_adapter_path):
            print(f"Loading LoRA adapter from: {lora_adapter_path}")
            self.lora_request = LoRARequest(
                lora_name="gemmax2_finetuned",
                lora_int_id=1,
                lora_path=lora_adapter_path,
            )
            print("LoRA adapter loaded successfully!")
        else:
            print(f"Warning: LoRA adapter not found at {lora_adapter_path}")
            print("Running without LoRA adapter...")
        
        print("VLLM engine initialized successfully (no quantization)!")
        print(f"Model loaded on device: cuda")
        print("Memory usage will be higher but compatibility is maximized")
        
        # Set default sampling parameters
        self.default_sampling_params = SamplingParams(
            temperature=0.1,  # Low temperature for consistent translations
            top_p=0.95,
            top_k=50,
            max_tokens=self.max_tokens,  # Significantly reduced to save shared memory
            repetition_penalty=1.05,
            presence_penalty=0.1,
            frequency_penalty=0.1,
        )

    @staticmethod
    def get_prompt(input_json: str, output_json: str = ""):
        instruction = "Translate the following sentences from english to arabic. Return the translations by id in JSON format. MAKE SURE THAT THE NUMBER OF ITEMS IN THE ENGLISH AND ARABIC JSON LISTS IS EQUAL."
        return f"{instruction}\nEnglish: {input_json}\nArabic: {output_json}"

    @staticmethod
    def reevaluate(a: str):
        """Re-evaluate malformed JSON output"""
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

    def infer_batch(self, input_json_list: list, logger: Logger = None):
        """
        Process multiple inputs in a single batch using VLLM's efficient batching
        """
        if not input_json_list:
            return []
        
        # Prepare prompts
        prompts = [self.get_prompt(input_json) for input_json in input_json_list]
        
        # Generate with VLLM
        outputs = self.llm.generate(
            prompts,
            self.default_sampling_params,
            lora_request=self.lora_request,
            use_tqdm=False,
        )
        
        # Process outputs
        results = []
        for i, output in enumerate(outputs):
            try:
                generated_text = output.outputs[0].text.strip()

                # Log the generated text
                if logger is not None:
                    # Get token counts from vLLM RequestOutput object
                    input_token_count = len(output.prompt_token_ids) if output.prompt_token_ids else 0
                    output_token_count = len(output.outputs[0].token_ids) if output.outputs and output.outputs[0].token_ids else 0
                    
                    logger.print_and_write(f"\n\nvLLM Model Output for Slide #{i + 1} in batch:")
                    logger.print_and_write(f"Input text: {prompts[i]}")
                    logger.print_and_write(f"Input token count: {input_token_count}")
                    logger.print_and_write(f"Generated text: {generated_text}")
                    logger.print_and_write(f"Output token count: {output_token_count}")
                    logger.print_and_write(f"Total tokens (input + output): {input_token_count + output_token_count}")
                
                
                # Extract JSON from output
                if 'Arabic: [{"id":' in generated_text:
                    generated_text = '[{"id":' + generated_text.split('Arabic: [{"id":')[1]
                
                # Clean output
                output_str_list = generated_text.replace('\\x0c', '\\n').replace('\\x0b', '\\n')
                
                # Try to parse JSON
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
                print(f"Error processing output: {e}")
                results.append(None)
        
        return results

    def infer(self, input_json: str):
        """Single inference - uses batch inference internally"""
        results = self.infer_batch([input_json])
        return results[0] if results else None

    def run_translation(self, request):
        """Main translation pipeline"""
        logger = Logger("0")
        try:
            request_id = request.get("id", "0")
            logger = Logger(request_id)

            filename = request.get("filename", request_id)
            logger.print_and_write(f"Processing request {request_id}, filename: {filename}, with model id: pro (VLLM No Quant)")

            logger.publish("Fetching files...")
            pdfFilename = f"{request_id}.pdf"
            pdfPath = download_file(request['pdfUrl'], pdfFilename)
            pptFilename = f"{request_id}.pptx"
            pptPath = download_file(request['pptUrl'], pptFilename)

            paddle_classifier = LayoutClassifier()
            source, number_of_slides = paddle_classifier.get_source(pptPath, pdfPath, request_id)

            logger.publish("Initializing translation...")

            # Prepare input JSON
            def process_slide_data(slide_data):
                i, slide = slide_data
                slideJson = []
                for j, text in enumerate(slide):
                    slideJson.append({
                        'id': j + 1,
                        'Text Type': text['type'],
                        'English': text['text']
                    })
                return i, slideJson

            inputJson = [None] * len(source)
            with ThreadPoolExecutor(max_workers=4) as executor:
                slide_results = list(executor.map(process_slide_data, enumerate(source)))
            
            for i, slideJson in slide_results:
                inputJson[i] = slideJson

            if not os.path.exists("outputs"):
                os.makedirs("outputs")
            
            logger.publish("Processing slides with VLLM batch translation...")
            
            # Prepare batch data for slides with content
            batch_slides = []
            slide_indices = []
            
            for i, slide in enumerate(inputJson):
                if len(slide) == 0:
                    logger.publish(f"Found empty slide: #{i + 1} of {number_of_slides}")
                else:
                    batch_slides.append(str(slide).replace('\'', '"'))
                    slide_indices.append(i)
            
            # Process slides in batches
            batch_size = min(self.batch_size, len(batch_slides))
            outputJson = [None] * number_of_slides
            
            if batch_slides:
                logger.publish(f"Processing {len(batch_slides)} slides in batches of {batch_size}...")
                
                for batch_start in range(0, len(batch_slides), batch_size):
                    batch_end = min(batch_start + batch_size, len(batch_slides))
                    current_batch = batch_slides[batch_start:batch_end]
                    current_indices = slide_indices[batch_start:batch_end]
                    
                    logger.publish(f"Translating slides {current_indices[0] + 1}-{current_indices[-1] + 1} of {number_of_slides}...")
                    
                    # Use VLLM batch inference
                    batch_results = self.infer_batch(current_batch, logger=logger)
                    
                    # Process batch results
                    for local_idx, (slide_idx, output_list) in enumerate(zip(current_indices, batch_results)):
                        selected_output = None
                        try_gpt = False
                        original_slide = inputJson[slide_idx]
                        
                        if output_list is None:
                            logger.publish(f"Translation parsing error in slide #{slide_idx + 1}.")
                            try_gpt = True
                        elif len(original_slide) != len(output_list):
                            logger.publish(f"Translation length error in slide #{slide_idx + 1}.")
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

            # Process output
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
            clear_id(request_id)
            return True
        
        except Exception as e:
            logger.error(e)
            clear_id(request_id)
            logger.publish(f"Error occurred. Please provide the following code to the developing team: {request_id}")
            return False 