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

os.environ["VLLM_ATTENTION_BACKEND"] = "FLASHINFER"  # Prefer FlashInfer for Gemma2

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
        
        # Get GPU info for better configuration
        gpu_count = torch.cuda.device_count()
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
        print(f"Detected {gpu_count} GPU(s), GPU 0 memory: {gpu_memory:.1f}GB")

        print("Loading tokenizer...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.base_model)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            print(f"Tokenizer loaded successfully. Vocab size: {self.tokenizer.vocab_size}")
        except Exception as e:
            print(f"Error loading tokenizer: {e}")
            raise

        print("Loading vLLM model with LoRA support...")
        
        # Determine optimal settings based on GPU memory
        gpu_memory_util = 0.9 if gpu_memory > 16 else 0.8  # More conservative for smaller GPUs
        max_model_len = 4096 if gpu_memory > 12 else 2048  # Reduce context for smaller GPUs
        
        # Check dtype compatibility
        try:
            bf16_supported = torch.cuda.is_bf16_supported() if hasattr(torch.cuda, 'is_bf16_supported') else False
        except:
            bf16_supported = False
        
        # Choose dtype based on compatibility and performance
        if bf16_supported and gpu_memory > 16:
            model_dtype = "bfloat16"
        else:
            model_dtype = "float16"
        
        print(f"Using dtype: {model_dtype}, max_model_len: {max_model_len}, gpu_memory_util: {gpu_memory_util}")
        
        try:
            # Initialize vLLM with LoRA support and error handling
            self.model = LLM(
                model=self.base_model,
                enable_lora=True,
                max_lora_rank=128,  # Adjust based on your LoRA rank
                max_loras=1,  # Number of LoRA adapters to support
                max_model_len=max_model_len,
                gpu_memory_utilization=gpu_memory_util,
                tensor_parallel_size=1,  # Adjust if using multiple GPUs
                dtype=model_dtype,
                load_format="auto",
                trust_remote_code=True,  # May be needed for some models
                enforce_eager=False,  # Set to True if you encounter CUDA graph issues
            )
            print(f"vLLM model loaded successfully with dtype {model_dtype}")
        except Exception as e:
            print(f"Error loading vLLM model: {e}")
            print("Trying with more conservative settings...")
            # Fallback with more conservative settings
            try:
                self.model = LLM(
                    model=self.base_model,
                    enable_lora=True,
                    max_lora_rank=64,  # Reduced rank
                    max_loras=1,
                    max_model_len=2048,  # Reduced context length
                    gpu_memory_utilization=0.7,  # More conservative memory usage
                    tensor_parallel_size=1,
                    dtype="float16",  # Always use float16 for compatibility
                    load_format="auto",
                    trust_remote_code=True,
                    enforce_eager=True,  # Force eager execution for compatibility
                )
                print("vLLM model loaded with conservative settings")
            except Exception as e2:
                print(f"Failed to load vLLM model even with conservative settings: {e2}")
                raise
        
        # Create LoRA request for inference
        try:
            self.lora_request = LoRARequest(
                lora_name="translation_adapter",
                lora_int_id=1,
                lora_path=lora_adapter_path
            )
            print(f"LoRA request created successfully for path: {lora_adapter_path}")
        except Exception as e:
            print(f"Error creating LoRA request: {e}")
            print("Continuing without LoRA adapters...")
            self.lora_request = None
        
        print(f"vLLM model initialization completed successfully.")

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
        try:
            stop_token_ids = []
            if hasattr(self.tokenizer, 'eos_token_id') and self.tokenizer.eos_token_id is not None:
                stop_token_ids = [self.tokenizer.eos_token_id]
            
            # Adjust max_tokens based on model's max length
            model_max_len = getattr(self.model.llm_engine.model_config, 'max_model_len', 4096)
            max_tokens = min(4096, model_max_len - 512)  # Leave room for input tokens
            
            sampling_params = SamplingParams(
                temperature=0.0,  # Deterministic for consistency
                max_tokens=max_tokens,
                stop_token_ids=stop_token_ids if stop_token_ids else None,
                skip_special_tokens=True,
                use_beam_search=False,  # Faster than beam search
                repetition_penalty=1.0,  # Prevent repetition
            )
        except Exception as e:
            print(f"Warning: Error configuring sampling parameters, using defaults: {e}")
            sampling_params = SamplingParams(
                temperature=0.0,
                max_tokens=2048,  # Conservative default
                skip_special_tokens=True,
                use_beam_search=False,
            )
        
        # Generate for entire batch using vLLM
        try:
            # Only use LoRA request if it was successfully created
            generate_kwargs = {
                "prompts": prompts,
                "sampling_params": sampling_params,
            }
            
            if self.lora_request is not None:
                generate_kwargs["lora_request"] = self.lora_request
            else:
                print("Warning: Running without LoRA adapters")
            
            outputs = self.model.generate(**generate_kwargs)
            
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
        
        # Reuse the same sampling parameter logic for consistency
        try:
            stop_token_ids = []
            if hasattr(self.tokenizer, 'eos_token_id') and self.tokenizer.eos_token_id is not None:
                stop_token_ids = [self.tokenizer.eos_token_id]
            
            model_max_len = getattr(self.model.llm_engine.model_config, 'max_model_len', 4096)
            max_tokens = min(4096, model_max_len - 512)
            
            sampling_params = SamplingParams(
                temperature=0.0,
                max_tokens=max_tokens,
                stop_token_ids=stop_token_ids if stop_token_ids else None,
                skip_special_tokens=True,
                repetition_penalty=1.0,
            )
        except Exception as e:
            print(f"Warning: Error configuring sampling parameters, using defaults: {e}")
            sampling_params = SamplingParams(
                temperature=0.0,
                max_tokens=2048,
                skip_special_tokens=True,
            )
        
        try:
            # Only use LoRA request if it was successfully created
            generate_kwargs = {
                "prompts": prompts,
                "sampling_params": sampling_params,
            }
            
            if self.lora_request is not None:
                generate_kwargs["lora_request"] = self.lora_request
            
            outputs = self.model.generate(**generate_kwargs)
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
            try:
                pdfFilename = f"{request_id}.pdf"
                pdfPath = download_file(request['pdfUrl'], pdfFilename)
                pptFilename = f"{request_id}.pptx"
                pptPath = download_file(request['pptUrl'], pptFilename)
                logger.publish("Files downloaded successfully.")
            except Exception as e:
                logger.publish(f"Error downloading files: {e}")
                raise

            try:
                paddle_classifier = LayoutClassifier()
                source, number_of_slides = paddle_classifier.get_source(pptPath, pdfPath, request_id)
                logger.publish(f"Layout classification completed. Found {number_of_slides} slides.")
            except Exception as e:
                logger.publish(f"Error in layout classification: {e}")
                raise

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