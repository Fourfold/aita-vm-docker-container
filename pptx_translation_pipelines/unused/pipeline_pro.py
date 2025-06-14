import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer, BitsAndBytesConfig # Added BitsAndBytesConfig
from peft import PeftModel # Added PeftModel
import os
import ast
from concurrent.futures import ThreadPoolExecutor
import re # For parsing adjustment values
from pipeline_utilities import *
from slide_flipping import process_pptx_flip
from pipeline_public import PipelinePublic
from paddle_classifier import LayoutClassifier


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
            try:
                if quantization_bit == 4:
                    # Check if bfloat16 is supported, fallback to float16
                    try:
                        bf16_supported = torch.cuda.is_bf16_supported() if hasattr(torch.cuda, 'is_bf16_supported') else False
                    except:
                        bf16_supported = False
                    
                    compute_dtype = torch.bfloat16 if bf16_supported else torch.float16
                    torch_dtype = compute_dtype
                    
                    bnb_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=compute_dtype
                    )
                    model_kwargs["torch_dtype"] = torch_dtype
                    print(f"Configured 4-bit quantization with {torch_dtype}")
                    
                elif quantization_bit == 8:
                    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
                    print("Configured 8-bit quantization")

                model_kwargs["quantization_config"] = bnb_config
                
            except Exception as e:
                print(f"Error setting up quantization, loading without quantization: {e}")
                use_quantization = False

        # Load the model with device_map and optional quantization
        # The device_map="auto" should handle device placement automatically
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            **model_kwargs
        )
        print(f"Base model loaded successfully on device: {self.model.device}")  # Should show cuda if successful

        # --- Load LoRA Adapters ---
        print(f"Loading LoRA adapters from: {lora_adapter_path}...")
        # Note: PeftModel.from_pretrained should preserve the device and quantization of the base model
        self.model = PeftModel.from_pretrained(self.model, lora_adapter_path)
        print("LoRA adapters loaded successfully.")
        print(f"Final model (with adapters) ready on device: {self.model.device}")
        
        # Optimize model for better GPU utilization
        print("Optimizing model for better performance...")
        
        # Set model to evaluation mode (essential)
        self.model.eval()
        
        # Enable basic optimizations with error handling for each
        try:
            # Enable optimized attention if available
            self.model.config.use_cache = True
            print("Enabled model caching")
        except Exception as e:
            print(f"Could not enable caching: {e}")
        
        try:
            # Enable Flash Attention 2 if available (much faster)
            if hasattr(self.model.config, 'attn_implementation'):
                # Only set if not already configured to avoid conflicts
                if not hasattr(self.model.config, '_attn_implementation_internal'):
                    self.model.config.attn_implementation = "flash_attention_2"
                    print("Enabled Flash Attention 2")
            else:
                print("Flash Attention 2 not available for this model")
        except Exception as e:
            print(f"Could not enable Flash Attention 2: {e}")
        
        try:
            # Enable CUDA optimizations if available
            if torch.cuda.is_available():
                torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
                print("Enabled CUDNN benchmark mode")
                
                # Enable TF32 if supported (Ampere and newer GPUs)
                if hasattr(torch.backends.cuda.matmul, 'allow_tf32'):
                    torch.backends.cuda.matmul.allow_tf32 = True
                    print("Enabled TF32 for matrix operations")
                
                if hasattr(torch.backends.cudnn, 'allow_tf32'):
                    torch.backends.cudnn.allow_tf32 = True
                    print("Enabled TF32 for convolutions")
                
                # Set memory pool settings for better allocation (optional)
                try:
                    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512,roundup_power2_divisions:16'
                    print("Configured CUDA memory allocation settings")
                except Exception as e:
                    print(f"Could not set CUDA memory settings: {e}")
                    
        except Exception as e:
            print(f"Could not enable CUDA optimizations: {e}")
        
        # Skip torch.compile due to compatibility issues with current environment
        print("Skipping torch.compile optimization due to compatibility issues")
        
        print("Model optimization complete!")


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
        except Exception as e:
            return None


    def infer_batch(self, input_json_list: list, use_text_stream: bool = False, logger: Logger = None):
        """
        Process multiple inputs in a single batch for better GPU utilization
        """
        if not input_json_list:
            return []
            
        EOS_TOKEN = self.tokenizer.eos_token
        
        # Prepare batch prompts - parallelize this CPU-intensive task
        with ThreadPoolExecutor(max_workers=4) as executor:
            prompts = list(executor.map(PipelinePro.get_prompt, input_json_list))
        
        # Tokenize batch with padding - this is CPU intensive
        inputs = self.tokenizer(
            prompts, 
            return_tensors="pt", 
            padding=True,  # Pad to same length
            truncation=True,  # Truncate if too long
            max_length=2048,  # Adjust based on your needs
        ).to("cuda")
        
        # Generate for entire batch with optimized settings
        with torch.no_grad():  # Save memory
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=4096, 
                use_cache=True,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=False,  # Deterministic for consistency
                num_beams=1,  # Faster than beam search
                temperature=1.0,  # Disable sampling overhead
                top_p=1.0,  # Disable top-p sampling overhead
                repetition_penalty=1.0  # Disable repetition penalty overhead
            )
        
        # Process batch results in parallel
        results = [None] * len(outputs)
        input_lengths = inputs["attention_mask"].sum(dim=1)  # Get actual input lengths
        
        def process_single_output(args):
            i, output, input_length = args
            try:
                # Extract only the generated part
                generated_tokens = output[input_length:]
                generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

                # Log the generated text
                if logger is not None:
                    input_token_count = len(inputs['input_ids'][i])
                    output_token_count = len(generated_tokens)
                    logger.print_and_write(f"\n\nGemmaX2 Model Output for Slide #{i + 1} in batch:")
                    logger.print_and_write(f"Input text: {prompts[i]}")
                    logger.print_and_write(f"Input token count: {input_token_count}")
                    logger.print_and_write(f"Generated text: {generated_text}")
                    logger.print_and_write(f"Output token count: {output_token_count}")
                    logger.print_and_write(f"Total tokens (input + output): {input_token_count + output_token_count}")
                
                # Clean up the output
                if 'Arabic: [{"id":' in generated_text:
                    generated_text = '[{"id":' + generated_text.split('Arabic: [{"id":')[1]
                
                output_str_list = generated_text.replace('\\x0c', '\\n').replace('\\x0b', '\\n')
                if output_str_list.endswith(EOS_TOKEN):
                    output_str_list = output_str_list[:-len(EOS_TOKEN)]
                
                try:
                    out_list_repr = ast.literal_eval(output_str_list)
                except (ValueError, SyntaxError, TypeError):
                    out_list_repr = PipelinePro.reevaluate(output_str_list)
                
                # Validate output format
                if out_list_repr is not None and isinstance(out_list_repr, list):
                    valid = all(isinstance(item, dict) and "id" in item and "Arabic" in item 
                              for item in out_list_repr)
                    if not valid:
                        out_list_repr = None
                        
                return i, out_list_repr
                
            except Exception as e:
                print(f"Error processing batch item {i}: {e}")
                return i, None
        
        # Parallelize output processing
        with ThreadPoolExecutor(max_workers=4) as executor:
            processed_outputs = list(executor.map(
                process_single_output, 
                [(i, output, input_length) for i, (output, input_length) in enumerate(zip(outputs, input_lengths))]
            ))
        
        # Sort results back to original order
        for i, result in processed_outputs:
            results[i] = result
        
        return results


    def infer(self, input_json: str, use_text_stream: bool = False):
        EOS_TOKEN = self.tokenizer.eos_token
        inputs = self.tokenizer([PipelinePro.get_prompt(input_json)], return_tensors="pt").to("cuda")
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

            paddle_classifier = LayoutClassifier()
            source, number_of_slides = paddle_classifier.get_source(pptPath, pdfPath, request_id)

            logger.publish("Initializing translation...")

            # Parallelize input JSON processing across CPU cores
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
            
            # Sort results back to original order
            for i, slideJson in slide_results:
                inputJson[i] = slideJson

            if not os.path.exists("outputs"):
                os.makedirs("outputs")
            
            logger.publish("Processing slides with batch translation...")
            
            # Prepare batch data for slides with content
            batch_slides = []
            slide_indices = []
            
            for i, slide in enumerate(inputJson):
                if len(slide) == 0:
                    logger.publish(f"Found empty slide: #{i + 1} of {number_of_slides}")
                else:
                    batch_slides.append(str(slide).replace('\'', '"'))
                    slide_indices.append(i)
            
            # Process slides in batches for better GPU utilization
            # With 15GB VRAM headroom, we can safely increase batch size
            batch_size = min(8, len(batch_slides))  # Increased from 4 to 8
            outputJson = [None] * number_of_slides  # Initialize with None for all slides
            
            if batch_slides:
                logger.publish(f"Processing {len(batch_slides)} slides in batches of {batch_size}...")
                
                for batch_start in range(0, len(batch_slides), batch_size):
                    batch_end = min(batch_start + batch_size, len(batch_slides))
                    current_batch = batch_slides[batch_start:batch_end]
                    current_indices = slide_indices[batch_start:batch_end]
                    
                    logger.publish(f"Translating slides {current_indices[0] + 1}-{current_indices[-1] + 1} of {number_of_slides}...")
                    
                    # Use batch inference
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
