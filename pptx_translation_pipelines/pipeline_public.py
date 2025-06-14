import os
import time
import json
import ast
import re # For parsing adjustment values

from openai import OpenAI
import concurrent.futures
import tiktoken
from collections import deque
import threading
import ast
import boto3
from botocore.exceptions import ClientError

from pipeline_utilities import *
from slide_flipping import process_pptx_flip
from paddle_classifier import LayoutClassifier


tpm_limit = 450000
parallelWorkers = 20
model = "chatgpt-4o-latest"


class PipelinePublic:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(PipelinePublic, cls).__new__(cls)
            cls._instance.model = model
            cls._instance.client = OpenAI(
                api_key=cls.get_secret()
            )
            cls._instance.parallelWorkers = parallelWorkers
            cls._instance.token_limiter = PipelinePublic.TokenRateLimiter(tpm_limit=tpm_limit, model_name=model)
        return cls._instance


    def get_secret():
        secret_name = "openAi/apiKeys"
        region_name = "us-east-1"

        # Create a Secrets Manager client
        session = boto3.session.Session()
        client = session.client(
            service_name='secretsmanager',
            region_name=region_name
        )

        try:
            get_secret_value_response = client.get_secret_value(
                SecretId=secret_name
            )
        except ClientError as e:
            # For a list of exceptions thrown, see
            # https://docs.aws.amazon.com/secretsmanager/latest/apireference/API_GetSecretValue.html
            raise e

        secret = get_secret_value_response['SecretString']
        return ast.literal_eval(secret)['aita-pipeline-ec2-1']


    class TokenRateLimiter:
        EXPECTED_COMPLETION_TOKENS_ESTIMATE = 800
        def __init__(self, tpm_limit=30000, model_name="chatgpt-4o-latest"):
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

        response = self.client.chat.completions.create(
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
        try:
            response = json.loads(response)
            return response
        except Exception as e:
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
    

    def parallel_infer(self, inputJson: list, logger: Logger = None):
        def log(message):
            if logger is not None:
                logger.publish(message)

        number_of_slides = len(inputJson)
        outputJson = [None] * number_of_slides
        with concurrent.futures.ThreadPoolExecutor(max_workers=parallelWorkers) as executor:
            futures = []
            for index, prompt in enumerate(inputJson):
                if len(prompt) > 0:
                    futures.append(executor.submit(self.infer, str(prompt).replace('\'', '"'), index))
                else:
                    log(f"Found empty slide: #{index + 1} of {number_of_slides}")
            for future in concurrent.futures.as_completed(futures):
                i, output_list = future.result()

                log(f"Translated slide #{i + 1} of {number_of_slides}...")
                if output_list is None:
                    outputJson[i] = None
                    log(f"Translation parsing error in slide #{i + 1}.")
                else:
                    if len(inputJson[i]) != len(output_list):
                        log(f"Translation length error in slide #{i + 1}.")
                    outputJson[i] = output_list
        
        return outputJson


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
                        'text_type': text['type'],
                        'english': text['text']
                    })
                inputJson.append(slideJson)
                outputJson.append(None)

            if not os.path.exists("outputs"):
                os.makedirs("outputs")

            # TODO: replace this with parallel_infer
            with concurrent.futures.ThreadPoolExecutor(max_workers=parallelWorkers) as executor:
                # futures = [executor.submit(self.infer, prompt, index) for index, prompt in enumerate(inputJson)]
                futures = []
                for index, prompt in enumerate(inputJson):
                    if len(prompt) > 0:
                        futures.append(executor.submit(self.infer, str(prompt).replace('\'', '"'), index))
                    else:
                        logger.publish(f"Found empty slide: #{index + 1} of {number_of_slides}")
                for future in concurrent.futures.as_completed(futures):
                    i, output_list = future.result()

                    logger.publish(f"Translated slide #{i + 1} of {number_of_slides}...")
                    if output_list is None:
                        logger.publish(f"Translation parsing error in slide #{i + 1}.")
                        outputJson[i] = None
                    else:
                        if len(inputJson[i]) != len(output_list):
                            logger.publish(f"Translation length error in slide #{i + 1}.")
                        outputJson[i] = output_list

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
