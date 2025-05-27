from openai import OpenAI
import time
import tiktoken
from collections import deque
import threading
from fastapi import FastAPI, HTTPException, Response

# Default estimate for completion tokens, can be tuned by the user if needed
EXPECTED_COMPLETION_TOKENS_ESTIMATE = 600  # Assuming an average, can be adjusted
parallelWorkers = 5
model = "gpt-3.5-turbo"
tpm_limit = 30000
client = OpenAI(
        api_key='sk-proj-zo8UYHSjQsRDj27x0UgPT3BlbkFJRhKTphRtEu8ITjFjfRBS'
    )
app = FastAPI()
token_limiter = None

class TokenRateLimiter:
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
        estimated_total_tokens = prompt_tokens_for_this_request + EXPECTED_COMPLETION_TOKENS_ESTIMATE
        
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


def chat_gpt(prompt, token_limiter):
    # Estimate prompt tokens for the current request
    prompt_tokens = token_limiter.estimate_prompt_tokens(prompt)
    
    # Wait if necessary based on current token usage and estimated tokens for this call
    token_limiter.wait_if_needed(prompt_tokens)

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    response_message = response.choices[0].message.content
    
    # Record actual tokens used (prompt + completion)
    if response.usage:
        actual_tokens = response.usage.total_tokens
        token_limiter.add_request_tokens(actual_tokens)
    
    # time.sleep(2)  # Removed: Replaced by the TokenRateLimiter logic
    return response_message.strip()

@app.on_event("startup")
async def startup():
    global token_limiter
    # Instantiate the limiter globally with the user-specified TPM limit
    token_limiter = TokenRateLimiter(tpm_limit=tpm_limit, model_name=model)
    print("Startup complete")

# --- API Endpoints ---
# GET request allowed for health check, works in browser
@app.get("/health")
def health_check():
    return {"status": "healthy"}

# POST request for actual API call, does not work in browser
@app.post("/gpt")
async def gpt(prompt: str):
    try:
        return Response(
            content=chat_gpt(prompt, token_limiter),
            media_type="text/plain"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

