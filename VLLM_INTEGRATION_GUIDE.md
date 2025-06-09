# vLLM Integration Guide for Your Translation Pipeline

## Overview
This guide shows how to integrate vLLM into your existing translation pipeline for significantly faster inference while maintaining LoRA adapter support.

## Performance Benefits
Based on benchmarks, vLLM typically provides:
- **2-4x faster inference** compared to transformers
- **Better memory efficiency** through PagedAttention
- **Higher throughput** especially with batch processing
- **Up to 43x faster** for batch processing compared to individual requests

## Key Changes

### 1. Model Initialization
**Before (Transformers):**
```python
self.model = AutoModelForCausalLM.from_pretrained(
    self.base_model,
    quantization_config=bnb_config,
    device_map="auto"
)
self.model = PeftModel.from_pretrained(self.model, lora_adapter_path)
```

**After (vLLM):**
```python
self.model = LLM(
    model=self.base_model,
    enable_lora=True,
    max_lora_rank=128,
    max_loras=1,
    max_model_len=4096,
    gpu_memory_utilization=0.9,
    tensor_parallel_size=1
)

self.lora_request = LoRARequest(
    lora_name="translation_adapter",
    lora_int_id=1,
    lora_path=lora_adapter_path
)
```

### 2. Inference Changes
**Before (Transformers):**
```python
inputs = self.tokenizer([prompt], return_tensors="pt").to("cuda")
outputs = self.model.generate(**inputs, max_new_tokens=4096)
```

**After (vLLM):**
```python
sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=4096,
    stop_token_ids=[self.tokenizer.eos_token_id]
)
outputs = self.model.generate(prompts, sampling_params, lora_request=self.lora_request)
```

## Installation Steps

### 1. Install vLLM
```bash
pip install vllm>=0.7.0
# or install from requirements_vllm.txt
pip install -r requirements_vllm.txt
```

### 2. Prepare Your LoRA Adapter
vLLM supports LoRA adapters, but there are some considerations:

**Option A: Use existing LoRA directly (recommended)**
```python
# Your existing LoRA adapter should work directly
lora_adapter_path = "path/to/your/lora/adapter"
```

**Option B: Merge and save (if needed)**
```python
# If you encounter issues, merge the LoRA with base model
from peft import PeftModel
from transformers import AutoModelForCausalLM

base_model = AutoModelForCausalLM.from_pretrained("ModelSpace/GemmaX2-28-9B-Pretrain")
model = PeftModel.from_pretrained(base_model, "gemmax2_9b_finetuned")
merged_model = model.merge_and_unload()
merged_model.save_pretrained("merged_model_path")
```

### 3. Update Your Code
Replace your existing `PipelinePro` class with `PipelineProVLLM` from the new file.

## Configuration Options

### Memory Optimization
```python
self.model = LLM(
    model=self.base_model,
    enable_lora=True,
    gpu_memory_utilization=0.9,  # Use 90% of GPU memory
    max_model_len=4096,          # Adjust based on your needs
    swap_space=4,                # GB of CPU swap space
)
```

### Multi-GPU Setup
```python
self.model = LLM(
    model=self.base_model,
    enable_lora=True,
    tensor_parallel_size=2,      # Number of GPUs
    pipeline_parallel_size=1,    # Pipeline parallelism
)
```

### Batch Size Optimization
```python
# vLLM can handle larger batches more efficiently
batch_size = min(8, len(batch_slides))  # Increased from 4 to 8
```

## Important Notes

### 1. Quantization
vLLM handles quantization differently than transformers. Instead of runtime quantization:
- Pre-quantize your model using tools like AutoGPTQ or AutoAWQ
- Or use vLLM's built-in quantization support for specific formats

### 2. LoRA Compatibility
- Most LoRA adapters work directly with vLLM
- Ensure your LoRA rank is compatible (max_lora_rank parameter)
- vLLM supports multiple LoRA adapters simultaneously

### 3. Memory Requirements
- vLLM is more memory-efficient but may have different memory patterns
- Adjust `gpu_memory_utilization` based on your GPU memory
- Monitor GPU usage during initial testing

## Migration Steps

### Step 1: Test with Existing Model
1. Install vLLM dependencies
2. Create `pipeline_pro_vllm.py` with the new implementation
3. Test with a small batch to ensure compatibility

### Step 2: Benchmark Performance
```python
import time

# Test current implementation
start_time = time.time()
results_transformers = pipeline_pro.infer_batch(test_batch)
transformers_time = time.time() - start_time

# Test vLLM implementation
start_time = time.time()
results_vllm = pipeline_pro_vllm.infer_batch(test_batch)
vllm_time = time.time() - start_time

print(f"Transformers time: {transformers_time:.2f}s")
print(f"vLLM time: {vllm_time:.2f}s")
print(f"Speedup: {transformers_time / vllm_time:.2f}x")
```

### Step 3: Gradual Rollout
1. Deploy vLLM version alongside existing implementation
2. Route a percentage of traffic to vLLM version
3. Monitor performance and accuracy
4. Gradually increase vLLM usage

## Troubleshooting

### Common Issues

1. **LoRA Loading Errors**
   ```python
   # Ensure LoRA path is correct and accessible
   # Check LoRA rank compatibility
   ```

2. **Memory Issues**
   ```python
   # Reduce gpu_memory_utilization
   # Decrease max_model_len
   # Use smaller batch sizes initially
   ```

3. **Output Format Differences**
   ```python
   # vLLM output format is slightly different
   # Update text extraction logic if needed
   ```

## Expected Performance Gains

Based on your current setup and benchmarks from similar implementations:

- **Single inference**: 2-3x faster
- **Batch inference (4 slides)**: 3-4x faster  
- **Batch inference (8+ slides)**: 4-6x faster
- **Memory usage**: 20-30% reduction
- **Throughput**: 2-4x higher requests per second

## Next Steps

1. Install vLLM and test with the new implementation
2. Run performance benchmarks on your specific hardware
3. Test with your actual LoRA adapter
4. Monitor quality to ensure output consistency
5. Gradually migrate production traffic

## Support and Resources

- [vLLM Documentation](https://docs.vllm.ai/)
- [vLLM LoRA Guide](https://docs.vllm.ai/en/latest/features/lora.html)
- [Performance Tuning Guide](https://docs.vllm.ai/en/latest/performance/optimization.html) 