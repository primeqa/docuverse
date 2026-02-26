from vllm import LLM, SamplingParams, TokensPrompt
from transformers import AutoTokenizer
import numpy as np
import time


prefill_tokens=25000
generate_tokens=256

sampling_params_prefill = SamplingParams(temperature=0.8, top_p=0.95,max_tokens=1)
sampling_params_generate = SamplingParams(temperature=0.8, top_p=0.95,max_tokens=generate_tokens)

# model_name = "ibm-granite/granite-4.0-tiny-preview"
model_name = "ibm-granite/granite-3.3-8b-instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
vocab_size = tokenizer.vocab_size


def gen_rnd_tokens(shape):
    # the vocabulary size is a global variable
    return np.random.randint(0, vocab_size, size=shape).tolist()


# Create a text prompt instead of random tokens for prefill
prefill_text = "This is a test prompt. " * (prefill_tokens // 6)  # Approximate tokens
prompts_token_ids = tokenizer.encode(prefill_text, add_special_tokens=True)[:prefill_tokens]

llm = LLM(model=model_name, max_model_len=50000)

print("warmup")
llm.generate(prompts=["Hello, how are you?"], sampling_params=sampling_params_prefill)

print("go for it")
t0 = time.perf_counter()
outputs = llm.generate(prompts=[TokensPrompt(prompt_token_ids=prompts_token_ids)], sampling_params=sampling_params_prefill)
t1 = time.perf_counter()
outputs = llm.generate(prompts=[TokensPrompt(prompt_token_ids=prompts_token_ids)], sampling_params=sampling_params_generate)
t2 = time.perf_counter()

actual_prefill_tokens = len(prompts_token_ids)
gen_throughput = generate_tokens/(t2-t1)
prefill_throughput = actual_prefill_tokens/(t1-t0)

print("%s prefill %f tok/sec (%d tokens), gen %f tok/sec (%d tokens), ratio = %f" % (model_name, prefill_throughput, actual_prefill_tokens, gen_throughput, generate_tokens, prefill_throughput/gen_throughput ))
