from globus_compute_sdk import Executor
from pprint import pprint
import argparse


# First, define the function ...
def submit_job(prompts:list):
    import subprocess
    # run llama with torch
    prompt_list_str = "["
    for prompt in prompts:
        prompt_list_str += f"\"{prompt}\","
    prompt_list_str += "]"
    output = subprocess.run(["torchrun", "--nproc_per_node", "1", 
                             "/home/zyliang/llama-test/llama/run.py",
                             "--prompts", prompt_list_str,
                             "--ckpt_dir", "/nfs/turbo/umms-dinov/LLaMA/1.0.1/llama/modeltoken/7B",
                             "--tokenizer_path", "/nfs/turbo/umms-dinov/LLaMA/1.0.1/llama/modeltoken/tokenizer.model"],
                            capture_output=True)
    if output.returncode == 0:
        return output.stdout.decode()
    else:
        return output.stderr.decode()


def run_llama7b(prompts: list):
    print("\nPrompt(s):")
    for count, prompt in enumerate(prompts):
        print(f"Prompt {count+1}: {prompt}")
    test_endpoint_id = 'b9d9099c-4aed-499c-a020-743041a15521'
    with Executor(endpoint_id=test_endpoint_id) as gce:
        # ... then submit for execution, ...
        future = gce.submit(submit_job, prompts)
        print("\nSubmitted the function to Globus endpoint.\n")
        # ... and finally, wait for the result
        result = future.result()
        print("Generation finished!")
        # result = result.split("\n")
        return result
