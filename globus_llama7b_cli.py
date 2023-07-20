from globus_compute_sdk import Executor
from globus_compute_sdk import Client
# from pprint import pprint
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


llama7b_endpoint = '0b88751e-a0d8-4a2a-9e97-7d2161241510'
gcc = Client()
endpoint_status = gcc.get_endpoint_status(llama7b_endpoint)
# print(endpoint_status)
if endpoint_status["status"] == "offline":
    raise Exception("Error: Globus endpoint offline!")

parser = argparse.ArgumentParser()
parser.add_argument("--prompts", nargs="+")
args = parser.parse_args()
prompts = args.prompts
if prompts is None:
    prompts = ["The capital of France is"]
print("Your prompts:")
for i in range(0, len(prompts)):
    print(f"prompt {i+1}: {prompts[i]}")

# create the executor
with Executor(endpoint_id=llama7b_endpoint) as gce:
    # submit for execution
    future = gce.submit(submit_job, prompts)
    print("\nSubmitted the function to Globus endpoint.\n")
    print(future.result())
