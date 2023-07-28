from globus_compute_sdk import Executor
from globus_compute_sdk import Client
import argparse
from datetime import datetime


def gl_job(prompts:list):
    import subprocess
    # run llama with torch
    prompt_list_str = "["
    for prompt in prompts:
        prompt_list_str += f"\"{prompt}\","
    prompt_list_str += "]"
    output = subprocess.run(["torchrun", "--nproc_per_node", "1", 
                             "/home/zyliang/llama-test/llama/run_llama.py",
                             "--prompts", prompt_list_str,
                             "--ckpt_dir", "/nfs/turbo/umms-dinov/LLaMA/1.0.1/llama/modeltoken/7B",
                             "--tokenizer_path", "/nfs/turbo/umms-dinov/LLaMA/1.0.1/llama/modeltoken/tokenizer.model"],
                            capture_output=True)
    if output.returncode == 0:
        return output.stdout.decode()
    else:
        return output.stderr.decode()


def armis2_job(prompts:list):
    import subprocess
    # run llama with torch
    prompt_list_str = "["
    for prompt in prompts:
        prompt_list_str += f"\"{prompt}\","
    prompt_list_str += "]"
    output = subprocess.run(["torchrun", "--nproc_per_node", "1", 
                             "/home/zyliang/llama-test/llama/run_llama.py",
                             "--prompts", prompt_list_str,
                             "--ckpt_dir", "/nfs/turbo/umms-dinov2/LLaMA/1.0.1/llama/modeltoken/7B",
                             "--tokenizer_path", "/nfs/turbo/umms-dinov2/LLaMA/1.0.1/llama/modeltoken/tokenizer.model"],
                            capture_output=True)
    if output.returncode == 0:
        return output.stdout.decode()
    else:
        return output.stderr.decode()


endpoint_gl = 'c7f61570-3ef3-4161-a1ba-c4b9d11b1edf'
endpoint_armis2 = '3dc2a8d4-78bf-4ca4-bb72-a9769f74e46b'

parser = argparse.ArgumentParser()
parser.add_argument("--prompts", nargs="+")
parser.add_argument('--cluster', nargs='?', choices=["gl", "armis2"], const="gl", type=str)
args = parser.parse_args()

cluster = args.cluster

# Check endpoint status
gcc = Client()
endpoint_status = None
if cluster == "gl":
    endpoint_status = gcc.get_endpoint_status(endpoint_gl)
elif cluster == "armis2":
    endpoint_status = gcc.get_endpoint_status(endpoint_armis2)
if endpoint_status["status"] == "offline":
    raise Exception("Error: Globus endpoint offline!")

# Print user prompts
prompts = args.prompts
if prompts is None:
    prompts = ["The capital of France is"]
print("Your prompts:")
for index, prompt in enumerate(prompts):
    print(f"prompt {index+1}: {prompt}")

future = None
if cluster == "gl":
# create the executor
    with Executor(endpoint_id=endpoint_gl) as gce:
        future = gce.submit(gl_job, prompts)
        current_time = datetime.now().strftime("%H:%M")
        print(f"\nSubmitted to Great Lakes endpoint at {current_time}.\n")
elif cluster == "armis2":
    with Executor(endpoint_id=endpoint_gl) as gce:
        future = gce.submit(armis2_job, prompts)
        current_time = datetime.now().strftime("%H:%M")
        print(f"\nSubmitted to Armis2 endpoint at {current_time}.\n")

print(future.result())
current_time = datetime.now().strftime("%H:%M")
print(f"\nFinished at {current_time}.\n")
