from globus_compute_sdk import Executor
from globus_compute_sdk import Client
import argparse
from datetime import datetime
import textwrap


def gl_job(prompt_list_str: str):
    import subprocess
    output = subprocess.run(["torchrun", "--nproc_per_node", "1", 
                             "/home/zyliang/llama-test/llama/benchmark.py",
                             "--prompts", prompt_list_str,
                             "--ckpt_dir", "/nfs/turbo/umms-dinov/LLaMA/1.0.1/llama/modeltoken/7B",
                             "--tokenizer_path", "/nfs/turbo/umms-dinov/LLaMA/1.0.1/llama/modeltoken/tokenizer.model"],
                            capture_output=True)
    if output.returncode == 0:
        return output.stdout.decode()
    else:
        return output.stderr.decode()


def armis2_job(prompt_list_str: str):
    import subprocess
    # output = subprocess.run(["torchrun", "--nproc_per_node", "1", 
    #                          "/home/zyliang/llama-test/llama/run_llama.py",
    #                          "--prompts", prompt_list_str,
    #                          "--ckpt_dir", "/nfs/turbo/umms-dinov2/LLaMA/1.0.1/llama/modeltoken/7B",
    #                          "--tokenizer_path", "/nfs/turbo/umms-dinov2/LLaMA/1.0.1/llama/modeltoken/tokenizer.model"],
    #                         capture_output=True)
    output = subprocess.run(["echo", "hello"], capture_output=True)
    if output.returncode == 0:
        return output.stdout.decode()
    else:
        return output.stderr.decode()


endpoint_gl = 'c7f61570-3ef3-4161-a1ba-c4b9d11b1edf'
endpoint_armis2 = '3dc2a8d4-78bf-4ca4-bb72-a9769f74e46b'

parser = argparse.ArgumentParser()
parser.add_argument("-p","--prompts", nargs="+")
parser.add_argument("-c", "--cluster", nargs="?", choices=["gl", "armis2"], type=str)
args = parser.parse_args()

cluster = args.cluster
if cluster is None:
    cluster = "gl"

# Check endpoint status
gcc = Client()
endpoint_status = None
if cluster == "gl":
    endpoint_status = gcc.get_endpoint_status(endpoint_gl)
elif cluster == "armis2":
    endpoint_status = gcc.get_endpoint_status(endpoint_armis2)
if endpoint_status["status"] != "online":
    raise Exception("Error: Globus endpoint offline!")

# Print user prompts
prompts = args.prompts
if prompts is None:
    prompts = ["The capital of France is"]
print("\nYour prompts:\n")
for index, prompt in enumerate(prompts):
    print(f"[prompt {index+1}] {prompt}\n")

prompt_list_str = "["
for prompt in prompts:
    prompt_list_str += f"\"{prompt}\","
prompt_list_str += "]"

result = None
if cluster == "gl":
# create the executor
    with Executor(endpoint_id=endpoint_gl) as gce:
        future = gce.submit(gl_job, prompt_list_str)
        current_time = datetime.now().strftime("%H:%M")
        print(f"\nSubmitted to Great Lakes endpoint at {current_time}.\n")
        result = future.result()
elif cluster == "armis2":
    with Executor(endpoint_id=endpoint_armis2) as gce:
        future = gce.submit(armis2_job, prompt_list_str)
        current_time = datetime.now().strftime("%H:%M")
        print(f"\nSubmitted to Armis2 endpoint at {current_time}.\n")
        result = future.result()

# print(result)
# current_time = datetime.now().strftime("%m%d%H%M")
# with open(f"output{current_time}.txt", "w") as file:
#     file.write(textwrap.fill(result, width=80, replace_whitespace=False))

current_time = datetime.now().strftime("%H:%M")
print(f"\nFinished at {current_time}.\n")

identifier = "The correct answer is "
position = result.find(identifier)
print(result[position+len(identifier)])
