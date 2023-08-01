from globus_compute_sdk import Executor
from globus_compute_sdk import Client
import argparse
from datetime import datetime


def submit_job():
    import subprocess, random
    rand = random.randint(10000, 20000)
    output = subprocess.run(["scontrol", "show", "hostnames"], capture_output=True)
    head_node = None
    if output.returncode == 0:
        head_node = output.stdout.decode().split()[0]
        # return head_node
    else:
        return output.stderr.decode()
    output = subprocess.run(["srun", "torchrun", "--nnodes", "2", "--nproc-per-node", "1",
                             "--rdzv_id", f"{rand}",
                             "--rdzv_backend", "c10d",
                             "--rdzv_endpoint", f"{head_node}.arc-ts.umich.edu:29500",
                             "/home/zyliang/llama-test/llama/example.py",
                             "--ckpt_dir",
                             "/ime/accounts/dinov_root/dinov0/LLaMA/1.0.1/llama/modeltoken/13B",
                             "--tokenizer_path",
                             "/ime/accounts/dinov_root/dinov0/LLaMA/1.0.1/llama/modeltoken/tokenizer.model"],
                            capture_output=True)
    if output.returncode == 0:
        return output.stdout.decode()
    else:
        return output.stderr.decode()


llama13b_endpoint = '94e4a0bc-a631-4624-be82-4de917acc9dd'
gcc = Client()
endpoint_status = gcc.get_endpoint_status(llama13b_endpoint)

if endpoint_status["status"] == "offline":
    raise Exception("Error: Globus endpoint offline!")

# parser = argparse.ArgumentParser()
# parser.add_argument("--prompts", nargs="+")
# args = parser.parse_args()
# prompts = args.prompts
# if prompts is None:
#     prompts = ["The capital of France is"]
# print("Your prompts:")
# for i in range(0, len(prompts)):
#     print(f"prompt {i+1}: {prompts[i]}")

# create the executor
with Executor(endpoint_id=llama13b_endpoint) as gce:
    # submit for execution
    future = gce.submit(submit_job)
    current_time = datetime.now().strftime("%H:%M")
    # future = gce.submit(submit_job, prompts)
    print(f"\nSubmitted the function to Globus endpoint at {current_time}.\n")
    result = future.result()
    print(result)
    current_time = datetime.now().strftime("%H:%M")
    print(f"\nFinished at {current_time}.\n")
    current_time = datetime.now().strftime("%m%d%H%M")
    with open(f"output{current_time}.txt", "w") as file:
        file.write(result)
