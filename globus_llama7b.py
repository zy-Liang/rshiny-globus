from globus_compute_sdk import Executor
from globus_compute_sdk import Client


llama7b_endpoint = '0b88751e-a0d8-4a2a-9e97-7d2161241510'


def submit_job(prompts:list):
    import subprocess
    # process prompt list
    prompt_list_str = "["
    for prompt in prompts:
        prompt_list_str += f"\"{prompt}\","
    prompt_list_str += "]"
    # run llama with torch
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


def endpoint_connection():
    # Check the status of the endpoint
    gcc = Client()
    endpoint_status = gcc.get_endpoint_status(llama7b_endpoint)
    return endpoint_status["status"] == "online"


def run_llama7b(prompts: list):
    # print prompts
    print("\nPrompt(s):")
    for count, prompt in enumerate(prompts):
        print(f"Prompt {count+1}: {prompt}")
    # create the executor
    with Executor(endpoint_id=llama7b_endpoint) as gce:
        # submit for execution
        future = gce.submit(submit_job, prompts)
        print("\nSubmitted the function to Globus endpoint.\n")
        result = future.result()
        print("Generation finished!")
        return result
