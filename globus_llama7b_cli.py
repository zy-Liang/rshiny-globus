from globus_compute_sdk import Executor
from globus_compute_sdk import Client

llama7b_endpoint = open("endpoint_id_llama7b.txt").read().strip()


def submit_job(prompts:list):
    import subprocess, pathlib, json
    home = pathlib.Path.home()
    run_llama = home / "llama" / "run_llama.py"
    # process prompt list
    prompt_list_str = "["
    for prompt in prompts:
        prompt_list_str += f"\"{prompt}\","
    prompt_list_str += "]"
    # run llama with torch
    print(f"Prompt list: {prompt_list_str}")
    output = subprocess.run(["torchrun", "--nproc_per_node", "1", 
                             str(run_llama),
                             "--prompts", prompt_list_str,
                             "--ckpt_dir", "/nfs/turbo/umms-dinov/LLaMA/2.0.0/llama/modeltoken/llama-2-7b",
                             "--tokenizer_path", "/nfs/turbo/umms-dinov/LLaMA/2.0.0/llama/modeltoken/tokenizer.model"],
                            capture_output=True)
    if output.returncode == 0:
        res = output.stdout.decode().split("\n")[4]
        res = "{'status': 'ok', " + res[1:]
        return res
    else:
        err = output.stderr.decode()
        dic = {"status": "error", "results": err}
        return str(dic)


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

if __name__ == "__main__":
    try:
        res = run_llama7b([
            "The capital of France is",
            "1 plus 1 is equal to"
            ])
        print(res)
    except Exception as e:
        print(e)
