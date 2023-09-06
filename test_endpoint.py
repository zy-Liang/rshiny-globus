from globus_compute_sdk import Executor
from globus_compute_sdk import Client

llama7b_endpoint = '0cb81fdc-582e-453d-8db5-8d6c31150b3b'


def submit_job():
    import subprocess
    output = subprocess.run(["torchrun", "--nproc_per_node", "1", 
                             "/home/tingtind/llama/example_text_completion.py",
                             "--ckpt_dir", "/nfs/turbo/umms-dinov/LLaMA/2.0.0/llama/modeltoken/llama-2-7b",
                             "--tokenizer_path", "/nfs/turbo/umms-dinov/LLaMA/2.0.0/llama/modeltoken/tokenizer.model"],
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


def run_llama7b():
    with Executor(endpoint_id=llama7b_endpoint) as gce:
        # submit for execution
        future = gce.submit(submit_job)
        print("\nSubmitted the function to Globus endpoint.\n")
        result = future.result()
        print("Generation finished!")
        return result

if __name__ == "__main__":
    try:
        res = run_llama7b()
        print(res)
    except Exception as e:
        print(e)