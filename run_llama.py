# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import fire

from llama import Llama


def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 128,
    max_gen_len: int = 64,
    max_batch_size: int = 4,
    prompts:list = ["The capital of France is"]
):
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    results = generator.text_completion(
        prompts,
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )
    print("\n==================================\n")
    for prompt, result in zip(prompts, results):
        print("[User prompt]")
        print(prompt)
        print("\n[Model output]")
        print(f"{result['generation']}")
        print("\n==================================\n")


if __name__ == "__main__":
    try:
        fire.Fire(main)
    except Exception as error:
        print(error)
