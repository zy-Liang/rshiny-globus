# LLAMA Text Generation
## Overview
This repository provides a text generation script, run_llama.py, leveraging the Llama model. The script allows users to generate text completions based on input prompts. Additionally, it includes a Shiny web application (app.R) for a user-friendly interface to interact with the model. The text generation process is orchestrated by a Python script (globus_llama7b.py) that interfaces with the Globus Compute SDK for remote execution.

## HPC Setup

- Login to HPC
- Clone the LLaMA repository to the home directory
- Clone this repository
- Create an endpoint repo, install Globus dependencies
- Copy `run_llama.py` to the LLaMA repo
- Configure Globus endpoint(s)
- From the config folder of this repo, copy the config file to the Globus config directory

## Web Server Setup

In this repository,
- create a file `endpoint_id_llama7b.txt`, add the endpoint id to the file
- Install python requirements in the file `py_requirements.txt`
- Install R requirements

## File Explanation
### run_llama.py
This Python script serves as the entry point for text generation using the Llama model. It utilizes the fire library for command-line interface handling. Users can specify parameters such as checkpoint directory, tokenizer path, temperature, top_p, maximum sequence length, maximum generation length, and more. The script then utilizes the Llama model to generate text completions based on the provided prompts.

### app.R
This R script defines a Shiny web application for a user-friendly interface to interact with the Llama model. Users can choose a model, input prompts, and receive text generation results. The application leverages the shiny, shinybusy, and reticulate libraries for building an interactive and responsive user interface.

### globus_llama7b.py
This Python script interfaces with the Globus Compute SDK for remote execution of the Llama model. It includes functions to check the status of the Globus endpoint, submit jobs for text generation, and handle file downloads. The script relies on the subprocess module to run the Llama model with Torch on the remote endpoint.