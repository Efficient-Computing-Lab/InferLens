## Description
InferLens is an LLM deep analysis toolbox that provides a unified interface able to measure and present various statistics and metrics during LLM inference in real time.

Users can explore how the layers of the selected LLM are activated during each token generation, the resources it consumes, the entropy of the selected token and more.

There is also an option for conversation history and prompt based continual learning (PCL) to explore more capabilities provided by modern LLMs.

The default LLM selected in Llama3 but it supports other huggingface LLMs as well, including Phi4, Nanbeige4 and Qwen2.5. 

## Required configuration

### 3B model license
[https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct]

### 8B model license
[https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct]

### TOKEN
* Get token from https://huggingface.co/settings/tokens.
* Create file src/hugface_token.py.
* Add TOKEN="{Your token}" in it.
* Save.

## Requirements
* Python 3.12+
* GPU that supports CUDA 12.8 if run in CUDA mode
* Access to port 5080 or another port that will host the GUI

## Installation

### Basic pipenv
```python 
python3.12 -m pip install pipenv
python3.12 -m pipenv install
```

### Installation for CUDA
```python 
python3.12 -m pipenv run pip install -r requirements-torch.txt
```

## Run

### Fast run (3B)
`python main.py --port 5080 --model_id meta-llama/Llama-3.2-3B-Instruct`

### Full run (8B)
`python main.py --port 5080`


## GUI
After succesfully running GUI will be available at `http://localhost:5080`