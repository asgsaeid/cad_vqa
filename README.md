# CAD-VQA: Computer-Aided Design Visual Question Answering Dataset for Evaluating Vision Languagve Models

This repository contains the CAD-VQA dataset and evaluation code introduced in the paper "How to Determine the Preferred Image Distribution of a Black-Box Vision-Language Model?".

## Dataset

CAD-VQA is a novel dataset designed to evaluate Vision-Language Models' understanding of 3D mechanical parts in Computer-Aided Design (CAD) contexts. The dataset consists of:

- 17 3D mechanical parts
- 85 multiple-choice questions covering aspects such as part names, geometrical features, assembly features, and functionality
- High-quality rendered images of parts from multiple perspectives

## Contents

## Contents

- `cadvlm_vqa.parquet`: Questions and answers
- `cad_qa_eval_api.py`: Code for evaluating VLM performance on CAD-VQA through API
- `tiled_images/`: Tiled images we used to evaluate models. We create a tiled image containing 6 images randomly selected from the 3 distributions (2 from each)
- `images.zip`: Rendered images of 3D parts. If you want to try images other than our tiled ones, download the original images here: [CAD-VQA Images](https://drive.google.com/file/d/1w77BSlQffFdmwmF7ArL1J9gbCV7ghMrT/view?usp=share_link). If the `tiled_images` folder is missing, it will be automatically created using the images inside the `images.zip` file you download.


## Requirements

- Dependencies listed in `requirements.txt`

## Installation

1. Clone this repository
2. Install required dependencies:
pip install -r requirements.txt

## Usage

To evaluate a VLM on the CAD-VQA dataset:

`python cad_qa_eval_api.py --parquet_file <path_to_parquet_file> 
--main_image_folder <path_to_main_image_folder> 
--tiled_images_folder <path_to_tiled_images_folder> 
--image_subfolder_names single transparent_zoomed_loose transparent_zoomed_tight 
--model_name <model_name>`

Replace `<model_name>` with one of the following:
- "gpt-4o"
- "gemini-1.5-pro-latest"
- "meta-llama/Meta-Llama-3.1-405B-Instruct"
- "claude-3-5-sonnet-20240620"

## API Keys

Before running the evaluation, make sure to set the following API keys in the script:

- `OPENAI_API_KEY`
- `GOOGLE_API_KEY`
- `ANTHROPIC_API_KEY`
- `DEEPINFRA_API_KEY`

## Evaluation Process

1. The script loads and tiles images from the specified folders.
2. It then evaluates the chosen VLM on the dataset.
3. Results are saved in the `evaluation_results_vlm_v3` directory.

## Code Structure

The main evaluation script `evaluate_vlm.py` contains the following key functions:

- `create_tiled_image()`: Creates a tiled image from multiple input images.
- `load_and_tile_images()`: Loads images from the dataset and creates tiled versions.
- `get_client()`: Initializes the appropriate client for the chosen VLM.
- `call_api()`: Makes API calls to the selected VLM.
- `format_question_with_prompt()`: Formats the question with a prompt for the VLM.
- `extract_answer()`: Extracts the predicted answer from the VLM's response.
- `evaluate_vlm()`: Evaluates the VLM's performance on the dataset.

## Baseline Results

We provide baseline results for state-of-the-art VLMs on CAD-VQA:

| Model | Accuracy |
|-------|----------|
| Claude-3.5-Sonnet | 61% |
| GPT-4o | 54% |
| Gemini-1.5-Pro | 54% |

## Citation

If you use the CAD-VQA dataset in your research, please cite our paper:

```bibtex
@inproceedings{anonymous2024cadvqa,
  title={How to Determine the Preferred Image Distribution of a Black-Box Vision-Language Model?},
  author={Anonymous},
  booktitle={38th Conference on Neural Information Processing Systems (NeurIPS 2024)},
  year={2024}
}




