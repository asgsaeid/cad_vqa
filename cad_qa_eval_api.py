import os
import random
import time
import argparse
import json
import base64
import re
from tqdm import tqdm

import pyarrow.parquet as pq
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

import openai
from openai import OpenAI
import anthropic
import google.generativeai as genai

# API Keys
OPENAI_API_KEY = "sk-proj-6m4gseG1912DfA94Z3Q1T3BlbkFJvChdLWnCNO6zHGoCvuVc"
GOOGLE_API_KEY = "AIzaSyAYLnwvq78v0OjZIajUEJpesPgnk7JUXxo"
ANTHROPIC_API_KEY = "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"
DEEPINFRA_API_KEY = "znsllrNNYIzh7XJbBGO64W6zTtvMbWk6"

def create_tiled_image(image_paths, output_path, max_width=362, max_height=256, padding=5):
    images = [Image.open(path) for path in image_paths]

    single_width = (max_width - 3 * padding) // 3
    single_height = (max_height - padding) // 2

    resized_images = [img.resize((single_width, single_height), Image.LANCZOS) for img in images]

    new_img = Image.new('RGB', (max_width, max_height), color='white')

    for i, img in enumerate(resized_images):
        row = i // 3
        col = i % 3
        x = col * (single_width + padding) + padding
        y = row * (single_height + padding) + padding
        new_img.paste(img, (x, y))

    new_img.save(output_path)

def load_and_tile_images(parquet_file, main_image_folder, tiled_images_folder, image_subfolder_names):
    table = pq.read_table(parquet_file)
    df = table.to_pandas()

    os.makedirs(tiled_images_folder, exist_ok=True)

    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing images"):
        id = row['id']
        tiled_image_path = os.path.join(tiled_images_folder, f"{id}_tiled.png")

        if not os.path.exists(tiled_image_path):
            image_paths = []
            for subfolder in image_subfolder_names:
                folder_path = os.path.join(main_image_folder, subfolder, str(id))
                if os.path.exists(folder_path):
                    png_files = [f for f in os.listdir(folder_path) if f.endswith('.png')]
                    if len(png_files) >= 2:
                        selected_images = random.sample(png_files, 2)
                        for img_file in selected_images:
                            image_paths.append(os.path.join(folder_path, img_file))

            if len(image_paths) == 6:
                create_tiled_image(image_paths, tiled_image_path)
            else:
                print(f"Warning: Not enough images found for ID {id}")

        df.at[_, 'tiled_image_path'] = tiled_image_path

    return df


def get_client(model_name):
    if model_name == "gpt-4o":
        openai.api_key = OPENAI_API_KEY
        client = openai
    elif model_name == "gemini-1.5-pro-latest":
        genai.configure(api_key=GOOGLE_API_KEY)
        generation_config = {
            "temperature": 0.0,
            "top_p": 1,
            "max_output_tokens": 4000,
            "response_mime_type": "text/plain",
        }
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]
        client = genai.GenerativeModel(
            model_name=model_name,
            safety_settings=safety_settings,
            generation_config=generation_config,
        )
    elif model_name == "claude-3-5-sonnet-20240620":
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    elif model_name == "meta-llama/Meta-Llama-3.1-405B-Instruct":
        client = OpenAI(api_key=DEEPINFRA_API_KEY, base_url="https://api.deepinfra.com/v1/openai")
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return client


def call_api(model_name, client, instruction, inputs):
    if model_name in ["gpt-4o", "meta-llama/Meta-Llama-3.1-405B-Instruct"]:
        message_text = [{"role": "user", "content": instruction + inputs}]
        completion = client.chat.completions.create(
            model=model_name,
            messages=message_text,
            temperature=0,
            max_tokens=4000,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        result = completion.choices[0].message.content
    elif model_name == "gemini-1.5-pro-latest":
        chat_session = client.start_chat(history=[])
        result = chat_session.send_message(instruction + inputs).text
    elif model_name == "claude-3-5-sonnet-20240620":
        message = client.messages.create(
            model=model_name,
            max_tokens=4000,
            system="",
            messages=[{"role": "user", "content": instruction + inputs}],
            temperature=0.0,
            top_p=1,
        )
        result = message.content[0].text
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return result

import re


def format_question_with_prompt(row):
    initial_prompt = (
        "The following is a multiple choice question about a 3D object. "
        "The accompanying image contains 6 different 2D renders of this 3D object, arranged in two rows with three images each. "
        "In some renders, the object is shown in an assembly where the main object is red. "
        "In others, the same object is shown in gray as an individual part. "
        "Examine the images carefully, think step by step, and then finish your answer "
        "with 'The answer is (X)' where X is the correct letter choice.\n\n"
    )

    question = f"Question: {row['question']}\n"
    question += "Options:\n"

    # Access options from the list stored in 'options' column
    options_list = row['options']
    for i, option in enumerate(options_list):
        option_letter = chr(65 + i)  # Convert to A, B, C, ...

        # Remove any leading letter and dot (e.g., "A. " or "B. ") from the option text
        option_text = re.sub(r'^[A-Z]\.\s*', '', option)

        question += f"{option_letter}. {option_text}\n"

    return initial_prompt + question

def extract_answer(text):
    match = re.search(r"The answer is \(?([A-Z])\)?", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    match = re.search(r"([A-Z]) is the correct answer", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    matches = re.findall(r"([A-Z])\)", text)
    if matches:
        return matches[-1].upper()

    return None


def evaluate_vlm(df, model_name, client):
    correct = 0
    total = 0
    results = []

    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc=f"Evaluating {model_name}"):
        question = format_question_with_prompt(row)
        image_path = row['tiled_image_path']

        with open(image_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

        inputs = f"\n[Image: data:image/png;base64,{encoded_image}]"

        # Sanity check: Print question and options
        # if total < 5:  # Print for the first 5 questions
        #     print(f"\nSanity Check - Question {total + 1}:")
        #     print(question)
        #     print("Image path:", image_path)
        #     print("---")

        response = call_api(model_name, client, question, inputs)
        predicted_answer = extract_answer(response)

        correct_answer = chr(65 + row['answer_index'])

        # Remove the existing letter prefix from each option
        options = {
            chr(65 + i): re.sub(r'^[A-Z]\.\s*', '', option)
            for i, option in enumerate(row['options'])
        }

        is_correct = predicted_answer == correct_answer if predicted_answer else False
        if is_correct:
            correct += 1
        total += 1

        results.append({
            'id': row['id'],
            'question': row['question'],
            'options': options,  # Use the cleaned options dictionary
            'correct_answer': correct_answer,
            'predicted_answer': predicted_answer,
            'is_correct': is_correct,
            'model_response': response
        })

    accuracy = correct / total
    print(f"Model: {model_name}")
    print(f"Accuracy: {accuracy:.2f}")

    return accuracy, results


def main(args):
    print("Loading and tiling images...")
    df = load_and_tile_images(args.parquet_file, args.main_image_folder, args.tiled_images_folder,
                              args.image_subfolder_names)

    main_results_dir = "evaluation_results_vlm_v3"
    os.makedirs(main_results_dir, exist_ok=True)

    print(f"\nEvaluating {args.model_name}...")
    client = get_client(args.model_name)
    accuracy, results = evaluate_vlm(df, args.model_name, client)

    # Save detailed results to JSON with accuracy at the top
    results_file = os.path.join(main_results_dir, f"{args.model_name}_detailed_results.json")
    with open(results_file, 'w') as f:
        json.dump({'accuracy': accuracy, 'results': results}, f, indent=2)
    print(f"Detailed results saved to {results_file}")

    print(f"\nFinal Results for {args.model_name}:")
    print(f"Accuracy: {accuracy:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate VLMs on 3D CAD data")
    parser.add_argument("--parquet_file", type=str, required=True,
                        help="Path to the Parquet file containing the dataset")
    parser.add_argument("--main_image_folder", type=str, required=True,
                        help="Path to the main folder containing image subfolders")
    parser.add_argument("--tiled_images_folder", type=str, required=True,
                        help="Path to the folder where tiled images will be saved")
    parser.add_argument("--image_subfolder_names", nargs='+',
                        default=["single", "transparent_zoomed_loose", "transparent_zoomed_tight"],
                        help="Names of image subfolders (default: %(default)s)")
    parser.add_argument("--model_name", type=str, required=True,
                        choices=["gpt-4o", "gemini-1.5-pro-latest", "meta-llama/Meta-Llama-3.1-405B-Instruct",
                                 "claude-3-5-sonnet-20240620"],
                        help="Name of the model to use")
    args = parser.parse_args()

    main(args)
