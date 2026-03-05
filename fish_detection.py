import json
import time
from datetime import datetime
import base64
import os
from io import BytesIO
from typing import List, Dict, Any, Tuple
from PIL import Image, ImageDraw, ImageFont
import concurrent.futures
import logging
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
import sys

from dotenv import load_dotenv

# Assuming the google-genai library is installed
# pip install google-genai
from google import genai
from google.genai import types

# Setup logging to files and console
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# Handler for run.log (all info and above)
run_handler = logging.FileHandler("run.log")
run_handler.setLevel(logging.INFO)
run_handler.setFormatter(formatter)
logger.addHandler(run_handler)

# Handler for err.log (warnings and errors)
err_handler = logging.FileHandler("err.log")
err_handler.setLevel(logging.WARNING)
err_handler.setFormatter(formatter)
logger.addHandler(err_handler)

# Handler for console
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

def get_gemini_response(client: genai.Client, model_id: str, contents: List[Any], config: types.GenerateContentConfig = None):
    """
    Handles API calls to Gemini with exponential backoff as per requirements.
    """
    for delay in [1, 2, 4, 8, 16]:
        try:
            response = client.models.generate_content(
                model=model_id,
                contents=contents,
                config=config
            )
            return response
        except Exception as e:
            logging.warning(f"API call failed (possibly rate limited or unavailable). Retrying in {delay} seconds... Error: {e}")
            time.sleep(delay)
    
    # Final attempt or raise error
    logging.error("Max retries reached. Making final attempt.")
    return client.models.generate_content(model=model_id, contents=contents, config=config)

def process_single_image(filename: str, client: genai.Client, model_id: str, prompt: str) -> Tuple[str, dict]:
    """
    Process a single image from end-to-end (read local, identify, detect, draw, save).
    """
    logging.info(f"[{filename}] Starting processing...")
    local_image_path = os.path.join("./images", filename)
    
    try:
        # Read the local image into memory
        with open(local_image_path, "rb") as f:
            image_bytes = f.read()
        
        # Determine basic mime type
        mime_type = "image/jpeg" if filename.lower().endswith(('.jpg', '.jpeg')) else "image/png"
        
        # Prepare image part for Gemini API
        image_part = types.Part.from_bytes(data=image_bytes, mime_type=mime_type)
    except Exception as e:
        logging.error(f"[{filename}] Failed to read local image {local_image_path}: {e}")
        return filename, None
        
    # Configure for structured JSON output matching the new prompt
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema={
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "label": {
                        "type": "STRING"
                        # Removed the hardcoded 'enum' list so the model relies entirely 
                        # on the categories you provide in prompt.txt!
                    },
                    "box_2d": {
                        "type": "ARRAY",
                        "items": {"type": "NUMBER"},
                        "minItems": 4,
                        "maxItems": 4
                    }
                },
                "required": ["label", "box_2d"]
            }
        }
    )

    logging.info(f"[{filename}] Requesting detection and classification from model...")
    response = get_gemini_response(
        client, 
        model_id, 
        [prompt, image_part], 
        config
    )

    image_detections = []
    try:
        detections = json.loads(response.text)
        logging.info(f"[{filename}] Found {len(detections)} detections.")
        for item in detections:
            label = item.get("label", "Unknown")
            box = item.get("box_2d", [])
            if len(box) == 4:
                # Create the tuple (species, box) to maintain compatibility with drawing/saving logic
                image_detections.append((label, box))
    except (json.JSONDecodeError, AttributeError) as e:
        logging.error(f"[{filename}] Failed to parse response: {e}")

    # Zip into a dictionary for this specific image
    image_result = {
        "image_file": filename,
        "detections": [dict([item]) for item in image_detections]
    }
    
    # Generate timestamp and base names once to ensure matching filenames
    base_name, ext = os.path.splitext(filename)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    
    # Save JSON output to ./labels/ directory with timestamp
    json_filename = f"{base_name}.{timestamp}.json"
    json_path = os.path.join("./labels", json_filename)
    with open(json_path, "w") as f:
        json.dump(image_result, f, indent=2)

    # STAGE 3: Draw boxes and class names on the image
    try:
        img = Image.open(local_image_path)
        draw = ImageDraw.Draw(img)
        width, height = img.size
        
        # Load a larger font (approx 3x standard default size)
        try:
            # Modern Pillow supports size in load_default
            font = ImageFont.load_default(size=45)
        except TypeError:
            # Fallback for older Pillow versions
            try:
                font = ImageFont.truetype("arial.ttf", 45)
            except IOError:
                font = ImageFont.load_default()
        
        for species, box in image_detections:
            ymin, xmin, ymax, xmax = box
            
            # Convert normalized coordinates (0-1000) to absolute pixels
            left = (xmin / 1000.0) * width
            top = (ymin / 1000.0) * height
            right = (xmax / 1000.0) * width
            bottom = (ymax / 1000.0) * height
            
            # Draw bounding box
            draw.rectangle([left, top, right, bottom], outline="red", width=3)
            
            # Draw label (species name) with increased offset to accommodate larger font
            draw.text((left, max(0, top - 45)), species, fill="red", font=font)
            
        # Save the boxed image with matching timestamp
        boxed_filename = f"{base_name}.{timestamp}.boxed{ext}"
        boxed_path = os.path.join("./labeled", boxed_filename)
        img.save(boxed_path)
        logging.info(f"[{filename}] Saved boxed image to {boxed_filename} and JSON labels to {json_filename}.")
    except Exception as e:
        logging.error(f"[{filename}] Failed to draw and save image: {e}")

    return filename, image_result

def process_fish_images():
    start_time = time.time()
    start_timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    
    # Setup output directories
    os.makedirs("./labels", exist_ok=True)
    os.makedirs("./labeled", exist_ok=True)

    # Configuration
    load_dotenv()
    
    # Vertex AI requires a Project ID and Location rather than just an API key.
    # Vertex uses Application Default Credentials (ADC) or a Service Account JSON.
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT") 
    location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
    model_id = os.getenv("MODEL")
    
    # Initialize the client specifically for Vertex AI
    client = genai.Client(
        vertexai=True, 
        project=project_id, 
        location=location
    )

    # Read the prompt from prompt.txt
    prompt_file = "./prompt.txt"
    if not os.path.exists(prompt_file):
        logging.error(f"Prompt file {prompt_file} does not exist. Please create it.")
        return json.dumps({})
        
    try:
        with open(prompt_file, "r") as f:
            prompt = f.read().strip()
    except Exception as e:
        logging.error(f"Failed to read {prompt_file}: {e}")
        return json.dumps({})
        
    if not prompt:
        logging.error(f"Prompt file {prompt_file} is empty.")
        return json.dumps({})
    
    image_dir = "./images"
    if not os.path.exists(image_dir):
        logging.error(f"Directory {image_dir} does not exist. Please create it and add images.")
        return json.dumps({})
        
    # Get all valid images from the local folder
    filenames = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not filenames:
        logging.warning(f"No PNG or JPEG images found in {image_dir}.")
        return json.dumps({})

    results = {}

    # Run tasks concurrently using ThreadPoolExecutor
    # With Vertex AI, you usually have higher quota limits, but 2-5 is still safe
    max_workers = 10
    logging.info(f"Starting batch processing of {len(filenames)} images with {max_workers} concurrent workers on Vertex AI...")
    
    # Redirect logging through tqdm to prevent the progress bar from breaking/jumping
    with logging_redirect_tqdm():
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks to the executor
            future_to_filename = {
                executor.submit(process_single_image, fname, client, model_id, prompt): fname 
                for fname in filenames
            }
            
            # Process results as they complete with tqdm progress bar
            for future in tqdm(concurrent.futures.as_completed(future_to_filename), total=len(filenames), desc="Processing Images"):
                fname = future_to_filename[future]
                try:
                    processed_fname, result = future.result()
                    if result:
                        results[processed_fname] = result
                except Exception as exc:
                    logging.error(f"[{fname}] Generated an exception: {exc}")

    end_time = time.time()
    elapsed_time = end_time - start_time
    runtime_msg = f"Finished processing {len(filenames)} images in {elapsed_time:.2f} seconds."
    logging.info(runtime_msg)
    
    # Log runtime to runtimes.log
    with open("runtimes.log", "a") as f:
        f.write(f"{start_timestamp} - {runtime_msg}\n")

    # Final Output
    return json.dumps(results, indent=2)

if __name__ == "__main__":
    # Execute the flow without printing the resulting JSON object
    process_fish_images()