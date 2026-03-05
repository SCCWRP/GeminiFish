import os
import json
import logging
import concurrent.futures
from PIL import Image, ImageDraw, ImageFont

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def draw_labels_on_image(json_filename: str, font: ImageFont.ImageFont):
    """
    Reads a single JSON label file, loads the corresponding image,
    draws the bounding boxes, and saves the boxed image.
    """
    json_path = os.path.join("./labels", json_filename)

    try:
        # Load the detection data FIRST so we can read the true image filename
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Get the actual image filename from the JSON payload (handles .jpg, .png, etc.)
        image_filename = data.get("image_file")
        if not image_filename:
            logging.error(f"[{json_filename}] Missing 'image_file' key in JSON. Skipping.")
            return

        image_path = os.path.join("./images", image_filename)

        if not os.path.exists(image_path):
            logging.error(f"[{image_filename}] Image not found at {image_path}. Skipping.")
            return

        # Figure out the base name, extension, and extract the timestamp from the JSON filename
        base_name, ext = os.path.splitext(image_filename)
        # Assuming json_filename format is: {base_name}.{timestamp}.json
        timestamp = json_filename.replace(f"{base_name}.", "").replace(".json", "")

        # Create the new matching boxed filename
        boxed_filename = f"{base_name}.{timestamp}.boxed{ext}"
        labeled_path = os.path.join("./labeled", boxed_filename)
        
        detections = data.get("detections", [])
        
        # Load the image
        img = Image.open(image_path)
        draw = ImageDraw.Draw(img)
        width, height = img.size
        
        # Parse detections and draw
        for detection_dict in detections:
            for species, box in detection_dict.items():
                # Reverted back to the correct Gemini standard [ymin, xmin, ymax, xmax]
                ymin, xmin, ymax, xmax = box
                
                # Convert normalized coordinates (0-1000) to absolute pixels
                left = (xmin / 1000.0) * width
                top = (ymin / 1000.0) * height
                right = (xmax / 1000.0) * width
                bottom = (ymax / 1000.0) * height
                
                # Draw bounding box
                draw.rectangle([left, top, right, bottom], outline="red", width=3)
                
                # Draw label (species name) with increased offset
                draw.text((left, max(0, top - 45)), species, fill="red", font=font)
                
        # Save the labeled image
        img.save(labeled_path)
        logging.info(f"[{image_filename}] Successfully saved boxed image to {boxed_filename}.")
    except Exception as e:
        logging.error(f"[{json_filename}] Failed to draw labels: {e}")

def main():
    # Setup output directory
    os.makedirs("./labeled", exist_ok=True)

    # Validate input directories exist
    if not os.path.exists("./labels") or not os.path.exists("./images"):
        logging.error("Directories ./labels or ./images do not exist. Please run the detection script first.")
        return

    # Load the font ONCE in the main thread to ensure thread-safety and avoid I/O overhead
    try:
        # Modern Pillow supports size in load_default
        global_font = ImageFont.load_default(size=45)
    except TypeError:
        # Fallback for older Pillow versions
        try:
            global_font = ImageFont.truetype("arial.ttf", 45)
        except IOError:
            global_font = ImageFont.load_default()

    # Get all JSON files in the labels directory
    all_json_files = [f for f in os.listdir("./labels") if f.endswith(".json")]

    if not all_json_files:
        logging.warning("No JSON label files found in ./labels directory.")
        return

    # Group by base name and find the latest timestamp
    latest_files_map = {}
    for f in all_json_files:
        parts = f.rsplit('.', 2)
        # Check if it matches the base_name.timestamp.json format
        if len(parts) == 3 and parts[2] == 'json' and parts[1].isdigit():
            base_name = parts[0]
            timestamp = parts[1]
            # Keep the one with the highest timestamp (latest)
            if base_name not in latest_files_map or timestamp > latest_files_map[base_name][1]:
                latest_files_map[base_name] = (f, timestamp)
        else:
            # Fallback for files without a proper timestamp
            latest_files_map[f] = (f, "")

    json_files = [file_info[0] for file_info in latest_files_map.values()]

    logging.info(f"Found {len(all_json_files)} total label files. Filtered down to {len(json_files)} latest files. Starting drawing process...")

    # Run tasks concurrently using ThreadPoolExecutor for speed
    max_workers = 5
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks to the executor, passing the pre-loaded global_font
        futures = [
            executor.submit(draw_labels_on_image, jf, global_font) 
            for jf in json_files
        ]
        
        # Wait for all futures to complete
        concurrent.futures.wait(futures)

    logging.info("Finished drawing all labels.")

if __name__ == "__main__":
    main()