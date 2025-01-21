import os
from pathlib import Path
from colorthief import ColorThief
from PIL import Image
import colorsys
import shutil
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_dominant_color(image_path):
    """Extract the dominant color from an image and return its HSV values."""
    try:
        color_thief = ColorThief(image_path)
        # Get the dominant color (RGB)
        dominant_color = color_thief.get_color(quality=1)
        # Convert RGB to HSV
        hsv = colorsys.rgb_to_hsv(
            dominant_color[0]/255.0, 
            dominant_color[1]/255.0, 
            dominant_color[2]/255.0
        )
        return hsv
    except Exception as e:
        logger.error(f"Error processing {image_path}: {e}")
        # Return a default value for failed images
        return (0, 0, 0)

def sort_and_rename_images():
    # Get the absolute path to the target directory
    base_dir = Path(__file__).parent.parent
    source_dir = base_dir / 'files' / 'channel_1315381196221321258'
    
    # Create a temporary directory for the sorted files
    temp_dir = base_dir / 'files' / 'sorted_images'
    temp_dir.mkdir(exist_ok=True)
    
    # Get all image files
    image_files = []
    valid_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.webp'}
    
    logger.info("Analyzing images...")
    
    # Collect all images and their dominant colors
    for file in source_dir.iterdir():
        if file.suffix.lower() in valid_extensions:
            try:
                hsv_values = get_dominant_color(str(file))
                image_files.append((file, hsv_values))
            except Exception as e:
                logger.error(f"Could not process {file}: {e}")
    
    # Sort images by hue, then saturation, then value
    image_files.sort(key=lambda x: (x[1][0], x[1][1], x[1][2]))
    
    logger.info("Renaming files...")
    
    # Rename files with new sequential numbers
    for index, (old_path, _) in enumerate(image_files, 1):
        # Create new filename with original extension
        new_filename = f"{index:03d}{old_path.suffix.lower()}"
        new_path = source_dir / new_filename
        
        try:
            # Rename the file
            old_path.rename(new_path)
            logger.info(f"Renamed {old_path.name} to {new_filename}")
        except Exception as e:
            logger.error(f"Error renaming {old_path}: {e}")

if __name__ == "__main__":
    sort_and_rename_images() 