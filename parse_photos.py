import argparse
import logging
import os
from pathlib import Path
from typing import Tuple, Optional, List, Iterator
from PIL import Image
import sys
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from itertools import islice
import pillow_heif  # Add HEIF support

# Register HEIF opener with Pillow
pillow_heif.register_heif_opener()

def setup_logging() -> None:
    """Configure logging with formatting and level."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def parse_arguments() -> Tuple[Path, Path, bool]:
    """
    Parse command line arguments for input and output directories.
    
    Returns:
        Tuple[Path, Path, bool]: Input directory path, output directory path, and batch-only flag
    """
    parser = argparse.ArgumentParser(description='Process images into jpg batches')
    parser.add_argument('input_dir', type=str, help='Input directory containing images')
    parser.add_argument('output_dir', type=str, help='Output directory for processed images')
    parser.add_argument('--batch-only', action='store_true', 
                       help='Only copy existing JPEG files into batched folders')
    
    args = parser.parse_args()
    return Path(args.input_dir), Path(args.output_dir), args.batch_only

def get_batch_folder_name(batch_number: int) -> str:
    """
    Generate batch folder name based on batch number.
    
    Args:
        batch_number: Current batch number (0-based)
        
    Returns:
        str: Folder name in format 'batch_XXXXX-XXXXX'
    """
    start_index = batch_number * 100
    end_index = start_index + 100
    return f'batch_{start_index:05d}-{end_index:05d}'

def convert_to_rgb(image: Image.Image) -> Image.Image:
    """
    Convert image to RGB if it's in RGBA mode.
    
    Args:
        image: PIL Image object
        
    Returns:
        Image.Image: RGB version of the image
    """
    if image.mode == 'RGBA':
        logging.debug(f'🎨 Converting image from RGBA to RGB')
        # Create white background
        background = Image.new('RGB', image.size, (255, 255, 255))
        # Composite the image onto the background using alpha channel
        background.paste(image, mask=image.split()[3])
        return background
    return image

def try_load_image(file_path: Path) -> Optional[Image.Image]:
    """
    Attempt to load a file as an image and convert to RGB if needed.
    Supports JPEG, PNG, HEIF/HEIC, and other common formats.
    Attempts JPEG loading if HEIC loading fails.
    
    Args:
        file_path: Path to potential image file
        
    Returns:
        Optional[Image.Image]: Loaded image (in RGB mode) or None if failed
    """
    try:
        # Special handling for HEIF/HEIC files
        if file_path.suffix.lower() in ['.heic', '.heif']:
            logging.debug(f'📸 Attempting HEIF/HEIC load for: {file_path}')
            try:
                heif_file = pillow_heif.read_heif(str(file_path))
                image = Image.frombytes(
                    heif_file.mode, 
                    heif_file.size, 
                    heif_file.data,
                    "raw",
                    heif_file.mode,
                    heif_file.stride,
                )
            except Exception as heic_error:
                logging.debug(f'🔄 HEIF/HEIC load failed, attempting JPEG load: {str(heic_error)}')
                # If HEIC loading fails, try loading as JPEG
                image = Image.open(file_path)
        else:
            image = Image.open(file_path)
        
        return convert_to_rgb(image)
    except Exception as e:
        logging.info(f'❌ Failed to load image: {file_path} | Error: {str(e)}')
        return None

def get_all_files_recursive(directory: Path) -> list[Path]:
    """
    Get all files recursively from a directory.
    
    Args:
        directory: Root directory to search
        
    Returns:
        list[Path]: List of all files found recursively
    """
    files = []
    for item in directory.rglob('*'):
        if item.is_file():
            files.append(item)
    return files

def chunk_list(lst: List, chunk_size: int) -> Iterator[List]:
    """
    Split a list into chunks of specified size.
    
    Args:
        lst: List to split
        chunk_size: Size of each chunk
        
    Returns:
        Iterator[List]: Iterator of list chunks
    """
    iterator = iter(lst)
    return iter(lambda: list(islice(iterator, chunk_size)), [])

def calculate_resize_dimensions(width: int, height: int, target_pixels: int = 1_920_000) -> Tuple[int, int]:
    """
    Calculate new dimensions that maintain aspect ratio while hitting target pixel count.
    
    Args:
        width: Current image width
        height: Current image height
        target_pixels: Target total pixel count (default 1600×1200 = 1,920,000)
        
    Returns:
        Tuple[int, int]: New (width, height) maintaining aspect ratio
    """
    current_pixels = width * height
    if current_pixels <= target_pixels:
        return width, height
        
    # Calculate scaling factor to hit target pixels
    scale = (target_pixels / current_pixels) ** 0.5
    
    # Round to integers while maintaining aspect ratio
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    return new_width, new_height

def process_file(args: Tuple[Path, Path, Path]) -> Optional[str]:
    """
    Process a single file in a worker process.
    
    Args:
        args: Tuple containing (file_path, output_dir, batch_folder)
        
    Returns:
        Optional[str]: Output filename if successful, None if failed
    """
    file_path, output_dir, batch_folder = args
    
    # Skip hidden files
    if file_path.name.startswith('.'):
        return None
        
    # Try to load and process the image
    image = try_load_image(file_path)
    if image is None:
        return None
        
    # Calculate new dimensions and resize if necessary
    new_width, new_height = calculate_resize_dimensions(image.width, image.height)
    if (new_width, new_height) != (image.width, image.height):
        logging.debug(f'📐 Resizing image from {image.width}×{image.height} to {new_width}×{new_height}')
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # Save as JPG
    output_filename = batch_folder / f'{file_path.stem}.jpg'
    try:
        image.save(output_filename, 'JPEG', quality=80)
        return str(output_filename)
    except Exception as e:
        logging.error(f'❌ Failed to save {output_filename}: {str(e)}')
        return None
    finally:
        image.close()

def process_images(input_dir: Path, output_dir: Path) -> None:
    """
    Process all files in input directory using multiple processes.
    
    Args:
        input_dir: Source directory containing images
        output_dir: Destination directory for processed images
    """
    logging.info('🚀 Starting parallel image processing...')
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # List all files recursively in input directory
    input_files = get_all_files_recursive(input_dir)
    logging.info(f'📁 Found {len(input_files)} files to process')
    
    # Track failed files
    failed_files: List[Path] = []
    
    # Calculate optimal chunk size and number of processes
    num_processes = multiprocessing.cpu_count()
    chunk_size = 100  # Keep batch size consistent
    logging.info(f'🧮 Using {num_processes} processes')
    
    # Prepare batches of work
    successful_conversions = 0
    with tqdm(total=len(input_files), desc='Processing images', unit='files') as pbar:
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            for batch_idx, file_batch in enumerate(chunk_list(input_files, chunk_size)):
                # Create batch folder
                batch_folder = output_dir / get_batch_folder_name(batch_idx)
                batch_folder.mkdir(exist_ok=True)
                
                # Prepare arguments for each file in the batch
                file_args = [(f, output_dir, batch_folder) for f in file_batch]
                
                # Process batch in parallel
                results = list(executor.map(process_file, file_args))
                
                # Update progress and track failures
                successful = sum(1 for r in results if r is not None)
                failed = len(file_batch) - successful
                if failed > 0:
                    failed_files.extend([f for f, r in zip(file_batch, results) if r is None])
                successful_conversions += successful
                pbar.update(len(file_batch))
                pbar.set_postfix({'converted': successful_conversions, 'failed': len(failed_files)})
    
    # Log summary of failures
    if failed_files:
        logging.info('📋 Summary of failed files:')
        for failed_file in failed_files:
            logging.info(f'   ❌ {failed_file}')
        
    logging.info(f'✨ Processing complete! Converted {successful_conversions} images, '
                f'failed {len(failed_files)} images, '
                f'across {(len(input_files) - 1) // chunk_size + 1} batches')

def is_jpeg(file_path: Path) -> bool:
    """
    Check if file has a JPEG extension.
    
    Args:
        file_path: Path to check
        
    Returns:
        bool: True if file has .jpg or .jpeg extension
    """
    return file_path.suffix.lower() in ['.jpg', '.jpeg']

def copy_file(args: Tuple[Path, Path, Path]) -> Optional[str]:
    """
    Copy a single JPEG file in a worker process.
    
    Args:
        args: Tuple containing (file_path, output_dir, batch_folder)
        
    Returns:
        Optional[str]: Output filename if successful, None if failed
    """
    file_path, output_dir, batch_folder = args
    output_filename = batch_folder / file_path.name
    
    try:
        output_filename.write_bytes(file_path.read_bytes())
        return str(output_filename)
    except Exception as e:
        logging.error(f'❌ Failed to copy {file_path}: {str(e)}')
        return None

def batch_copy_jpeg(input_dir: Path, output_dir: Path) -> None:
    """
    Copy JPEG files using multiple processes.
    
    Args:
        input_dir: Source directory containing JPEG files
        output_dir: Destination directory for batched files
    """
    logging.info('🚀 Starting parallel JPEG batch copy...')
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all JPEG files recursively
    jpeg_files = [f for f in get_all_files_recursive(input_dir) 
                 if is_jpeg(f) and not f.name.startswith('.')]
    logging.info(f'📁 Found {len(jpeg_files)} JPEG files to process')
    
    # Calculate optimal number of processes
    num_processes = multiprocessing.cpu_count()
    chunk_size = 100  # Keep batch size consistent
    logging.info(f'🧮 Using {num_processes} processes')
    
    # Process files in parallel
    successful_copies = 0
    with tqdm(total=len(jpeg_files), desc='Copying JPEG files', unit='files') as pbar:
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            for batch_idx, file_batch in enumerate(chunk_list(jpeg_files, chunk_size)):
                # Create batch folder
                batch_folder = output_dir / get_batch_folder_name(batch_idx)
                batch_folder.mkdir(exist_ok=True)
                
                # Prepare arguments for each file in the batch
                file_args = [(f, output_dir, batch_folder) for f in file_batch]
                
                # Process batch in parallel
                results = list(executor.map(copy_file, file_args))
                
                # Update progress
                successful = sum(1 for r in results if r is not None)
                successful_copies += successful
                pbar.update(len(file_batch))
                pbar.set_postfix({'copied': successful_copies})
    
    logging.info(f'✨ Copy complete! Moved {successful_copies} JPEG files '
                f'across {(len(jpeg_files) - 1) // chunk_size + 1} batches')

def main() -> None:
    """Main entry point of the script."""
    setup_logging()
    try:
        input_dir, output_dir, batch_only = parse_arguments()
        if batch_only:
            batch_copy_jpeg(input_dir, output_dir)
        else:
            process_images(input_dir, output_dir)
    except KeyboardInterrupt:
        logging.info('⚠️ Processing interrupted by user')
        sys.exit(1)
    except Exception as e:
        logging.error(f'💥 An error occurred: {str(e)}')
        sys.exit(1)

if __name__ == '__main__':
    main()
