import argparse
import logging
from pathlib import Path
from typing import Tuple, Optional, List
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
import pandas as pd
import numpy as np


def setup_logging() -> None:
    """Configure logging for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def load_model() -> StableDiffusionPipeline:
    """
    Load the Stable Diffusion model.
    
    Returns:
        StableDiffusionPipeline: Loaded model
    """
    logging.info('🤖 Loading Stable Diffusion model...')
    
    # Check CUDA availability
    if torch.cuda.is_available():
        logging.info(f'✅ CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}')
        device = 'cuda'
        dtype = torch.float16
    else:
        logging.warning('⚠️ CUDA is not available. Using CPU (this will be slow)')
        device = 'cpu'
        dtype = torch.float32
    
    try:
        model_id = 'runwayml/stable-diffusion-v1-5'
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            safety_checker=None  # Disable safety checker for faster generation
        )
        pipe = pipe.to(device)
        
        # Enable memory optimization
        pipe.enable_attention_slicing()
        if device == 'cuda':
            pipe.enable_vae_slicing()
        
        logging.info('✅ Model loaded successfully')
        return pipe
    except Exception as e:
        logging.error(f'❌ Error loading model: {str(e)}')
        raise


def generate_ai_image(
    pipe: StableDiffusionPipeline,
    output_path: Path,
    index: int,
    width: int,
    height: int,
    prompt: str = 'psychedelic abstract art, vibrant colors, symmetrical patterns, kaleidoscope effect, high quality, detailed',
    negative_prompt: str = 'blurry, low quality, distorted, ugly, bad anatomy',
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    seed: Optional[int] = None
) -> Path:
    """
    Generate an AI image using Stable Diffusion and save it to disk.
    
    Args:
        pipe: Stable Diffusion pipeline
        output_path: Directory to save the image
        index: Index number for the filename
        width: Width of the image in pixels
        height: Height of the image in pixels
        prompt: Text prompt for image generation
        negative_prompt: Text prompt for what to avoid
        num_inference_steps: Number of denoising steps
        guidance_scale: How closely to follow the prompt
        seed: Random seed for reproducibility
        
    Returns:
        Path: Path to the generated image
    """
    logging.info(f'🎨 Generating AI image {index}...')
    
    try:
        # Set random seed if provided
        if seed is not None:
            generator = torch.Generator('cuda' if torch.cuda.is_available() else 'cpu').manual_seed(seed)
        else:
            generator = None
        
        # Generate the image
        with torch.inference_mode():
            result = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
                generator=generator
            )
        
        # Get the image and verify it's not black
        image = result.images[0]
        
        # Convert to numpy array and check if image is black
        img_array = np.array(image)
        if np.mean(img_array) < 1.0:  # If average pixel value is very low
            logging.error('❌ Generated image is black or nearly black')
            raise ValueError('Generated image is black or nearly black')
        
        # Save the image
        output_file = output_path / f'ai_art_{index}.png'
        image.save(output_file)
        
        logging.info(f'✅ AI image {index} generated at {output_file}')
        return output_file
        
    except Exception as e:
        logging.error(f'❌ Error generating image: {str(e)}')
        raise


def overlay_quote(
    image_path: Path,
    quote: str,
    author: str,
    output_path: Path,
    font_size: int = 32,
    text_color: Tuple[int, int, int] = (255, 255, 255)
) -> Path:
    """
    Overlay a quote on an image.
    
    Args:
        image_path: Path to the image
        quote: Quote text to overlay
        author: Author of the quote
        output_path: Path to save the resulting image
        font_size: Size of the font
        text_color: RGB color tuple for the text
        
    Returns:
        Path: Path to the output image with quote
    """
    logging.info(f'💬 Adding quote to {image_path.name}...')
    
    # Open the image
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)
    
    # Use default font since we don't have a specific font file
    try:
        font = ImageFont.load_default()
        
        # Calculate positions - place quote in upper part, author in lower right
        img_width, img_height = img.size
        quote_position = (50, 50)
        
        # Format the text with quote and attribution
        formatted_text = f'"{quote}"\n- {author}'
        
        # Draw the text
        draw.text(quote_position, formatted_text, font=font, fill=text_color)
        
        # Save the image
        img.save(output_path)
        logging.info(f'✅ Quote added and saved to {output_path}')
        return output_path
    
    except Exception as e:
        logging.error(f'❌ Error adding quote: {e}')
        raise
    finally:
        img.close()


def generate_ai_art_quotes(
    quotes_file: Path,
    output_dir: Path,
    num_images: int = 10,
    width: int = 512,
    height: int = 512,
    prompt: str = 'psychedelic abstract art, vibrant colors, symmetrical patterns, kaleidoscope effect, high quality, detailed',
    negative_prompt: str = 'blurry, low quality, distorted, ugly, bad anatomy',
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    seed: Optional[int] = None
) -> List[Path]:
    """
    Generate AI art images and overlay quotes from a CSV file.
    
    Args:
        quotes_file: Path to CSV file with quotes
        output_dir: Directory to save output images
        num_images: Number of images to generate
        width: Width of images in pixels
        height: Height of images in pixels
        prompt: Text prompt for image generation
        negative_prompt: Text prompt for what to avoid
        num_inference_steps: Number of denoising steps
        guidance_scale: How closely to follow the prompt
        seed: Random seed for reproducibility
        
    Returns:
        List[Path]: Paths to generated images with quotes
    """
    logging.info(f'🔄 Starting AI art generation with quotes from {quotes_file}')
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Read quotes from CSV
    try:
        quotes_df = pd.read_csv(quotes_file)
        required_columns = ['Quote', 'Author']
        missing_columns = [col for col in required_columns if col not in quotes_df.columns]
        
        if missing_columns:
            raise ValueError(f'❌ CSV file is missing required columns: {", ".join(missing_columns)}')
            
        logging.info(f'📊 Found {len(quotes_df)} quotes in CSV file')
        
    except pd.errors.EmptyDataError:
        raise ValueError(f'❌ CSV file {quotes_file} is empty')
    except pd.errors.ParserError:
        raise ValueError(f'❌ Error parsing CSV file {quotes_file}. Please ensure it is a valid CSV file')
    
    # Check if we have enough quotes
    if len(quotes_df) < num_images:
        logging.warning(f'⚠️ Only {len(quotes_df)} quotes available, but {num_images} requested')
        num_images = min(num_images, len(quotes_df))
    
    # Load the model
    pipe = load_model()
    
    output_files = []
    
    # Generate AI art and overlay quotes
    for i in range(num_images):
        try:
            # Generate AI art
            image_path = generate_ai_image(
                pipe=pipe,
                output_path=output_dir,
                index=i,
                width=width,
                height=height,
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                seed=seed
            )
            
            # Get quote and author
            quote = quotes_df.iloc[i]['Quote']
            author = quotes_df.iloc[i]['Author']
            
            # Create output path for the image with quote
            quote_image_path = output_dir / f'ai_art_quote_{i}.png'
            
            # Overlay quote on image
            final_path = overlay_quote(image_path, quote, author, quote_image_path)
            output_files.append(final_path)
            
        except Exception as e:
            logging.error(f'❌ Failed to generate image {i}: {str(e)}')
            continue
    
    if not output_files:
        raise RuntimeError('❌ Failed to generate any images')
    
    logging.info(f'✅ Generated {len(output_files)} AI art images with quotes')
    return output_files


def main():
    """Main entry point for the AI art generator."""
    parser = argparse.ArgumentParser(description='Generate AI art images with quotes')
    
    parser.add_argument('--quotes', type=str, default='quotes.csv', 
                        help='Path to CSV file with quotes (columns: author, quote)')
    parser.add_argument('--output', type=str, required=True,
                        help='Directory to save output images')
    parser.add_argument('--count', type=int, default=10,
                        help='Number of images to generate')
    parser.add_argument('--width', type=int, default=512,
                        help='Width of image in pixels')
    parser.add_argument('--height', type=int, default=512,
                        help='Height of image in pixels')
    parser.add_argument('--prompt', type=str, 
                        default='psychedelic abstract art, vibrant colors, symmetrical patterns, kaleidoscope effect, high quality, detailed',
                        help='Text prompt for image generation')
    parser.add_argument('--negative-prompt', type=str,
                        default='blurry, low quality, distorted, ugly, bad anatomy',
                        help='Text prompt for what to avoid')
    parser.add_argument('--steps', type=int, default=50,
                        help='Number of denoising steps')
    parser.add_argument('--guidance', type=float, default=7.5,
                        help='How closely to follow the prompt')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    # Convert paths to Path objects
    quotes_path = Path(args.quotes)
    output_path = Path(args.output)
    
    # Generate AI art with quotes
    generate_ai_art_quotes(
        quotes_file=quotes_path,
        output_dir=output_path,
        num_images=args.count,
        width=args.width,
        height=args.height,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        seed=args.seed
    )


if __name__ == '__main__':
    main()
