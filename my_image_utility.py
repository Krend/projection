from PIL import Image
import my_math_functions as mmf

def load_image_to_bitmap(file_path):
    """
    Load an image file into a bitmap.

    :param file_path: Path to the image file
    :return: Image object (bitmap)
    """
    try:
        image = Image.open(file_path)        
        return image
    except Exception as e:
        print(f"Error loading image: {e}")
        return None
    

def get_pixel_color(image: Image, x: int, y: int):
    """
    Get the color of a pixel at (x, y) in the image.

    :param image: Image object (bitmap)
    :param x: X coordinate of the pixel
    :param y: Y coordinate of the pixel
    :return: Color of the pixel
    """
    if image is not None:
        return image.getpixel((x, y))
    return None

#def project_texture(