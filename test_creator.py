import os
import random
from PIL import Image

def split_image(image_path, rows=5, cols=5):
    image = Image.open(image_path)
    img_width, img_height = image.size
    piece_width = img_width // cols
    piece_height = img_height // rows

    pieces = []
    for i in range(rows):
        for j in range(cols):
            box = (j * piece_width, i * piece_height, (j + 1) * piece_width, (i + 1) * piece_height)
            piece = image.crop(box)
            pieces.append(piece)

    return pieces

def save_pieces(pieces, output_dir, start_index=0):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    piece_paths = []
    for idx, piece in enumerate(pieces, start=start_index):
        # Convert to grayscale
        #piece = piece.convert('L')
        piece_path = os.path.join(output_dir, f'piece_{idx}.png')
        piece.save(piece_path)
        piece_paths.append(piece_path)

    return piece_paths

def save_pieces_random(pieces, output_dir, start_index=0):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    piece_indices = list(range(start_index, start_index + len(pieces)))
    random.shuffle(piece_indices)

    piece_paths = []
    for idx, piece in zip(piece_indices, pieces):
        # Convert to grayscale
        #piece = piece.convert('L')
        piece_path = os.path.join(output_dir, f'piece_{idx}.png')
        piece.save(piece_path)
        piece_paths.append(piece_path)

    return piece_paths

# Example usage:
image_path = '/home/matvey/PUZZLE PROJECT/6.jpg'  # Replace with the path to your image
output_dir = 'output_pieces'  # Replace with the desired output directory
output_dir_random = 'output_pieces_random'  # Directory for random order pieces
rows, cols = 5, 5  # Number of rows and columns to split the image into

pieces = split_image(image_path, rows, cols)
save_pieces(pieces, output_dir)
save_pieces_random(pieces, output_dir_random)

