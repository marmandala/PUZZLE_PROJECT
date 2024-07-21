import os
import random
from PIL import Image
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import converter


def split_image(image_path, rows, cols, output_dir):
    image = Image.open(image_path)
    img_width, img_height = image.size
    piece_width = img_width // cols
    piece_height = img_height // rows

    pieces = []
    for i in range(rows):
        for j in range(cols):
            box = (j * piece_width, i * piece_height, (j + 1) * piece_width, (i + 1) * piece_width)
            piece = image.crop(box)
            pieces.append(piece)
    save_pieces(pieces, output_dir)
    return pieces


def save_pieces(pieces, output_dir, start_index=0):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    piece_paths = []
    for idx, piece in enumerate(pieces, start=start_index):
        piece_path = os.path.join(output_dir, f'piece_{idx}.png')
        piece.save(piece_path)
        piece_paths.append(piece_path)

    return piece_paths


def process_images(input_directory, output_directory, rows, cols):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for file_name in os.listdir(input_directory):
        if file_name.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'gif')):
            input_path = os.path.join(input_directory, file_name)
            output_subdir = os.path.join(output_directory, os.path.splitext(file_name)[0])
            split_image(input_path, rows, cols, output_subdir)


input_directory = 'test/input'
output_directory = 'test/output'

converter.convert_images_to_png(input_directory)

process_images(input_directory, output_directory, 5, 5)
