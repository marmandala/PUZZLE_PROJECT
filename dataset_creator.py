import os
import random
from PIL import Image
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


class ImagePairsViewer:
    def __init__(self, pairs, add_separator=True, separator_width=1):
        self.pairs = pairs
        self.add_separator = add_separator
        self.separator_width = separator_width
        self.current_index = 0

        self.fig, self.ax = plt.subplots()
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.show_pair()

    def show_pair(self):
        piece1_path, piece2_path, label = self.pairs[self.current_index]
        piece1 = Image.open(piece1_path)
        piece2 = Image.open(piece2_path)

        # Combine images horizontally
        combined_image = np.hstack((np.array(piece1), np.array(piece2)))

        if self.add_separator:
            # Add separator between images
            separator = np.zeros((combined_image.shape[0], self.separator_width, 3), dtype=np.uint8)
            combined_image = np.hstack((np.array(piece1), separator, np.array(piece2)))

        self.ax.clear()
        self.ax.imshow(combined_image)
        self.ax.set_title(f'Pair {self.current_index + 1} - Label: {label}')
        self.fig.canvas.draw()

    def on_key(self, event):
        if event.key == 'left':
            self.current_index = (self.current_index - 1) % len(self.pairs)
            self.show_pair()
        elif event.key == 'right':
            self.current_index = (self.current_index + 1) % len(self.pairs)
            self.show_pair()


def split_image(image_path, rows=10, cols=10):
    image = Image.open(image_path)
    img_width, img_height = image.size
    piece_width = img_width // cols
    piece_height = img_height // rows

    pieces = []
    for i in range(rows):
        for j in range(cols):
            box = (j * piece_width, i * piece_height, (j + 1) * piece_width, (i + 1) * piece_height)
            piece = image.crop(box)
            pieces.append(np.array(piece))

    return pieces


def save_pieces(pieces, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    piece_paths = []
    for idx, piece in enumerate(pieces):
        piece_image = Image.fromarray(piece)
        piece_path = os.path.join(output_dir, f'piece_{idx}.png')
        piece_image.save(piece_path)
        piece_paths.append(piece_path)

    return piece_paths


def create_dataset(image_paths, rows=10, cols=10, output_dir='dataset', num_negative_samples=1000):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    all_piece_paths = []
    pairs = []

    # Split images into pieces and save them
    for image_path in image_paths:
        pieces = split_image(image_path, rows, cols)
        piece_paths = save_pieces(pieces, output_dir)
        all_piece_paths.extend(piece_paths)

        # Create positive pairs
        for i in range(len(piece_paths) - 1):
            if (i + 1) % cols != 0:
                pairs.append((piece_paths[i], piece_paths[i + 1], 1))

    # Create negative pairs
    for _ in range(num_negative_samples):
        piece1_path, piece2_path = random.sample(all_piece_paths, 2)
        pairs.append((piece1_path, piece2_path, 0))

    # Write pairs to a text file
    pairs_file_path = os.path.join(output_dir, 'pairs.txt')
    with open(pairs_file_path, 'w') as f:
        for pair in tqdm(pairs):
            piece1_path, piece2_path, label = pair
            f.write(f'{piece1_path} {piece2_path} {label}\n')

    return pairs


# Пример использования
image_paths = ['m.jpg']
pairs = create_dataset(image_paths)

viewer = ImagePairsViewer(pairs)
plt.show()
