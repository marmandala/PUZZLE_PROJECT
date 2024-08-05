import os
import json
import random
from PIL import Image
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


class ImagePairsViewer:
    def __init__(self, pairs, num_positive_samples=5, num_negative_samples=5, add_separator=True, separator_width=1):
        self.pairs = pairs
        self.add_separator = add_separator
        self.separator_width = separator_width

        # Randomly select pairs
        positive_pairs = [pair for pair in pairs if pair[2] == 1]
        negative_pairs = [pair for pair in pairs if pair[2] == 0]

        self.sample_pairs = random.sample(positive_pairs, min(num_positive_samples, len(positive_pairs))) + \
                            random.sample(negative_pairs, min(num_negative_samples, len(negative_pairs)))

        random.shuffle(self.sample_pairs)  # Shuffle the selected pairs
        self.current_index = 0

        self.fig, self.ax = plt.subplots()
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.show_pair()

    def show_pair(self):
        piece1_path, piece2_path, label = self.sample_pairs[self.current_index]
        piece1 = Image.open(piece1_path)
        piece2 = Image.open(piece2_path)

        piece1 = piece1.convert('RGB')
        piece2 = piece2.convert('RGB')

        piece1_array = np.array(piece1)
        piece2_array = np.array(piece2)

        # Resize images to have the same height
        height = min(piece1_array.shape[0], piece2_array.shape[0])
        piece1_array = self.resize_to_height(piece1_array, height)
        piece2_array = self.resize_to_height(piece2_array, height)

        if (self.add_separator):
            separator = np.zeros((height, self.separator_width, 3), dtype=np.uint8)
            combined_image = np.hstack((piece1_array, separator, piece2_array))
        else:
            combined_image = np.hstack((piece1_array, piece2_array))

        self.ax.clear()
        self.ax.imshow(combined_image)
        self.ax.set_title(f'Pair {self.current_index + 1} - Label: {label}')
        self.fig.canvas.draw()

    def resize_to_height(self, image_array, height):
        img = Image.fromarray(image_array)
        width = int(img.width * height / img.height)
        img_resized = img.resize((width, height), Image.LANCZOS)
        return np.array(img_resized)

    def on_key(self, event):
        if event.key == 'left':
            self.current_index = (self.current_index - 1) % len(self.sample_pairs)
            self.show_pair()
        elif event.key == 'right':
            self.current_index = (self.current_index + 1) % len(self.sample_pairs)
            self.show_pair()


def split_image(image_path, rows, cols):
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
        piece_path = os.path.join(output_dir, f'piece_{idx}.png')

        # Convert the image to RGB if it is in CMYK color mode
        if piece.mode == 'CMYK':
            piece = piece.convert('RGB')

        piece.save(piece_path)
        piece_paths.append(piece_path)

    return piece_paths


def get_image_paths_from_directory(directory, extensions=['.jpg', '.jpeg', '.png']):
    image_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            if any(file.lower().endswith(ext) for ext in extensions):
                image_paths.append(os.path.join(root, file))
    return image_paths


def create_dataset_from_directory(directory, rows=5, cols=5, output_dir='dataset', num_negative_samples_per_image=40):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    image_paths = get_image_paths_from_directory(directory)
    all_piece_paths = []
    pairs = []

    piece_index = 0  # Global index for image pieces

    for image_path in tqdm(image_paths, desc='Processing images'):
        pieces = split_image(image_path, rows, cols)
        piece_paths = save_pieces(pieces, output_dir, start_index=piece_index)
        all_piece_paths.extend(piece_paths)

        # Generate positive pairs
        for i in range(rows):
            for j in range(cols - 1):
                idx1 = i * cols + j
                idx2 = idx1 + 1
                pairs.append((piece_paths[idx1], piece_paths[idx2], 1))

        # Generate negative pairs within the same image
        for _ in range(num_negative_samples_per_image):
            piece1_path, piece2_path = random.sample(piece_paths, 2)
            while piece_paths.index(piece1_path) // cols == piece_paths.index(piece2_path) // cols:
                piece2_path = random.choice(piece_paths)
            pairs.append((piece1_path, piece2_path, 0))

        piece_index += len(pieces)  # Update global index

    pairs_file_path = os.path.join(output_dir, 'pairs.txt')
    with open(pairs_file_path, 'w') as f:
        for pair in tqdm(pairs, desc='Writing pairs to file'):
            piece1_path, piece2_path, label = pair
            f.write(f'{piece1_path} {piece2_path} {label}\n')

    return pairs


with open('queries.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

queries = data['queries']

for query in queries:
    print(query)

# downloader.download_main(queries=queries, limit=30, output_directory="downloads")

directory = 'downloads/data'

# converter.convert_images_to_png(directory)

pairs = create_dataset_from_directory(directory)

viewer = ImagePairsViewer(pairs, num_positive_samples=50, num_negative_samples=50, add_separator=False)
plt.show()

