import os
import json
import random
from PIL import Image
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import downloader
import torch
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F

# Класс улучшенной сверточной нейросети
class ImprovedCNN(nn.Module):
    def __init__(self):
        super(ImprovedCNN, self).__init__()
        self.conv1 = nn.Conv2d(6, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(256 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 1)  # Один выходной нейрон для бинарной классификации (совпадают или нет)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = x.view(-1, 256 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

# Функция для предсказания сходства двух изображений
def predict_similarity(model, piece1_path, piece2_path, device):
    # Преобразование изображений
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # Изменение размера изображения до 128x128
        transforms.ToTensor(),           # Преобразование изображения в тензор
    ])

    # Загрузка и преобразование изображений
    piece1 = Image.open(piece1_path).convert('RGB')
    piece2 = Image.open(piece2_path).convert('RGB')
    piece1 = transform(piece1).unsqueeze(0).to(device)  # Добавляем размерность батча и перемещаем на устройство
    piece2 = transform(piece2).unsqueeze(0).to(device)  # Добавляем размерность батча и перемещаем на устройство

    # Конкатенация изображений по каналам
    inputs = torch.cat((piece1, piece2), dim=1)

    # Предсказание с помощью модели
    with torch.no_grad():
        output = model(inputs)

    # Возврат вероятности сходства
    similarity = output.item()
    return similarity

# Функция для загрузки и тестирования модели на датасете
def test_model_on_dataset(model_path, pairs, threshold=0.5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Загрузка модели
    model = ImprovedCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    correct = 0
    total = len(pairs)

    for piece1_path, piece2_path, label in tqdm(pairs, desc='Testing pairs'):
        similarity = predict_similarity(model, piece1_path, piece2_path, device)
        prediction = 1 if similarity > threshold else 0
        if prediction == label:
            correct += 1

    accuracy = correct / total
    print(f'Test Accuracy: {accuracy * 100:.2f}%')

# Загрузка запросов для скачивания изображений
with open('queries_tester.json', 'r', encoding='utf-8') as file:
    data = json.load(file)
queries = data['queries']

# Скачивание изображений
# downloader.download_main(queries=queries, limit=3, output_directory="downloads_tester")

# Генерация датасета
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
            pieces.append(piece)

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

# Создание датасета
directory = 'downloads_tester/data'
pairs = create_dataset_from_directory(directory)

# Тестирование модели на созданном датасете
model_path = 'model_c.pth'  # Путь к натренированной модели
test_model_on_dataset(model_path, pairs)

