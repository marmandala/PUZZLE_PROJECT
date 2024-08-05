import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
from collections import defaultdict

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

# Функция для загрузки и предсказания с использованием натренированной модели
def predict_similarity(model, piece1_path, piece2_path, transform, device):
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

    # Возвращаем вероятность схожести
    return output.item()

# Основная функция для сравнения всех изображений в директории
def compare_all_images_in_directory(model_path, image_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Загрузка модели
    model = ImprovedCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Преобразование изображений
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # Изменение размера изображения до 128x128
        transforms.ToTensor(),           # Преобразование изображения в тензор
    ])

    # Получение всех файлов изображений в директории
    image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('png', 'jpg', 'jpeg'))]

    total_comparisons = 0
    positive_results = defaultdict(list)

    # Сравнение всех изображений со всеми, включая оба порядка
    for i in range(len(image_files)):
        for j in range(len(image_files)):
            if i != j:
                total_comparisons += 1
                similarity = predict_similarity(model, image_files[i], image_files[j], transform, device)
                if similarity > 0.5:
                    pair = (image_files[i], image_files[j])
                    positive_results[image_files[i]].append((image_files[j], similarity))
                    positive_results[image_files[j]].append((image_files[i], similarity))

    # Фильтрация результатов, чтобы выбрать пару с наибольшей вероятностью для каждой позиции
    best_results = {}
    for key, values in positive_results.items():
        best_match = max(values, key=lambda x: x[1])
        if key not in best_results or best_match[1] > best_results[key][1]:
            best_results[key] = best_match

    # Уникальные результаты для вывода
    unique_results = set()
    for key, (match, similarity) in best_results.items():
        pair = tuple(sorted([key, match]))
        if pair not in unique_results:
            unique_results.add(pair)
            print(f"{pair[0]} <--> {pair[1]} with probability: {similarity}")

    print(f"Total comparisons made: {total_comparisons}")

# Пример использования функции для сравнения всех изображений в директории
model_path = 'model_b.pth'  # Путь к натренированной модели
image_dir = 'output_pieces'  # Директория с изображениями

compare_all_images_in_directory(model_path, image_dir)
