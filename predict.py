import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
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

# Функция для загрузки и предсказания с использованием натренированной модели
def predict_similarity(model_path, piece1_path, piece2_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # Загрузка модели
    model = ImprovedCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

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

    # Вывод результата
    similarity = output.item()
    if similarity > 0.5:
        print(f"The pieces are similar with probability: {similarity}. Similarity: {similarity}")
    else:
        print(f"The pieces are not similar with probability: {1 - similarity}. Similarity: {similarity}")

# Пример использования функции для предсказания сходства двух изображений
model_path = 'model_b.pth'  # Путь к натренированной модели
piece1_path = 'output_pieces/piece_11.png'  # Путь к первому изображению
piece2_path = 'output_pieces/piece_13.png'  # Путь ко второму изображению

predict_similarity(model_path, piece1_path, piece2_path)

