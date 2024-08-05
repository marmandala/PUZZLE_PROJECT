import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import torch.nn.functional as F
import time

# Класс датасета для загрузки парных изображений
class PuzzlePairDataset(Dataset):
    def __init__(self, pairs_file, root_dir, transform=None):
        self.pairs = []
        self.root_dir = root_dir
        with open(pairs_file, 'r') as f:
            for line in f:
                piece1_path, piece2_path, label = line.strip().split()
                self.pairs.append((piece1_path, piece2_path, int(label)))
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        piece1_path, piece2_path, label = self.pairs[idx]
        piece1_path = os.path.join(self.root_dir, piece1_path)
        piece2_path = os.path.join(self.root_dir, piece2_path)

        # Загрузка изображений и конвертация в RGB (если необходимо)
        piece1 = Image.open(piece1_path).convert('RGB')
        piece2 = Image.open(piece2_path).convert('RGB')

        # Применение трансформаций, если они указаны
        if self.transform:
            piece1 = self.transform(piece1)
            piece2 = self.transform(piece2)

        return piece1, piece2, label

# Улучшенная архитектура сверточной нейросети
class EnhancedCNN(nn.Module):
    def __init__(self):
        super(EnhancedCNN, self).__init__()
        self.conv1 = nn.Conv2d(6, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)  # Новый слой
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(512 * 4 * 4, 1024)  # Обновленный размер
        self.fc2 = nn.Linear(1024, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.pool(F.relu(self.conv5(x)))  # Новый слой
        x = x.view(-1, 512 * 4 * 4)  # Обновленный размер
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

# Функция обучения модели
def train_model(dataset_path, pairs_file, num_epochs=60, batch_size=8, learning_rate=0.0004, save_model_path='model_e.pth'):
    # Подготовка данных
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # Изменение размера изображения до 128x128
        transforms.ToTensor(),           # Преобразование изображения в тензор
    ])

    dataset = PuzzlePairDataset(pairs_file, root_dir=dataset_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Инициализация модели, оптимизатора и критерия
    model = EnhancedCNN()

    # Проверка наличия GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Обучение модели
    model.train()
    start_time = time.time()
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        epoch_loss = 0.0
        for i, (piece1, piece2, labels) in enumerate(dataloader):
            # Перемещение данных на GPU
            piece1, piece2, labels = piece1.to(device), piece2.to(device), labels.to(device)

            # Объединение изображений пазлов по каналам
            inputs = torch.cat((piece1, piece2), dim=1)  # Важно указать dim=1 для конкатенации по каналам

            labels = labels.unsqueeze(1).float()

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            if (i+1) % 10 == 0:  # Вывод каждые 10 итераций
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item()}')

        epoch_duration = time.time() - epoch_start_time
        remaining_time = (num_epochs - epoch - 1) * epoch_duration
        print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {epoch_loss / len(dataloader)}, Time Taken: {epoch_duration:.2f}s, Estimated Remaining Time: {remaining_time:.2f}s')

    # Сохранение модели
    torch.save(model.state_dict(), save_model_path)
    print(f'Model saved to {save_model_path}')

# Пример использования функции для обучения модели
dataset_path = 'dataset'  # Путь к директории с датасетом
pairs_file = os.path.join(dataset_path, 'pairs.txt')  # Полный путь к файлу pairs.txt

train_model(dataset_path, pairs_file)

