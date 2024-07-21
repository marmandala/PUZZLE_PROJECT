import os
import shutil


def gather_files(source_dir, target_dir):
    # Проверяем, существует ли целевая директория, если нет - создаем её
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # Проходим по всем подкаталогам и файлам в исходной директории
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            source_file = os.path.join(root, file)
            target_file = os.path.join(target_dir, file)

            # Копируем файл в целевую директорию
            shutil.copy2(source_file, target_file)
            print(f'Файл {source_file} скопирован в {target_file}')


source_directory = 'downloads'  # Замените на путь к исходной директории
target_directory = 'downloads'  # Замените на путь к целевой директории

gather_files(source_directory, target_directory)
