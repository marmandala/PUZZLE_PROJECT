def remove_sequence_from_file(file_path, sequence):
    try:
        # Чтение файла
        with open(file_path, 'r') as file:
            file_content = file.read()

        # Удаление последовательности
        modified_content = file_content.replace(sequence, '')

        # Запись обратно в файл
        with open(file_path, 'w') as file:
            file.write(modified_content)

        print(f"Последовательность '{sequence}' удалена из файла '{file_path}'.")

    except FileNotFoundError:
        print(f"Файл '{file_path}' не найден.")
    except Exception as e:
        print(f"Произошла ошибка: {e}")

# Пример использования
file_path = 'dataset/pairs.txt'
sequence_to_remove = 'dataset/'
remove_sequence_from_file(file_path, sequence_to_remove)

