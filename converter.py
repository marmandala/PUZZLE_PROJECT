from PIL import Image
import os


def convert_images_to_png(directory):
    if not os.path.exists(directory):
        print(f"Директория '{directory}' не существует.")
        return

    files = os.listdir(directory)

    for file in files:
        file_path = os.path.join(directory, file)

        if os.path.isfile(file_path) and any(file_path.endswith(ext) for ext in ['.jpg', '.jpeg', '.bmp', '.gif', '.png', '.JPG']):
            try:
                with Image.open(file_path) as img:
                    filename, _ = os.path.splitext(file)

                    min_dim = min(img.size)
                    left = (img.width - min_dim) // 2
                    top = (img.height - min_dim) // 2
                    right = (img.width + min_dim) // 2
                    bottom = (img.height + min_dim) // 2

                    img_cropped = img.crop((left, top, right, bottom))

                    img_gray = img_cropped.convert("L")

                    img_cropped.save(os.path.join(directory, f"{filename}.png"), format="PNG")
                    print(f"Файл '{file}' успешно конвертирован и преобразован в PNG и градации серого.")

                os.remove(file_path)

            except Exception as e:
                print(f"Ошибка при конвертации файла '{file}': {str(e)}")

        else:
            print(f"Файл '{file}' не является изображением.")


if __name__ == "__main__":
    directory_path = 'downloads/data'
    convert_images_to_png(directory_path)
