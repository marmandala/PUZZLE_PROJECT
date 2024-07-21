from bing_image_downloader import downloader
import converter
import os
import time
import shutil


def download_images(query, limit, output_directory):
    try:
        # Скачивание изображений
        downloader.download(query, limit=limit, output_dir=output_directory, adult_filter_off=True, force_replace=False,
                            timeout=3)
        time.sleep(1)

        query_directory = os.path.join(output_directory, query)
        data_directory = os.path.join(output_directory, "data")

        if not os.path.exists(data_directory):
            os.makedirs(data_directory)

        for filename in os.listdir(query_directory):
            unique_filename = f"{query.replace(' ', '_')}_{filename}"
            unique_path = os.path.join(data_directory, unique_filename)

            # Обеспечение уникальности имени файла
            counter = 1
            while os.path.exists(unique_path):
                unique_filename = f"{query.replace(' ', '_')}_{counter}_{filename}"
                unique_path = os.path.join(data_directory, unique_filename)
                counter += 1

            shutil.move(os.path.join(query_directory, filename), unique_path)

        shutil.rmtree(query_directory)
        print(f"Downloaded images for {query}")

    except Exception as e:
        print(f"Error occurred: {e}")


def download_main(queries, limit, output_directory):
    try:
        for query in queries:
            download_images(query=query, limit=limit, output_directory=output_directory)

        data_directory = os.path.join(output_directory, "data")
        converter.convert_images_to_png(data_directory)

    except Exception as e:
        print(f"Error in download_main: {e}")

if __name__ == "__main__":
    queries = [
        "Beautiful nature landscapes",
        "City architecture at night",
        "Pets playing",
        "Delicious dishes on plate",
        "Street art and graffiti",
        "Sports events in motion",
        "Cars in motion",
        "People in traditional costumes",
        "Exotic animals in zoo",
        "Underwater world and fish",
        "Astronomical objects and stars",
        "Birds in flight",
        "Ancient ruins and artifacts",
        "Models on runway",
        "Aerial views and drones",
        "People working in office",
        "Children playing on playground",
        "Fruits and vegetables at market",
        "Dance and ballet",
        "Music concerts and performances"
    ]

    output_directory = "downloads"
    download_main(queries=queries, limit=1, output_directory=output_directory)
