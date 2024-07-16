from bing_image_downloader import downloader
import converter
import os
import time


def download_images(query, limit, output_directory):
    try:
        downloader.download(query, limit=limit, output_dir=output_directory, adult_filter_off=True, force_replace=False,
                            timeout=3)
        time.sleep(1)
        os.rename(os.path.join(output_directory, query), os.path.join(output_directory, "data"))
        print(f"Downloaded images to {os.path.join(output_directory, 'data')}")
        converter.convert_images_to_png(f"{output_directory}\\data")
    except Exception as e:
        print(f"Error occurred: {e}")


if __name__ == "__main__":
    download_images(query="Раскраска автомобиль", limit=10, output_directory="downloads")
