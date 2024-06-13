import gzip
import shutil
import os

def main():
    # Define the path to the compressed file and the output path for the decompressed file
    compressed_file_path = '../data/raw_data.csv.gz'
    decompressed_file_path = '../data/raw_data.csv'

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(decompressed_file_path), exist_ok=True)

    # Decompress the gzip file
    with gzip.open(compressed_file_path, 'rb') as f_in:
        with open(decompressed_file_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    print(f"Decompressed file saved to: {decompressed_file_path}")

if __name__ == '__main__':
    main()