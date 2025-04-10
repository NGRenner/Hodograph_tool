import os
import zipfile
import gzip
import shutil
from grutils.io import read_gzip, write_rv3x

def convert_gz_to_rv3x(gz_path, rv3x_output_path):
    radar = read_gzip(gz_path)
    write_rv3x(rv3x_output_path, radar)

def process_zip_files(folder_path):
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.zip'):
            zip_path = os.path.join(folder_path, file_name)
            base_name = os.path.splitext(file_name)[0]
            output_dir = os.path.join(folder_path, 'converted_rv3x')
            os.makedirs(output_dir, exist_ok=True)

            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                for member in zip_ref.namelist():
                    if member.endswith('.gz'):
                        extracted_path = zip_ref.extract(member, folder_path)
                        rv3x_filename = os.path.splitext(os.path.basename(member))[0] + '.rv3x'
                        rv3x_path = os.path.join(output_dir, rv3x_filename)

                        try:
                            convert_gz_to_rv3x(extracted_path, rv3x_path)
                            print(f"Converted: {member} -> {rv3x_filename}")
                        except Exception as e:
                            print(f"Error converting {member}: {e}")
                        finally:
                            os.remove(extracted_path)

if __name__ == "__main__":
    folder_with_zips = r"C:\path\to\your\radar\zips"  # Change this to your actual folder
    process_zip_files(folder_with_zips)
