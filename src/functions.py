import zipfile
import tarfile
import platform
from typing import Literal
import os
import requests
import subprocess


class ChatMessageModel:
    message: str
    userType: Literal['system', 'human', 'ai']


def list_all_files(directory):
    file_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_paths.append(os.path.join(root, file))
    return file_paths


def download_tesseract(cwd):
    """
    For Windows, download the Tesseract executable from the official GitHub release page.
    For macOS, install Tesseract using Homebrew. (untested, cause I don't have a Mac System)
    For Linux, install Tesseract using apt-get. (untested, cause I don't have a Linux System)
    """
    system = platform.system()
    tesseract_path = os.path.join(cwd, 'tesseract')

    if system == 'Windows':
        tesseract_exe = os.path.join(tesseract_path, 'tesseract.exe')
        if not os.path.exists(tesseract_exe):
            url = 'https://github.com/tesseract-ocr/tesseract/releases/download/5.5.0/tesseract-ocr-w64-setup-5.5.0.20241111.exe'
            download_and_extract(url, tesseract_path, 'exe')
        return tesseract_exe

    elif system == 'Darwin':  # macOS
        tesseract_bin = os.path.join(tesseract_path, 'bin', 'tesseract')
        if not os.path.exists(tesseract_bin):
            os.makedirs(tesseract_path, exist_ok=True)
            subprocess.run(['brew', 'install', 'tesseract'],
                           cwd=tesseract_path, check=True)
        return tesseract_bin

    elif system == 'Linux':
        tesseract_bin = os.path.join(tesseract_path, 'bin', 'tesseract')
        if not os.path.exists(tesseract_bin):
            os.makedirs(tesseract_path, exist_ok=True)
            subprocess.run(['apt-get', 'update'],
                           cwd=tesseract_path, check=True)
            subprocess.run(['apt-get', 'install', '-y',
                           'tesseract-ocr'], cwd=tesseract_path, check=True)
        return tesseract_bin

    else:
        raise Exception(f"Unsupported platform: {system}")


def download_and_extract(url, dest_path, file_type):
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        if file_type == 'exe':
            with open(os.path.join(dest_path, 'tesseract.exe'), 'wb') as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
        elif file_type == 'tar.gz':
            tar_path = os.path.join(dest_path, 'tesseract.tar.gz')
            with open(tar_path, 'wb') as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            with tarfile.open(tar_path, 'r:gz') as tar:
                tar.extractall(path=dest_path)
            os.remove(tar_path)
        elif file_type == 'zip':
            zip_path = os.path.join(dest_path, 'tesseract.zip')
            with open(zip_path, 'wb') as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(dest_path)
            os.remove(zip_path)
        print("Tesseract downloaded and extracted successfully.")
    else:
        raise Exception("Failed to download Tesseract executable.")
