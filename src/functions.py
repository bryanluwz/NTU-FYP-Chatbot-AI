from typing import Literal
import os


class ChatMessageModel:
    message: str
    userType: Literal['system', 'human', 'ai']


def list_all_files(directory):
    file_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_paths.append(os.path.join(root, file))
    return file_paths
