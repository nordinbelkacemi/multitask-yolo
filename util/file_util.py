from typing import Dict


def get_file_name_from_absolute_path(absolute_path: str) -> str:
    return absolute_path.split("/")[-1].split(".")[0]
