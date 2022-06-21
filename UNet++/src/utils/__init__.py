#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Jun 20 19:17:33 2022

@author: Nacriema

Refs:

"""
from pathlib import Path
from typing import Union, List


def coerce_to_path_and_check_exist(path: str) -> Path:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f'{path.absolute()} does not exist !')
    return path


def coerce_to_path_and_create_dir(path: str) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_files_from_dir(dir_path: str, valid_extensions: Union[List[str], str] = None, recursive: bool = False, sort: bool = False):
    # 1. Check the path is exist
    path = coerce_to_path_and_check_exist(dir_path)
    # 2. Find all files
    if recursive:
        # we can use resolve() and absolute()
        files = [f.absolute() for f in path.glob('**/*') if f.is_file()]
    else:
        files = [f.absolute() for f in path.glob('*') if f.is_file()]

    if valid_extensions is not None:
        valid_extensions = [valid_extensions] if isinstance(valid_extensions, str) else valid_extensions
        valid_extensions = ['.{}'.format(ext) if not ext.startswith('.') else ext for ext in valid_extensions]
        files = list(filter(lambda f: f.suffix in valid_extensions, files))

    return sorted(files) if sort else files


# TODO: Implement seed decorator to make the training process deterministic
