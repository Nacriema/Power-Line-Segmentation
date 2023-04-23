#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Jun 20 19:29:54 2022

@author: Nacriema

Refs:

"""
from . import coerce_to_path_and_check_exist
import logging
import time


# For more colors, see example: https://www.geeksforgeeks.org/print-colors-python-terminal/
class TerminalColors:
    HEADER = '\033[95m'
    OKBLUE = '\033[34m'
    OKGREEN = '\033[32m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def get_time():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def print_info(s):
    print(TerminalColors.OKGREEN + "[" + get_time() + "] [INFO] " + str(s) + TerminalColors.ENDC)


def print_warning(s):
    print(TerminalColors.WARNING + "[" + get_time() + "] [WARN] " + str(s) + TerminalColors.ENDC)


def print_error(s):
    print(TerminalColors.FAIL + "[" + get_time() + "] [ERR] " + str(s) + TerminalColors.ENDC)


def get_logger(log_dir, name):
    log_dir = coerce_to_path_and_check_exist(log_dir)
    logger = logging.getLogger(name)
    file_path = log_dir / "{}.log".format(name)
    hdlr = logging.FileHandler(file_path)
    formmater = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s")
    hdlr.setFormatter(formmater)
    logger.addHandler(hdlr)
    logger.setLevel(logging.INFO)
    return logger
