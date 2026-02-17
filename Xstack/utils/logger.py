#!/usr/bin/env python3
"""
==================
Module for logging
==================
:Authors:   Shi-Jiang Chen (MPE, USTC)
            Johannes Buchner (MPE)
            Teng Liu (USTC)
:Email:     JohnnyCsj666@gmail.com


"""
import logging
import os
import psutil
from datetime import datetime,timezone


def get_logger(logname):
	"""
	Define a new customized logger.

	Parameters
	----------
	logname : str
		The name of the log file to write to.
	"""
	# Use a unique logger name based on log file name
	logger_name = logname.replace(".log", "")
	logger = logging.getLogger(logger_name)
	logger.setLevel(logging.DEBUG)

	# Avoid adding multiple handlers if the same logger is reused
	if os.path.exists(logname):
		os.remove(logname)
	file_handler = logging.FileHandler(logname)
	file_handler.setLevel(logging.INFO)
	formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
	file_handler.setFormatter(formatter)
	logger.addHandler(file_handler)

	return logger


def utc_now_iso():
	"""
	Get the current UTC time in ISO 8601 format (YYYY-MM-DDTHH:MM:SSZ).
	"""
	return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def get_ram_gb():
	"""
	Get the current RAM usage of the process in gigabytes (GB).
	"""
	process = psutil.Process(os.getpid())
	return process.memory_info().rss / 1024**3  # GB