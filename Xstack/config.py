#!/usr/bin/env python3
"""
Configuration files for Xstack.
"""
import os

version_file = os.path.join(os.path.dirname(__file__), "VERSION")
with open(version_file) as f:
    lines = f.readlines()
    VERSION = lines[0].strip()
    LASTUPDATE = lines[1].strip()
WEB = "https://github.com/AstroChensj/Xstack"


default_nh_file = os.path.join(os.path.dirname(__file__),"data","tbabs_1e20.txt")