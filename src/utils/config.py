"""
Shared paths, computed relative to this repo — NOT hardcoded to any one
person's computer or OneDrive username. This is what makes the same code
work unchanged after every teammate clones the GitHub repo.

After cloning, each person creates their OWN local copy of the dataset at:
    data/raw/DB1_B/*.tif
    data/raw/DB2_B/*.tif
    data/raw/DB3_B/*.tif
    data/raw/DB4_B/*.tif
(same FVC2002 zips everyone already downloaded — just extract them here
instead of into OneDrive). `data/` is in .gitignore, so the dataset itself
never gets pushed to GitHub — only the code does.
"""

import os

# This file lives at <repo_root>/src/utils/config.py, so climb up two levels.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

RAW_DIR = os.path.join(REPO_ROOT, "data", "raw")
PROCESSED_DIR = os.path.join(REPO_ROOT, "data", "processed")
RESULTS_DIR = os.path.join(REPO_ROOT, "results")

DBS = ["DB1_B", "DB2_B", "DB3_B", "DB4_B"]

# Default NFIQ2 install location from the official Windows MSI installer.
# Every teammate installs NFIQ2 the same way (see README) — this path should
# then be correct on every machine without editing.
NFIQ2_EXE = r"C:\Program Files\NFIQ 2\bin\nfiq2.exe"
