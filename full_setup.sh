#!/bin/bash
set -e
python3 -m venv .venv_SBDS
source .venv_SBDS/bin/activate
pip install -r requirements.txt
python3 setup.py build_ext --inplace --force -j$(nproc)
deactivate
