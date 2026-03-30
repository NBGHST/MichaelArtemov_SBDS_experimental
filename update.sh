source .venv_SBDS/bin/activate
python3 setup.py build_ext --inplace --force -j$(nproc)
deactivate