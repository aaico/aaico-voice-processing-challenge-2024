#!/bin/bash

# Define the name of the virtual environment
venv_name=".venv"

# Create the virtual environment
python3.11 -m venv "$venv_name"

echo 'Success virtual env created'

# Activate the virtual environment
if [ "$OSTYPE" == "msys" ]; then  # For Windows
    activate_script="$venv_name/Scripts/activate"
    source "$activate_script"
else  # For macOS and Linux
    activate_script="$venv_name/bin/activate"
    source "$activate_script"
fi

# Check if the virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Error: Virtual environment is not activated. Exiting."
    exit 1
else
    echo "$venv_name is activated"
fi

# Upgrade pip inside the virtual environment
pip install --upgrade pip -q

echo "pip upgraded"

# Install requirements
echo "Start to install requirements"
pip install -r requirements.txt -q
echo "Finished to install requirements"

# Prompt user to install dev-requirements
read -p "Do you want to install dev-requirements? (y/n): " install_dev
if [ "$install_dev" == "y" ]; then
    echo "Start to install dev-requirements"
    pip install -r dev_requirements.txt -q
    echo "Finished to install dev-requirements"
else
    echo "Skipping dev-requirements installation"
fi
