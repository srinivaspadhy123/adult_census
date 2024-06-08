echo [$(date)]: "START"

echo [$(date)]: "Creating conda environment for the project"

conda create --prefix ./env python=3.11.7 -y

echo [$(date)]: "Activating the environment"

source activate ./env

echo [$(date)]: "Installing the required packages"

pip install -r requirements.txt