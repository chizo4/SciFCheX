#!/bin/bash
#
# AUTHOR:
# Filip J. Cierkosz.
#
# INFO:
# The script is an automation for environment setup in Google
# Colab notebooks used for GPU access. It sets up both the pipeline
# (python 3.7) and the reranker (python 3.9) environments, incl.
# pip deps. It should work fine on Linux OS as well.
#
# USAGE (directly for Colab):
# !bash ./SciFCheX/script/colab-setup.sh

# STEP 0: Confirm conda installation.
if ! conda --version; then
  echo "Conda is not installed. Exiting the script..."
  exit 1
else
  echo "Confirmation: conda installed."; echo
fi

##################

# SCIFCHEX ENV SETUP

# STEP 1: Confirm "scifchex" conda env existance. Otherwise, build it.
if conda info --envs | grep "scifchex"; then
  echo "The conda env 'scifchex' already exists."; echo
else
  echo "Creating conda env 'scifchex'..."
  conda create --name scifchex python=3.7 conda-build -y
  echo; echo "Successfully created 'scifchex' environment."
fi

# STEP 2: Activate the environment and install pip dependencies.
echo; echo "Activating 'scifchex' and installing pip deps..."
source activate scifchex
pip install -r SciFCheX/requirements_scifchex.txt

# STEP 3: Install the target torch setup with CUDA support.
pip install torch==1.7.0+cu110 -f https://download.pytorch.org/whl/torch_stable.html
conda deactivate

##################

# RERANKER ENV SETUP

# STEP 1: Confirm "reranker" conda env existence. Otherwise, build it.
if conda info --envs | grep "reranker"; then
  echo "The conda env 'reranker' already exists."; echo
else
  echo "Creating conda env 'reranker'..."
  conda create --name reranker python=3.9 conda-build -y
  echo; echo "Successfully created 'reranker' environment."
fi

# STEP 2: Activate the environment and install pip dependencies.
echo; echo "Activating 'reranker' and installing pip deps..."
source activate reranker
pip install -r SciFCheX/requirements_reranker.txt
conda deactivate
