# Project Desciption
To reformat and refactor the provided script into a version that is production ready and create a PR.

## Command to create an environment from env.yaml file
conda env create -f env.yaml
conda env export --name mle-dev > env.yaml

## Command to activate environment
conda activate mle-dev

## Commands to install dependencies
conda install numpy
conda install pandas
conda install matplotlib

## Command to run the code
python nonstandardcode.py