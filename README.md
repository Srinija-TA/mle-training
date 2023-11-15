# Project Description

# Command to create an environment from the env.yml file
conda env create -f env.yml
conda env export --name mle-dev > env.yml

# Command to activate the environment
conda activate mle-dev

# Command to run the python script
python <scriptname.py>