# Scalable Learning Coefficient Estimation with SGLD

Installation: Use [pipenv](https://pipenv.pypa.io/en/latest/) to install required packages by running `pipenv install` in the top directory containing `Pipfile.lock` and start up the virtual environment with `pipenv shell`. 


# MNIST experiments
The SLURM job scripts to obtain MNIST training results for both SGD and entropy-SGD runs are contained in the directory `job_scripts`. Parameters and RNG seeds used in the paper are written as default in the scripts. Each will produce an output directory. 

Once those jobs are completed, point `dir1, dir2` in `plot_mnist_expt.ipynb` to the resulting output directories to generate plots. 

# Last epoch local coefficient computation - SGLD hyperparameter sweeps. 
Point `directories` in `run_local_coeff_computation.py` the same output directories and and run `python run_local_coeff_computation.py` to produce local learning coefficient estimates on many combinations of hyper parameters. 

This experiment is handled by [sacred](https://sacred.readthedocs.io/en/stable/index.html) with MongoDB observer running on `localhost:27017`, you might need `mongodb` database process running on your computer or change to a different observer. 


# Reproducing normal crossing potential experiments
Run `python parallel_sgld_2dnormalcrossing_experiments.py` to reproduce the 2D normal crossing potential experiments for the same combination of parameters and RNG seeds. This experiment is again observed by `sacred + MongoDB`. 

In `plot_toypotential_expt.ipynb`, point the document database filter `{"status": "COMPLETED", "expt_name": YOUR_EXPT_NAME}` to the correct name to generate plots and tables for these experiments. 

# Reproducing two component potential experiments
Run 
```
python two_component_potential_experiment.py --num_repeat 100 --outputdir ./outputs/twocomp/expt_migration_test/ --max_workers 6 --num_training_data 1000
```
to reproduce the two component potential experiments. The corresponding histogram and json result file will be saved at the specified output directory. 

One can point `plot_toypotential_expt.ipynb`, point `RESULT_DIR` to corresponding output directory to read in the result and generate the same histogram. 
