from sgld_2d_normalcrossing import ex
from utils import expand_dictionary
from multiprocessing import Pool
import functools


# Change the values in each config dictionary for different experiments. 
# All values of type `list` in the dictionary will be expanded with itertools.product
rngseeds = list(range(0, 10))

monomial_prior_spec = (
    ((0, 1), None), 
    ((1, 0), None), 
    ((1, 2), None), 
    ((1, 3), None),
    ((1, 3), (1, 0)),
    ((3, 1), (1, 1)),
    ((1, 1), (0, 0)),
    ((2, 1), (2, 2)),
    ((2, 3), (0, 4)),
    ((1, 1), (1, 1)),
    ((4, 1), (6, 3)),
    ((3, 1), (3, 2))
)


ALL_CONFIG_VALUES_SGLD = {
    "sampler_type": "sgld",
    "num_training_data": 1000, 
    "sigma": 0.5, 
    "prior_sigma": 1.0, 
    "epsilon": [1e-4, 5e-4], 
    "monomial_prior_spec": monomial_prior_spec,
    "num_steps": 10000, 
    "seed": rngseeds, 
}

ALL_CONFIG_VALUES_MCMC = {
    "sampler_type": "mcmc",
    "num_training_data": 1000, 
    "sigma": 0.5, 
    "prior_sigma": 1.0, 
    "monomial_prior_spec": monomial_prior_spec,
    "num_steps": 10000, 
    "seed": rngseeds,    
}

def _create_config_list(all_config_vals): 
    # assume none of the config values are suppose to be list. 
    to_be_expanded = [key for key, val in all_config_vals.items() if isinstance(val, list)]
    return expand_dictionary(all_config_vals, to_be_expanded)


def run_one(config, outputdir=None, no_plot=True, expt_name=None):
    config.update({"outputdir": outputdir, "no_plot": no_plot})
    run = ex.run(config_updates=config,meta_info={"expt_name": expt_name})
    print(run.config["outputdir"])
    return

if __name__ == "__main__":
    MAX_NUM_WORKERS = 6 # Maximum of parallel processes to run
    NOPLOT = True
    EXPT_NAME = "myexptname"
    
    OUTPUTDIR = f"./outputs/toypotentials/{EXPT_NAME}"
    config_list = _create_config_list(ALL_CONFIG_VALUES_SGLD) + _create_config_list(ALL_CONFIG_VALUES_MCMC)
    print(f"Total experiments count: {len(config_list)}")
    exec_fn = functools.partial(run_one, outputdir=OUTPUTDIR, no_plot=NOPLOT, expt_name=EXPT_NAME)
    with Pool(processes=MAX_NUM_WORKERS) as pool:
        pool.map(exec_fn, config_list)