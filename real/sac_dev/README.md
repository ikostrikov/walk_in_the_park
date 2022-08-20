# Soft Actor-Critic (SAC)

Implementation of soft actor-critic along with some bells and whistles:
https://arxiv.org/abs/1801.01290

## Getting Started

Install requirements:

`pip install -r requirements.txt`

and it should be good to go.

## Training Models

To train a policy, run the following command:

``python run.py --env HalfCheetah-v2 --max_samples 2000000 --visualize``

- `HalfCheetah-v2` can be replaced with other environments.
- `--max_samples` specifies the maximum number of training samples, after which training will terminate.
- `--visualize` enables visualization, and rendering can be disabled by removing the flag.
- The log and model will be saved to the `output/` directory by default. But the output directory can also be specified with `--output_dir [output-directory]`.

To train with multiprocessing, run with MPI:
``mpiexec -n 4 python run.py --env HalfCheetah-v2 --max_samples 2000000``

- `-n` specifies the number of workers used to parallelize training.

## Loading Models

To load a trained model, run the following command:

``python run.py --test --env HalfCheetah-v2 --model_file data/policies/halfcheetah_awr.ckpt --visualize``

- `--model_file` specifies the `.ckpt` file that contains the trained model.

## Code

- `learning/rl_agent.py` is the base agent class, and implements basic RL functionalties.
- `learning/sac_agent.py` implements the SAC algorithm. The `_update()` method performs one update iteration.
- `sac_configs.py` can be used to specify hyperparameters for the different environments. If no configurations are specified for a particular environment, than the algorithm will use the default hyperparameter settings in `learning/sac_agent.py`.
