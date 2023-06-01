# Universal Adversarial Perturbations on Visual Odometry Systems
> Final project submission for <br>[**CS236781 — Deep Learning on Computational Accelerators**](https://vistalab-technion.github.io/cs236781)<\br>
>
> Date: September, 2022. 
## Authors

* Tal Peer : tal.peer@campus.technion.ac.il

## Environment setup

Project enviorment dependencies listed in 
`src\environment.yml`.
### Anaconda/Miniconda

From project root folder, run:
```bash
conda env create -f environment.yml
conda activate pytorch-cupy-3
```

## Datasets
The synthetic data produced by the course staff at the project assigment ./data folder.

## Structure

### Code

All code is implemented in the `src/` directory, which includes:

* `attacks/`: include different attack methods for the experiments.


    -[attack.py](src/attack.py): Contains all functions needed to run the attacks.

    -[PGD.py](src/PGD.py): Inhetired class, contains the PGD attack.
    
    -[personalize_attack.py](src/persenalize.py): Inhetired class, contains the APGD attack.

* [run_attacks_train2.py](src/run_attacks_train2.py):Same original run_attack.py, and additional function user for evaluating and saving results in a self convinent manner. Contains high level optimization scheme using 5-fold Cross-Validation and data split to train and evaluation sets.
* [loss.py](src/loss.py): Contains loss implementationfor running the experiments.
* [utils.py](src/utils.py): Contains all arguments options for running the experiments.


#### Slurm
For results checking with SLURM, run the following command:

`srun -c 2 --gres=gpu:1 --pty python src/run_attacks_train2.py --seed 42 --model-name tartanvo_1914.pkl --test-dir "VO_adv_project_train_dataset_8_frames"  --max_traj_len 5 --batch-size 1 --worker-num 1 --save_csv --attack 'personalize_attack.py' --attack_k 100.
