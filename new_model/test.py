#import libraries
import numpy as np
from ase.io import read, write
from ase import Atoms
import torch
import yaml
import json
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
import re
import shutil  # Added to safely copy files between folders
os.makedirs('img_res', exist_ok=True) #creates a folder to store the loss graphs
os.makedirs('test_res', exist_ok=True) #creates a folder to store the files of the testing of the model
import functions as f #import functions used in this notebook

#setting the loop for running
epoch_values = [1,2] # Use a list to ensure ascending order for sequential training
#epoch_values = list(range(40, 501, 20))

for epochs in epoch_values:
    ###############################################################
    # variable setup (MODIFIED TO SUPPORT SHARED CHECKPOINTS)
    ###############################################################
    name = "Fe_Si_B_260311"
    type = 'rnd_e'
    frozen_layers = 4
    device = 'cuda'
    model = "MACE-matpes-pbe-omat-ft"
    model_id = f'matpes_nofe8b4_freeze_{frozen_layers}'
    learning_rate = 1e-4
    num_epoch = epochs 
    batch_size = 10 
    seed = 123
    
    # Target folders matching your original exact naming format
    folder = f'{model_id}_{learning_rate}_{num_epoch}_{batch_size}_{type}'
    path = f'model_{name}/freeze_training/{folder}'
    ### arregle esto porque si no, no hace bien lo de entrenar desde el ulitmo paso
    model_name = f'model_{type}_{model_id}_lr{learning_rate}_continuous_{batch_size}'
    ### tambien esto agregue
    eval_model_name = f'model_{type}_{model_id}_lr{learning_rate}_{num_epoch}_{batch_size}'
    
    os.makedirs(path, exist_ok=True)
    os.makedirs(f'{path}/img_res', exist_ok=True)
    os.makedirs(f'{path}/test_res', exist_ok=True)

    # NEW: Define a single shared workspace directory so MACE can see previous checkpoints
    shared_folder = f'{model_id}_{learning_rate}_continuous_{batch_size}_{type}'
    shared_path = f'model_{name}/freeze_training/{shared_folder}'
    os.makedirs(f"{shared_path}/checkpoints", exist_ok=True)

    structure = 'Fe8B4'
    train_file = f"model_{name}/train_{type}_no_{structure}.xyz"
    test_file = f"model_{name}/test_{type}_no_{structure}.xyz"

    ###############################################################
    # writing the yml file (MODIFIED)
    ###############################################################
    config = {
        'foundation_model': f'{model}.model',
        'multiheads_finetuning': False,
        'freeze': frozen_layers,
        "name": model_name,                  # Kept original
        "model_dir": shared_path,            # Diverted to shared workspace
        "log_dir": f"{shared_path}/log",      # Diverted to shared workspace
        "checkpoints_dir": f"{shared_path}/checkpoints", # Diverted to shared workspace
        "results_dir": f"{shared_path}/results", # Diverted to shared workspace
        "train_file": train_file,
        "valid_fraction": 0.1,
        "test_file": test_file,
        "energy_key": "REF_energy",
        "forces_key": "REF_forces",
        "batch_size": batch_size,
        "max_num_epochs": num_epoch,
        "lr": learning_rate,
        "device": device,
        "seed": seed,
        
        # This tells MACE to automatically resume training if files exist in checkpoints_dir
        "restart_latest": True 
    }
    with open(f"model_{name}/config_freeze.yml", "w") as f_yml:
        yaml.dump(config, f_yml, sort_keys=False)


    ###############################################################
    # Perform training (UNTOUCHED)
    ###############################################################
    import warnings
    warnings.filterwarnings('ignore')
    from mace.cli.run_train import main as mace_run_train_main
    import sys
    import logging

    def train_mace(config_file_path):
        logging.getLogger().handlers.clear()
        sys.argv = ['program', '--config', config_file_path]
        mace_run_train_main()

    train_mace(f'model_{name}/config_freeze.yml') 

    ###############################################################
    # POST-TRAINING FILE DISTRIBUTION (NEW ADDITION)
    # This automatically syncs MACE's shared workspace outputs back 
    # to your target folder format so the rest of your script runs perfectly.
    ###############################################################
    os.makedirs(f"{path}/results", exist_ok=True)
    #os.makedirs(f"{path}/checkpoints", exist_ok=True)
    
    # 1. Copy the training log text file
    shared_results_file = f'{shared_path}/results/{model_name}_run-{seed}_train.txt'
    target_results_file = f'{path}/results/{eval_model_name}_run-{seed}_train.txt'
    if os.path.exists(shared_results_file):
        shutil.copy(shared_results_file, target_results_file)
        
    # 2. Copy the newly compiled model file
    shared_compiled_model = f'{shared_path}/{model_name}.model'
    target_compiled_model = f'{path}/{eval_model_name}.model'
    if os.path.exists(shared_compiled_model):
        shutil.copy(shared_compiled_model, target_compiled_model)

    ###############################################################
    # reading the information on the results file (UNTOUCHED)
    ###############################################################
    results = f'{path}/results/{eval_model_name}_run-{seed}_train.txt' 
    data = [] 
    with open(results, 'r') as f_res:
        for line in f_res:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    ###############################################################
    # plotting the results (UNTOUCHED)
    ###############################################################
    df = pd.DataFrame(data)
    train_df = df[df['mode']=='opt'].copy()
    val_df = df[df['mode']=='eval'].copy().dropna()
    train_df = train_df.groupby('epoch', as_index=False)['loss'].mean()
    train_df['epoch'] += 1
    val_df['epoch'] += 1

    import functions as f_plots
    f_plots.plot_mae(val_df, 'epoch', ['mae_e_per_atom','mae_f'])
    f_plots.plot_loss([(train_df, 'train'),(val_df,'val')],'epoch','loss',model_name, path)

    ###############################################################
    # evaluation (UNTOUCHED)
    ###############################################################
    from mace.cli.eval_configs import main as mace_eval_configs_main
    import sys

    def eval_mace(model_path, configs, output, device=device):
        if device == 'cuda':
            torch.cuda.empty_cache()
            batch_size = '1'
        else:
            batch_size = '10' # added safety catch for non-cuda fallback structures

        sys.argv=['program', '--configs', configs, '--model', model_path, '--output', output, '--device', device, '--batch_size', batch_size]
        mace_eval_configs_main()

    eval_mace(model_path=f'{path}/{model_name}.model',
            configs=train_file,
            output=f'{path}/test_res/{model_name}_train.xyz')

    eval_mace(model_path=f'{path}/{model_name}.model',
            configs=test_file,
            output=f'{path}/test_res/{model_name}_test.xyz')

    test_df = f_plots.eval_read(model_name, 'test', path)
    train_df = f_plots.eval_read(model_name, 'train', path)
    train_df = train_df[3:] 
    test_df1 = test_df[['config', 'n_atoms', 'REF_energy', 'MACE_energy', 'REF_e/atom_meV', 'MACE_e/atom_meV']].copy()
    train_df1 = train_df[['config', 'n_atoms', 'REF_energy', 'MACE_energy', 'REF_e/atom_meV', 'MACE_e/atom_meV']].copy()
    train_df2 = f_plots.forces(train_df)
    test_df2 = f_plots.forces(test_df)

    f_plots.plot_comparison([test_df1, train_df1], 
                    ['REF_energy', 'REF_e/atom_meV'], ['MACE_energy','MACE_e/atom_meV'], ['Test data', 'Train data'], 'energy', model_name, path)

    f_plots.plot_comparison([test_df2, train_df2], 
                    ['REF_force'], ['MACE_force'], ['Test data', 'Train data'], 'forces', model_name, path)

    ref = 'REF_e/atom_meV'
    pred = 'MACE_e/atom_meV'
    test_error = f_plots.errors(test_df, 'test_energy', ref, pred)
    train_error = f_plots.errors(train_df, 'train_energy', ref, pred)
    e_errors = pd.concat([test_error, train_error])
    test_config_error = f_plots.config_errors(test_df, 'test_energy', ref, pred)
    train_config_error = f_plots.config_errors(train_df, 'train_energy', ref, pred)
    config_errors = pd.concat([test_config_error, train_config_error])
    ref = 'REF_force'
    pred = 'MACE_force'
    test_error = f_plots.errors(test_df2, 'test_force', ref, pred)
    train_error = f_plots.errors(train_df2, 'train_force', ref, pred)
    f_errors = pd.concat([test_error, train_error])
    errors = pd.concat([e_errors,f_errors])

    with pd.ExcelWriter(f'{path}/test_res/{model_name}_test.xlsx') as writer:
        test_df1.to_excel(writer, sheet_name='test_predictions')
        train_df1.to_excel(writer, sheet_name='train_predictions')
        errors.to_excel(writer, sheet_name='errors')
        config_errors.to_excel(writer, sheet_name='config_errors')
