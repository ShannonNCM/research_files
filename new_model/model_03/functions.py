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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

#function to set the names of the tags in the xyz file
def file_format(input_file, output_file):
    with open(input_file, "r") as f:
        lines = f.readlines()

    new_lines = []
    i=0

    while i < len(lines):
        n_atom_line = lines[i]
        new_lines.append(n_atom_line)
        i += 1

        try:
            n_atoms = int(n_atom_line.strip())
        except ValueError:
            continue

        header = lines[i]

        if n_atoms == 1:
            header = re.sub(r"config_type=\S+", "config_type=IsolatedAtom", header)

        if "energy" in header:
            header = header.replace("energy=", "REF_energy=")
        if "Properties=" in header and "forces" in header:
            header = header.replace("forces", "REF_forces")
    
        new_lines.append(header)
        i += 1

        for _ in range(n_atoms):
            new_lines.append(lines[i])
            i += 1    

    # Write the fixed file
    with open(output_file, "w") as f:
        f.writelines(new_lines)


#function to read xyz file and extract information
def file_read(file_name):
    file = read(file_name, index=':')
    
    data = []
    for a in file:
        n_atoms = len(a)
        ref_e = a.info['REF_energy']
        #ref_e_atom = ref_e / n_atoms
        #ref_e_meV = ref_e*1000
        #ref_e_meV_atom = ref_e_meV / n_atoms

        data.append({'n_atoms':n_atoms, 'REF_energy':ref_e#, 'REF_e/atom_eV':ref_e_atom, 'ref_energy_meV':ref_e_meV, 'REF_e/atom_meV':ref_e_meV_atom
                     })

    df = pd.DataFrame(data)
    return df


#function for extracting information from the evaluation of the model
def eval_read(model_name, file):
    file = read(f'test_res/{model_name}_{file}.xyz', index=':')
    
    data = []
    for a in file:
        n_atoms = len(a)
        ref_e = a.info['REF_energy']
        mace_e = a.info['MACE_energy']
        ref_e_meV = ref_e*1000
        mace_e_meV = mace_e*1000
        ref_f = a.arrays['REF_forces']
        mace_f = a.arrays['MACE_forces']
        ref_e_meV_atom = ref_e_meV / n_atoms
        mace_e_meV_atom = mace_e_meV / n_atoms

        data.append({'n_atoms':n_atoms, 'REF_energy':ref_e, 'MACE_energy':mace_e, 'ref_energy_meV':ref_e_meV, 'mace_energy_meV':mace_e_meV, 'REF_e/atom_meV':ref_e_meV_atom, 'MACE_e/atom_meV':mace_e_meV_atom, 'REF_forces':ref_f, 'MACE_forces':mace_f})

    df = pd.DataFrame(data)
    return df


#function to extract the forces
def forces(df):
    ref_f = np.concatenate(df['REF_forces'].values).flatten()
    mace_f = np.concatenate(df['MACE_forces'].values).flatten()
    new_df = pd.DataFrame({'REF_force':ref_f, 'MACE_force':mace_f})
    return new_df


#function to calculate the errors
def errors(df,file, ref, pred):
    rmse = np.sqrt(mean_squared_error(df[ref],df[pred]))
    mae = mean_absolute_error(df[ref],df[pred])
    r2 = r2_score(df[ref],df[pred])
    results = pd.DataFrame({'error':file, 'rmse':[rmse], 'mae':[mae], 'r2':r2})
    return results


#### ________________________________
#### PLOTTING FUNCTIONS 
#### ________________________________

#function for plotting the mean absolute error for forces and energy
def plot_mae(dataframe, x, y):
    colors = plt.cm.tab10.colors
    n=len(y)
    #establezco el layout de la figura
    fig, axes = plt.subplots(1,2, figsize=(4*n, 4), layout='constrained')
    for i, col in enumerate(y):
        axs = axes[i]
        axs.scatter(dataframe[x], dataframe[col], color=colors[i %len(colors)], label=f'{col} (eV)')
        axs.set_xlabel(x)
        axs.set_ylabel('mae')
        axs.legend()


#function to plot the training and validation errors as funcionts of epochs
def plot_loss(dataframes, x, y, model_name):
    colors=plt.cm.tab10.colors
    n=len(y)

    fig, axes=plt.subplots(1,2, figsize=(3*n,6), layout='constrained')
    fig.suptitle('Training and validation loss')
    for i, (df, label) in enumerate(dataframes):
        axs = axes[i]
        axs.scatter(df[x], df[y], color=colors[i % len(colors)], label=label)
        axs.set_xlabel(x)
        axs.set_ylabel(y)
        axs.legend()
        #si lo voy a guardar como pdfs separados
        #filename = f'img_res/{model_name}_{label}_loss.pdf'
        #plt.savefig(filename)

    filename = f'img_res/{model_name}_loss.pdf'
    plt.savefig(filename)


#function for plotting the reference vs predicted energies
def plot_comparison(dfs, x_cols, y_cols):
    row = len(dfs)
    cols = len(x_cols)

    fig, axes=plt.subplots(row, cols, figsize=(5*cols, 5*row), 
                           layout='constrained')
    if row == 1:
        axes = np.array([axes])
    if cols == 1:
        axes = axes.reshape(row,1)
    
    for i, df in enumerate(dfs):
        for j, (x,y) in enumerate(zip(x_cols, y_cols)):
            axs = axes[i][j]
            axs.scatter(df[x], df[y])
            axs.plot([df[x].min(), df[x].max()], [df[y].min(), df[y].max()], 'r--')
            axs.set_xlabel(f'{x}')
            axs.set_ylabel(f'{y}')