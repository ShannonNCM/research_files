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
import random
from collections import defaultdict
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

''' 
--------------------------------
DATA ARRANGING FUNCTIONS
________________________________ 
'''

#function for splitting data
def split(type, output_file, name):
    db = read(output_file, ':')
    isolated_atoms = [atoms for atoms in db if len(atoms)==1]
    structures = [atoms for atoms in db if len(atoms)>1]
    if type == 'rnd':
        random.seed(42)
        random.shuffle(structures)
        split1 = int(0.8*len(structures))
        train_rnd = isolated_atoms+structures[:split1]
        test_rnd = structures[split1:]
        write(f'model_{name}/train_{type}.xyz', train_rnd)
        write(f'model_{name}/test_{type}.xyz', test_rnd)
    elif type == 'rnd_e':
        data = []
        for a in structures:
            num_atoms = len(a)
            toten = a.info['REF_energy']
            e_per_atom = toten/num_atoms
            data.append({#'n_atoms':num_atoms, 'REF_energy':toten, 
                        'energy/atom':e_per_atom})
        data1 = pd.DataFrame(data)
        #determinamos los minimos y maximos para los bins
        counts, bins = np.histogram(data1)
        bin_indx = np.digitize(data1['energy/atom'], bins)
        bin_dict = defaultdict(list)
        for idx, bin_id in enumerate(bin_indx):
            bin_dict[bin_id].append(structures[idx])
        random.seed(42)
        train_rnd = []
        test_rnd = []
        for bin_id, atoms_list in bin_dict.items():
            n = len(atoms_list)
            split = max(1,int(0.8 * n))
            random.shuffle(atoms_list)
            train_rnd.extend(atoms_list[:split])
            test_rnd.extend(atoms_list[split:])
        train_rnd = isolated_atoms + train_rnd
        write(f'model_{name}/train_{type}.xyz', train_rnd)
        write(f'model_{name}/test_{type}.xyz', test_rnd)
    else:
        n = len(db)
        split = int(0.8*n)
        write(f'model_{name}/train_{type}.xyz', db[:split])
        write(f'model_{name}/test_{type}.xyz', db[split:])


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
        ref_e_atom = ref_e / n_atoms
        #ref_e_meV = ref_e*1000
        #ref_e_meV_atom = ref_e_meV / n_atoms

        data.append({'n_atoms':n_atoms, 'REF_energy':ref_e, 'REF_e/atom_eV':ref_e_atom, #'ref_energy_meV':ref_e_meV, 'REF_e/atom_meV':ref_e_meV_atom
                     })

    df = pd.DataFrame(data)
    return df

#function to check the distribution of structures in the file
def dist_analysis(file_name):
    file = read(file_name, index=':')
    config_counts = {}
    for atoms in file:
        config = atoms.get_chemical_formula()
        config_counts[config] = config_counts.get(config, 0)+1
        df = pd.DataFrame(list(config_counts.items()), columns=['config', 'n_config'])
    return df


#function for extracting information from the evaluation of the model
def eval_read(model_name, file, path):
    file = read(f'{path}/test_res/{model_name}_{file}.xyz', index=':')
    
    data = []
    for a in file:
        n_atoms = len(a)
        ref_e = a.info['REF_energy']
        mace_e = a.info['MACE_energy']
        ref_e_meV = ref_e*1000
        mace_e_meV = mace_e*1000
        ref_f = a.arrays['REF_forces']*1000
        mace_f = a.arrays['MACE_forces']*1000
        ref_e_meV_atom = ref_e_meV / n_atoms
        mace_e_meV_atom = mace_e_meV / n_atoms
        config_type = a.info.get('config_type')

        data.append({'config':config_type, 'n_atoms':n_atoms, 'REF_energy':ref_e, 'MACE_energy':mace_e, 'ref_energy_meV':ref_e_meV, 'mace_energy_meV':mace_e_meV, 'REF_e/atom_meV':ref_e_meV_atom, 'MACE_e/atom_meV':mace_e_meV_atom, 'REF_forces':ref_f, 'MACE_forces':mace_f})

    df = pd.DataFrame(data)
    return df


#function to extract the forces
def forces(df):
    ref_f = np.concatenate(df['REF_forces'].values).flatten()
    mace_f = np.concatenate(df['MACE_forces'].values).flatten()
    new_df = pd.DataFrame({'REF_force':ref_f, 'MACE_force':mace_f})
    return new_df

#function to exclude specific configuration from the data
def exclude_config(file, structure, name):
    db = read(file, ':')
    include_data = [atoms for atoms in db if atoms.info.get('config_type') != structure]
    name_file = os.path.splitext(os.path.basename(file))[0]
    write(f'model_{name}/{name_file}_no_{structure}.xyz', include_data)


''' 
--------------------------------
ERROR CALCULATION FUNCTIONS
________________________________ 
'''

#functions to calculate the errors
def errors(df,file, ref, pred):
    rmse = np.sqrt(mean_squared_error(df[ref],df[pred]))
    mae = mean_absolute_error(df[ref],df[pred])
    r2 = r2_score(df[ref],df[pred])
    
    results = pd.DataFrame({'error':file, 'rmse':[rmse], 'mae':[mae], 'r2':r2})
    return results

def config_errors(df, file, ref, pred):
    mae_config = df.groupby(['config']).apply(lambda x: pd.Series({
        'error':file,
        'n_configs': len(x),
        'rmse': np.sqrt(mean_squared_error(x[ref],x[pred])),
        'mae': mean_absolute_error(x[ref],x[pred])}), include_groups=False).reset_index()
    
    return mae_config

'''
--------------------------------
PLOTTING FUNCTIONS 
________________________________
'''

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
def plot_loss(dataframes, x, y, model_name, path):
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

    filename = f'{path}/img_res/{model_name}_loss.pdf'
    plt.savefig(filename)


#function for plotting the reference vs predicted energies
def plot_comparison(dfs, x_cols, y_cols, titles, label, model_name, path):
    row = len(dfs)
    cols = len(x_cols)

    fig, axes=plt.subplots(row, cols, figsize=(5*cols, 5*row), 
                           layout='constrained')
    if row == 1:
        axes = np.array([axes])
    if cols == 1:
        axes = axes.reshape(row,1)
    
    for i, df in enumerate(dfs):
        title = titles[i] if titles else f'df_{i}'

        for j, (x,y) in enumerate(zip(x_cols, y_cols)):
            axs = axes[i][j]
            axs.scatter(df[x], df[y])
            axs.plot([df[x].min(), df[x].max()], [df[y].min(), df[y].max()], 'r--')
            axs.set_xlabel(f'{x}')
            axs.set_ylabel(f'{y}')
            axs.set_title(f'{title}')

    fig_name = f'{path}/img_res/{model_name}_{label}.pdf'
    plt.savefig(fig_name)