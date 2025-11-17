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

#function to calculate the errors
def errors(df,file):
    rmse = np.sqrt(mean_squared_error(df['REF_e/atom_meV'],df['MACE_e/atom_meV']))
    mae = mean_absolute_error(df['REF_e/atom_meV'],df['MACE_e/atom_meV'])
    r2 = r2_score(df['REF_e/atom_meV'],df['MACE_e/atom_meV'])
    results = pd.DataFrame({'error':file, 'rmse':[rmse], 'mae':[mae], 'r2':r2})
    return results


#### PLOTTING FUNCTIONS 

#function for plotting the mean absolute error for forces and energy
def plot_mae(dataframe, x, y):
    for i, col in enumerate(y):
        colors=plt.cm.tab10.colors
        plt.scatter(dataframe[x], dataframe[col], color=colors[i %len(colors)], label=f'{col} (eV)')
        plt.xlabel(x)
        plt.ylabel('mae')
        plt.legend()
        plt.autoscale
        plt.show()


#function to plot the training and validation errors as funcionts of epochs
def plot_loss(dataframes, x, y, model_name):

    colors=plt.cm.tab10.colors

    #plotting the combined graph
    for i, (df, label) in enumerate(dataframes):
        plt.scatter(df[x], df[y], color=colors[i % len(colors)], label=label)
        plt.xlabel(x)
        plt.ylabel(y)
        plt.legend()
        plt.autoscale()
    plt.show()
    
    #plotting one for each dataframe
    for i, (df, label) in enumerate(dataframes):
        plt.scatter(df[x], df[y], color=colors[i % len(colors)], label=label)
        plt.xlabel(x)
        plt.ylabel(y)
        plt.legend()
        plt.autoscale()
        #last_epoch = int(df[x].max())
        filename = f'img_res/{model_name}_{label}_loss.pdf'
        plt.savefig(filename)
        plt.show()


#function for plotting the reference vs predicted energies
def plot_energy_comparison(df,x,y):
    plt.scatter(df[x],df[y])
    plt.plot([df[x].min(), df[x].max()], [df[y].min(), df[y].max()], 'r--')
    plt.xlabel('ref energy per atom (meV)')
    plt.ylabel('predicted energy per atom (meV)')
    plt.xlim(df[x].min(), df[x].max())
    plt.ylim(df[y].min(), df[y].max())
    plt.autoscale()
    plt.show()    