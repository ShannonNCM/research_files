#import libraries
import numpy as np
from ase.io import read, write
from ase import Atoms
import torch
import yaml
import json
import matplotlib.pyplot as plt
import pandas as pd
import os#
import sys

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
        last_epoch = int(df[x].max())
        filename = f'img_res/{model_name}_{label}_loss.pdf'
        plt.savefig(filename)
        plt.show()


#function for plotting the reference vs predicted energies
def plot_energy_comparison(df,x,y):
    plt.scatter(df[x],df[y])
    plt.plot([df[x].min(), df[x].max()], [df[y].min(), df[y].max()], 'r--')
    plt.xlabel('ref energy (eV)')
    plt.ylabel('predicted (eV)')
    plt.xlim(df[x].min(), df[x].max())
    plt.ylim(df[y].min(), df[y].max())
    plt.autoscale()
    plt.show()    