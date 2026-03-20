# Numerical Experiments: Quantum Work Extraction from QHMM
This repository contains the Python script used to perform the numerical experiments detailed in Section 9 of the paper "Reinforcement learning for quantum processes with memory". The code simulates an agent extracting thermodynamic work from a quantum hidden Markov model (QHMM).

## Overview
The script computes the optimal work extraction policy via backward dynamic programming and evaluates the cumulative work dissipation over multiple episodes. It leverages a discretized belief simplex and continuous target purity parameters to model the physical work-extraction protocol.

## Requirements
1. numpy
2. scipy
3. matplotlib
4. tqdm (to keep track of the process for multi-processing)

## Usage
The script is designed to be executed directly from the command line. Because it utilizes Python's multiprocessing module, it will automatically detect and use the available CPU cores on your machine to parallelize the simulation repetitions
Run the script using:

Bash QHMM_cumulative_dissipation.py

## Parameters
The core simulation parameters are hardcoded in the if __name__ == '__main__': block. 
You can modify these to test different horizons or granularities:

L: Interaction horizon (default: 5)
N: Number of discretized belief states (default: 51)
M: Number of discrete actions/measurements (default: 11)
K: Number of iterations/episodes (default: 5000)
R: Number of independent repetitions for averaging (default: 200)
p: Transition probability parameter (default: 0.9)
r: Initial overlap between quantum states (default: 0.2)

## Outputs 
Upon completion, the script saves a numpy file (.npy) containing a stacked matrix of the dissipation data across all repetitions. 
The default output filename is formatted as: dissipation_data_without_exploration_L5_M11_K5000_R200.npy.
This data can be subsequently loaded to reproduce the cumulative dissipation plots shown in the paper's numerical results.
