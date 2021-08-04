# NeuralNetworkSolverforShockWaveChaos
This repo contains the code for the project: Neural Network Solver for Shock Wave Chaose

# Steps for running the code:

1. **KFR Data Generator** produces the "ground truth" values at the required parameter values. The function is of the form #DataGenerator(Alpha, Beta, L, t, N, Amp, ext). Where ext is the extension of the data file. By default, it saves as .mat file in the folder DataGenerator. The file name is of the form: amp_{}alpha_{} e.g amp_0alpha_5.1
2. **PINN_KFR** is the main code for the neural network solver. Change the name of param_value (amp_{}alpha_{}) to load and train for different alpha values. Hyperparameters can also easily be changed.
3. If hyperparameters are to be tuned, go to **PINN_KFR_opt** code.
4. Rest of the files are there to support these main codes.
