# PnP-ULA on images

## Environment
To set up the environment, use the command :
```
conda env create -f requirement.yml 
```

## Experiment for gray-scale images

To run a PnP-ULA sampling for 256*256 gray-scale images, a demonstration is provided by using the command :
```
bash pnp_ula_experiment.sh
```
Results are save in the folder 'results/result_gray'. For each experiment a folder is create with a .npy file gives the all experiment parameters and results (sampling, MMSE, PSNR, SIM...). Different visualization are also generated.

## Experiment for color images

To run a PnP-ULA sampling for 256*256 color images, a demonstration is provided by using the command :
```
bash pnp_ula_experiment_rgb.sh
```
Results are save in the folder 'results/result_rgb'. For each experiment a folder is create with a .npy file gives the all experiment parameters and results (sampling, MMSE, PSNR, SIM...). Different visualization are also generated.

## Other folders
The folder 'images' contains all the images used in our experiments. 'models' contains the different Neural Network architecture. 'Pretrained_models' contains the weights of pretrained models used in our experiments. 'pnpula_experiment.py', 'pnpula_experiment_rgb.py' and 'utils.py' contains the code to compute the PnP-ULA dynamic and generate the results.