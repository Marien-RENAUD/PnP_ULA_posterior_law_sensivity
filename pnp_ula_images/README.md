# PnP_ULA_posterior_law_sensivity

## Environment
To set up the environment, use the command :
```
conda env create -f requirement.yml 
```

## Experiment for gray-scale images

To run a PnP-ULA sampling for gray-scale images, use the DnCNN architecture, use, for example, the command :
```
python pnpula_experiment.py --n_iter 10000 --img 'simpson_nb512.png' --path_result 'gray/simpson' --model_name "layers_models_50epochs/Layers17" --gpu_number 0
```
Results are save in the folder 'results/gray/simpson' for this example. The file 'simpson_nb512_sigma1_s5_sampling.pth' save the sampling during the process and the file 'simpson_nb512_sigma1_s5_result.npy' gives the all experiment parameters and results (sampling, MMSE, PSNR, SIM...).

## Experiment for color images

To run a PnP-ULA sampling for color RGB images, use the Drunet architecture, use, for example, the command :
```
python pnpula_experiment_rgb.py --n_iter 10000 --img 'woman01.jpg' --path_result 'color/woman01' --gpu_number 1
```
Results are save in the folder 'results\woman01' for this example. The file 'woman01_sigma1_s5_sampling.pth' save the sampling during the process and the file 'woman01_sigma1_s5_result.npy' gives the all experiment parameters and results (sampling, MMSE, PSNR, SIM...).

## Result analysis
result_analysis.py and result_analysis_rgb.py are use to analysis the result give by the different Markov Chain and compute the posterior-$L_2$ distance
between denoisers and the Wasserstein distance between sampling. result_analysis_2.py and result_analysis_2_rgb.py are only here to generate graphics of the result.

## Other folders
Folder model and models conatain the architecture used in our experiments. Pretrained_model contains the weight of pretrained models, in the file layers_models_50epochs, weights for DnCNN with different number of layers (between 1 and 17) and the other file '.pth' contains Drunet weights learnt with different dataset on 256*256 RGB images. And the folder images contains the images used in our experiments.

