# PnP-ULA on images

## Environment
To set up the environment, use the command :
```
conda env create -f environment.yml 
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

## File structure
```
pnp_ula_images
  |-images : list of images used in our experiments.
  |-models
    |-model_dncnn : architecture DnCNN from Ernest K. Ryu, Jialin Liu, Sicheng Wang, Xiaohan Chen, Zhangyang Wang, and Wotao Yin. Plug-
and-Play Methods Provably Converge with Properly Trained Denoisers. In (ICML) International
Conference on Machine Learning, may 2019.
    |-model_drunet : architecture DRUNet from Kai Zhang, Yawei Li, Wangmeng Zuo, Lei Zhang, Luc Van Gool, and Radu Timofte. Plug-and-play
image restoration with deep denoiser prior. IEEE Transactions on Pattern Analysis and Machine
Intelligence, 44(10):6360–6376, 2021.
  |-Pretrained_models : weights of Pretrained model.
    |-layers_models_50epochs : weights of models DnCNN trained with a different number of epochs on CBSD68 with 50 epochs.
    |-celebA(woman faces).pth : weights of DruNER trained on CelebA dataset only with woman faces.
  |-results
    |-result_gray: result of the PnP-ULA on various gray-scale images included cameraman, castle, goldhill, simpson_nb512.
    |-result_gray: result of the PnP-ULA on various RGB color images included castle, woman01, woman02, woman03.
  |-environment.yml: setting of python environment to run the code.
  |-pnp_ula_experiment.sh : bash file to run experiment of PnP-ULA on gray-scale images.
  |-pnp_ula_experiment_rgb.sh : bash file to run experiment of PnP-ULA on color images.
  |-pnpula_experiment.py : python code of PnP-ULA on gray-scale images.
  |-pnpula_experiment_rgb.py : python code of PnP-ULA on color images.
  |-README.md
  |-utils.py : python file with useful function to run the process.
```


