for i in 17
#i is the number of layers of the denoiser can be between an integer between 1 and 17
do
  python pnpula_experiment.py --img 'simpson_nb512.png' --num_of_layers $i --gpu_number 0 --model_name "layers_models_50epochs/Layers$i" --n_iter 100000 --path_result "result_gray/result_layers_$i"
done