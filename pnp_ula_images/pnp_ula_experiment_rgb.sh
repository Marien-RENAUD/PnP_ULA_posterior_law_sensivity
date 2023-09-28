# For a Uniform kernel of blur
for i in 3 
#(2*i+1)*(2*i+1) is the size of the uniform kernel of blur
do
  python pnpula_experiment_rgb.py --img 'woman01.jpg' --n_iter 30000 --path_result "result_rgb/uniform_$i" --s 5 --gpu_number 0 --l $i
done

# For a Gaussian kernel of blur
# for i in 3 
# #i is the standard deviation of the Gaussian blur kernel
# do
#   python pnpula_experiment_rgb.py --img 'woman10.jpg' --n_iter 30000 --path_result "result_rgb/gaussian_si$i" --s 5 --gpu_number 0 --blur_type 'gaussian' --si $i --l 30
# done