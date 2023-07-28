import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as PSNR
import os
import utils
from ryu_utils.utils import load_model
from models.denoiser import Denoiser
import argparse
from model.models import DnCNN
import torch.nn as nn
import ot as ot #for optimal transport computation
from sklearn.decomposition import PCA
from pytorch_fid import fid_score
import argparse
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument("--img", type=str, default='simpson_nb512', help='image to reconstruct')
parser.add_argument("--path_result", type=str, default='result_dissertation/simpson', help='image to reconstruct')
parser.add_argument("--gpu_number", type=int, default = 0, help='gpu number use')
pars = parser.parse_args()

s = 5
s1 = s/255.
s2 = (s1)**2
cuda = True
device = "cuda:"+str(pars.gpu_number)

path_result = 'results/'+pars.path_result+'/'

name = pars.img # "goldhill", "simpson_nb512", 'cameraman', 'castle'

path_model = 'Pretrained_models/layers_models_50epochs/Layers17'

dict = torch.load(path_result+name+'_sigma1_s5_sampling.pth', map_location=device)

Samples_t = dict['Samples']
Mmse_t = dict['Mmse']
Mmse2_t = dict['Mmse2']
y_t = dict['observation']
im_t = dict['ground_truth']
name_im = dict['image_name']
n_iter = dict['n_iter']

#conversion in numpy array
im = im_t.cpu().detach().numpy()
im = im[0,0,:,:]
y = y_t.cpu().detach().numpy()
y = y[0,0,:,:]
Samples, Mmse, Mmse2, Psnr_sample, SIM_sample = [], [], [], [], []
for sample in Samples_t:
    samp = sample.cpu().detach().numpy()
    Psnr_sample.append(PSNR(im, samp, data_range = 1))
    SIM_sample.append(ssim(im, samp, data_range = 1))
    Samples.append(samp)
for m in Mmse_t:
    Mmse.append(m.cpu().detach().numpy())
for m in Mmse2_t:
    Mmse2.append(m.cpu().detach().numpy())
del Samples_t
del Mmse_t 
del Mmse2_t

###
# PLOTS
###

#save the observation
y = y_t.cpu().detach().numpy()
y = y[0,0,:,:]
plt.imsave(path_result + '/observation.png', y, cmap = 'gray')

#creation of a video of the samples
writer = imageio.get_writer(os.path.join(path_result,"samples_video_"+name_im+".mp4"), fps=100)
for im_ in Samples:
    im_uint8 = np.clip(im_ * 255, 0, 255).astype(np.uint8)
    writer.append_data(im_uint8)
writer.close()

# Compute PSNR and SIM for the online MMSE
n = len(Mmse)
print(n)
PSNR_list = []
SIM_list = []
for i in range(1,n):
    PSNR_list.append(PSNR(im, np.mean(Mmse[:i], axis = 0), data_range = 1))
    SIM_list.append(ssim(im, np.mean(Mmse[:i], axis = 0), data_range = 1))

# PSNR plots
fig, ax = plt.subplots(figsize = (10,10))
ax.plot(Psnr_sample, "+")
ax.set_title("PSNR between samples and GT")
fig.savefig(path_result +"/PSNR_between_samples_and_GT_"+name_im+"n_iter{}".format(n_iter)+".png")
plt.show()

fig, ax = plt.subplots(figsize = (10,10))
ax.plot(PSNR_list, "+")
ax.set_title("PSNR between online MMSE and GT")
fig.savefig(path_result +"/PSNR_between_online_MMSE_and_GT_"+name_im+"n_iter{}".format(n_iter)+".png")
plt.show()

fig, ax = plt.subplots(figsize = (10,10))
ax.plot(SIM_sample, "+")
ax.set_title("SIM between samples and GT")
fig.savefig(path_result +"/SIM_between_samples_and_GT_"+name_im+"n_iter{}".format(n_iter)+".png")
plt.show()

fig, ax = plt.subplots(figsize = (10,10))
ax.plot(Psnr_sample, "+")
ax.set_title("SIM between online MMSE and GT")
fig.savefig(path_result +"/SIM_between_online_MMSE_and_GT_"+name_im+"n_iter{}".format(n_iter)+".png")
plt.show()

# Computation of the mean and std of the whole Markov chain
xmmse = np.mean(Mmse, axis = 0)
pmmse = PSNR(im, xmmse, data_range = 1)
smmse = ssim(im, xmmse, data_range = 1)

# Saving of the MMSE of the sample
plt.imsave(path_result + '/mmse_' + name_im + '_psnr{:.2f}_ssim{:.2f}.png'.format(pmmse, smmse), xmmse, cmap = 'gray')

psb = PSNR(im, y, data_range = 1)
ssb = ssim(im, y, data_range = 1)

# Saving of the MMSE compare to the original and observation
fig = plt.figure(figsize = (10, 10))
ax1 = fig.add_subplot(1,3,1)
ax1.imshow(xmmse, cmap = 'gray')
ax1.axis('off')
ax1.set_title("MMSE (PSNR={:.2f}/SSIM={:.2f})".format(pmmse, smmse))
ax2 = fig.add_subplot(1,3,2)
ax2.imshow(im, cmap = 'gray')
ax2.axis('off')
ax2.set_title("GT")
ax3 = fig.add_subplot(1,3,3)
ax3.imshow(y, cmap = 'gray')
ax3.axis('off')
ax3.set_title("Obs (PSNR={:.2f}/SSIM={:.2f})".format(psb, ssb))
fig.savefig(path_result+'/MMSE_and_Originale_and_Observation_'+name_im+'n_iter{}'.format(n_iter)+'.png')
plt.show()

# Computation of the std of the Markov chain
xmmse2 = np.mean(Mmse2, axis = 0)
var = xmmse2 - xmmse**2
var = var*(var>=0) + 0*(var<0)
std = np.sqrt(var)

# Saving of the standard deviation and the difference between MMSE and Ground-Truth (GT)
fig = plt.figure(figsize = (10, 5))
ax1 = fig.add_subplot(1,2,1)
ax1.imshow(std, cmap = 'gray')
ax1.axis('off')
ax1.set_title("Std of the Markov Chain")
ax2 = fig.add_subplot(1,2,2)
ax2.imshow(np.abs(im-xmmse), cmap = 'gray')
ax2.axis('off')
ax2.set_title("Diff MMSE-GT")
fig.savefig(path_result+'/Std_of_the_Markov_Chain_'+name_im+'n_iter{}'.format(n_iter)+'.png')
plt.show()

# Saving of the Fourier transforme of the standard deviation, to detect possible artecfact of sampling
plt.imsave(path_result+"/Fourier_transform_std_MC_"+name_im+'n_iter{}'.format(n_iter)+".png",np.fft.fftshift(np.log(np.abs(np.fft.fft2(std))+1e-10)))














































###
# FID computation
###

# copy images
# for r in range(11):
#     os.makedirs(path_result+"result_ex"+str(r)+'/Samples'+name+'/', exist_ok = True)
#     im_list = torch.load(path_result+"result_ex"+str(r)+"/"+"Samples_"+name+"_sigma1_s5.pt", map_location=device)
#     print('A sample of {} images of the posteriori dist'.format(len(im_list)))

#     for i,im in tqdm(enumerate(im_list)):
#         im_ = im.cpu().detach().numpy()
#         plt.imsave(path_result+"result_ex"+str(r)+'/Samples'+name+'/'+str(i)+'.png',im_, cmap = 'gray')

# r0 = 0
# Dist_FID = [0]

# for r1 in range(1,11):

#     fid_value = fid_score.calculate_fid_given_paths([path_result+"result_ex"+str(r0)+'/Samples'+name+'/', path_result+"result_ex"+str(r1)+'/Samples'+name+'/'],
#                                                         batch_size=50,
#                                                         device=device,
#                                                         dims=2048)
#     Dist_FID.append(fid_value)

# print(Dist_FID)
# np.save(path_result+"result_"+name+"/Dist_FID", np.array(Dist_FID))


# PSNR = []

# for i in list_index:
#     specific_path = path_result+'/result_ex'+str(i)
#     dir = os.listdir(specific_path)
#     for file in dir:
#         if file[:5+len(name)] =='mmse_'+name:
#             txt = file.split('_')
#             PSNR.append(float(txt[4][4:]))


# fig = plt.figure(figsize = (10, 5))
# plt.plot(list_index,PSNR)
# fig.savefig(path_result+name+'PSNR_MMSE.png')

# ###
# # Compute posterior L2 distance between denoisers
# ###

# def load_denoiser(path, change_name = True, num_of_layers = 17):
#     """
#     Load the denoiser train until the epoch r
#     """
#     net = DnCNN(channels=1, num_of_layers=num_of_layers)
#     model = nn.DataParallel(net,device_ids=[int(str(device)[-1])],output_device=device)#.cuda()

#     dicti = torch.load(path, map_location=torch.device(device if torch.cuda.is_available() else "cpu"))
#     dicti_ = {}
#     for keys, values in dicti.items():
#         dicti_["module."+keys] = values.to(device)
#     if change_name:
#         model.load_state_dict(dicti_)
#     else:
#         model.load_state_dict(dicti)
#     model.eval()

#     def torch_denoiser(x,model):
#         """pytorch_denoiser
#         Inputs:
#             xtilde      noisy tensor
#             model       pytorch denoising model
    
#         Output:
#             x           denoised tensor
#         """
#         # denoise
#         with torch.no_grad():
#             #xtorch = xtilde.unsqueeze(0).unsqueeze(0)
#             r = model(x)
#             #r = np.reshape(r, -1)
#             x_ = x - r
#             out = torch.squeeze(x_)
#         return out

#     Ds = lambda x : torch_denoiser(x, model)
#     return Ds

# Metric_list = []
# Norm_1_list = []
# Norm_2_list = []

# r = ref_ind #the sampling which is taken as a reference
# im_list = torch.load(path_result+"result_"+str(r)+"/"+"Samples_"+name+"_sigma1_s5.pt", map_location=device)
# print('A sample of {} images of the posteriori dist'.format(len(im_list)))

# for i in list_index:
#     #load models pretrain
#     path = path_model+str(r)+".pth"
#     Ds_1 = load_denoiser(path, change_name = True, num_of_layers=r)
#     path = path_model+str(i)+".pth"
#     Ds_2 = load_denoiser(path, change_name = True, num_of_layers=i)

#     #compute the posterior L2 distance between denoisers
#     Metric = 0
#     Norm_1 = 0
#     Norm_2 = 0
#     for im in tqdm(im_list):
#         im = im.view(1,1,256,256)
#         im_den1 = Ds_1(im)
#         im_den2 = Ds_2(im)
#         u = (im_den1 - im_den2)**2
#         Metric += torch.sum(u).cpu().detach().numpy()
#         Norm_1 += torch.sum(im_den1**2).cpu().detach().numpy()
#         Norm_2 += torch.sum(im_den2**2).cpu().detach().numpy()

#     Metric = np.sqrt(Metric / len(im_list)); Metric_list.append(Metric)
#     Norm_1 = np.sqrt(Norm_1 / len(im_list)); Norm_1_list.append(Norm_1)
#     Norm_2 = np.sqrt(Norm_2 / len(im_list)); Norm_2_list.append(Norm_2)

# print(Metric_list)
# print(Norm_1_list)
# print(Norm_2_list)

# fig = plt.figure(figsize = (10, 5))
# plt.plot(list_index,Metric_list)
# fig.savefig(path_result+name+'Distance_denoisers.png')
# np.save(path_result+"Distance_d1"+name, np.array(Metric_list))

# # ###
# # # Computation of the Wasserstein distance between sampling
# # ###

# Dist_wass = []

# im_list_1 = []
# for im in im_list:
#     im_list_1.append(im.cpu().detach().numpy())
# im_list_ = np.array(im_list_1)
# im_list_1 = im_list_.reshape(1000, 256*256)

# for r0 in tqdm(list_index):
#     im_list = torch.load(path_result+"result_"+str(r0)+"/" +"Samples_"+name+"_sigma1_s5.pt", map_location = device)
#     im_list_2 = []
#     for im in im_list:
#         im_list_2.append(im.cpu().detach().numpy())
#     im_list_2 = np.array(im_list_2)
#     im_list_2 = im_list_2.reshape(1000, 256*256)
#     M = ot.dist(im_list_1, im_list_2, p = 1)
#     dist = ot.emd2(a = [], b = [], M = M)
#     Dist_wass.append(dist)

# print(Dist_wass)

# fig = plt.figure(figsize = (10, 5))
# plt.plot(list_index,Dist_wass)
# fig.savefig(path_result+name+'Distance_wasserstein.png')
# np.save(path_result+"Distance_wasserstein"+name, np.array(Dist_wass))


# def correlation(x, y):
#     return np.mean((x-np.mean(x))*(y-np.mean(y)))/(np.std(x)*np.std(y))

# corr = correlation(Dist_wass, Metric_list)
# print(corr)

# fig, ax1 = plt.subplots(figsize = (10,6))
# color = 'tab:orange'
# ax1.plot(list_index,Dist_wass, color = color, marker = 'o', alpha = 1)
# ax1.set_ylabel('Wasserstein distance between samplings', color=color, fontsize=15)
# ax1.tick_params(axis='y', labelcolor=color)
# ax2 = ax1.twinx()
# color = 'tab:blue'
# ax2.plot(list_index,Metric_list, color = color, marker = 'o', alpha = 0.8)
# ax2.set_ylabel('Distance between operators', color=color, fontsize=15)
# ax2.tick_params(axis='y', labelcolor=color)
# ax1.set_xlabel('Variance of the gaussian kernel of blur, correlation = {0:.4g}'.format(corr), fontsize=15)
# fig.tight_layout()
# plt.legend()
# fig.savefig(path_result+name+'Comparaison.png')



###
# PCA
###

# for r in tqdm(list_index):
#     im_list = torch.load(path_result+"result_ex"+str(r)+"/Samples_"+name+"_sigma1_s5.pt", map_location=device)
#     N = len(im_list)
#     im_l = np.zeros((N, 256, 256))
#     for i, im in enumerate(im_list):
#         im_l[i] = im.cpu().detach().numpy()

#     ims = im_l.reshape(N, 256*256)


#     ims_PCA = PCA(n_components=2).fit_transform(ims)

#     fig = plt.figure(figsize = (8, 5))
#     ax = fig.add_subplot()
#     sctt = ax.scatter(ims_PCA.T[0], ims_PCA.T[1], c = np.arange(N))
#     fig.colorbar(sctt, ax = ax)
#     fig.savefig(path_result+"result_ex"+str(r)+'/'+name+'PCA_samples.png')


#     ims_PCA = PCA(n_components=3).fit_transform(ims)

#     fig = plt.figure(figsize = (8, 5))
#     ax = fig.add_subplot(projection='3d')
#     sctt = ax.scatter(ims_PCA.T[0], ims_PCA.T[1], ims_PCA.T[2], c = np.arange(N))
#     fig.colorbar(sctt, ax = ax)
#     fig.savefig(path_result+"result_ex"+str(r)+'/'+name+'PCA3D_samples.png')

###
# Computation of the distance betwwen MMSE and STD
###

# dist_mmse = []
# r = ref_ind
# mmse_list = torch.load(path_result+"result_ex"+str(r)+"/"+"Mmse_"+name+"_sigma1_s5.pt", map_location=device)
# xmmse = mmse_list[-1].cpu().detach().numpy()

# Mmse = [] #as previous paper, take the mean of the MMSE
# for m in mmse_list:
#     Mmse.append(m.cpu().detach().numpy())
# xmmse = np.mean(Mmse, axis = 0)

# # #image
# # path_img = 'images/'
# # img = name + '.png'
# # name_im = img.split('.')[0]
# # im_total = plt.imread(path_img+img)
# # if img == "castle.png" or img == "duck.png":
# #     im_total = np.mean(im_total, axis = 2)
# # if img == "duck.png":
# #     im = im_total[400:656,550:806]
# # else:
# #     im = im_total[100:356,0:256]
# # #image normalization
# # im = im/np.max(im)
# # #computation of PSNR and MMSE
# # pmmse = PSNR(im, xmmse, data_range = 1)
# # smmse = ssim(im, xmmse, data_range = 1)
# # print("PSNR : ",pmmse)
# # print("SSIM : ",smmse)
# # plt.imsave(path_result+"result_ex"+str(r)+"/"+"MMSE.png", xmmse, cmap = 'gray')

# for i in tqdm(list_index):

#     mmse_list_i = torch.load(path_result+"result_ex"+str(i)+"/"+"Mmse_"+name+"_sigma1_s5.pt", map_location=device)
#     # xmmse_i = mmse_list_i[-1].cpu().detach().numpy()
    
#     Mmse_i = [] #as previous paper, take the mean of the MMSE
#     for m in mmse_list_i:
#         Mmse_i.append(m.cpu().detach().numpy())
#     xmmse_i = np.mean(Mmse_i, axis = 0)

#     dist_mmse.append(np.sum((xmmse_i - xmmse)**2))

# fig = plt.figure(figsize = (10, 5))
# plt.plot(list_index,dist_mmse)
# fig.savefig(path_result+name+'Distance_MMSE.png')



# dist_std = []
# r = ref_ind
# mmse2_list = torch.load(path_result+"result_ex"+str(r)+"/"+"Mmse2_"+name+"_sigma1_s5.pt", map_location=device)

# Mmse2 = [] #as previous paper, take the mean of the MMSE
# for m in mmse2_list:
#     Mmse2.append(m.cpu().detach().numpy())
# xmmse2 = np.mean(Mmse2, axis = 0)

# var = xmmse2 - xmmse**2
# var = var*(var>=0) + 0*(var<0)
# std = np.sqrt(var)

# for i in list_index:

#     Mmmse_i = torch.load(path_result+"result_ex"+str(i)+"/"+"Mmse_"+name+"_sigma1_s5.pt", map_location=device)
#     Mmmse2_i = torch.load(path_result+"result_ex"+str(i)+"/"+"Mmse2_"+name+"_sigma1_s5.pt", map_location=device)

#     Mmse_i_list = [] #as previous paper, take the mean of the MMSE
#     Mmse2_i_list = []
#     for m in Mmmse_i:
#         Mmse_i_list.append(m.cpu().detach().numpy())
#     xmmse_i = np.mean(Mmse_i_list, axis = 0)
#     for m in Mmmse2_i:
#         Mmse2_i_list.append(m.cpu().detach().numpy())
#     xmmse2_i = np.mean(Mmse2_i_list, axis = 0)
#     var_i = xmmse2_i - xmmse_i**2
#     var_i = var_i*(var_i>=0) + 0*(var_i<0)
#     std_i = np.sqrt(var_i)

#     dist_std.append(np.sum((std_i - std)**2))

# fig = plt.figure(figsize = (10, 5))
# plt.plot(list_index,dist_std)
# fig.savefig(path_result+name+'Distance_STD.png')


###
# Typicall Wasserstein distance for two sampling of the same distribution
###

# Name = ["duck","castle","simpson_nb512","goldhill","cameraman","02","03","04","05","06","07","09","10","11","12"]

# Dist_typ = []

# for name in tqdm(Name):

#     r = ref_ind
#     im_list = torch.load(path_result+"result_"+str(r)+"/"+"Samples_"+name+"_sigma1_s5.pt", map_location=device)

#     Im_list = np.zeros((1000, 256, 256)) #as previous paper, take the mean of the MMSE
#     for i, im in tqdm(enumerate(im_list)):
#         Im_list[i] = im.cpu().detach().numpy()

#     Im_norm2_list = np.sum(Im_list**2, axis = (1,2))
#     Im_mean = np.mean(Im_list, axis = 0)

#     var = np.mean(Im_norm2_list) - np.sum(Im_mean**2)
#     print('data variance: ', var)

#     sigma = np.sqrt(var/(256*256))
#     print('sigma :', sigma)

#     X = sigma*np.random.randn(1000, 256*256)
#     Y = sigma*np.random.randn(1000, 256*256)
#     M = ot.dist(X, Y)
#     dist = ot.emd2(a = [], b = [], M = M)
#     Dist_typ.append(dist)

# print(Name)
# print(Dist_typ)

# dist_typ = np.mean(Dist_typ)
# print(dist_typ)

# fig = plt.figure(figsize = (10, 5))
# plt.plot(list_index,Dist_wass, label = 'Wasserstein distance between 2 sampling by 2 NN')
# plt.plot([list_index[0],list_index[-1]],[dist_typ, dist_typ], label = 'Wasserstein distance between 2 sampling of the same distribution')
# plt.title('Wasserstein distance between samplings')
# plt.legend()
# fig.savefig(path_result+name+'Distance_wasserstein_ref.png')


# np.save(path_result+"Distance_wasserstein_ref", (Name,Dist_typ))