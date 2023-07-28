import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as PSNR
import os
import utils
import argparse
import torch.nn as nn
import ot as ot #for optimal transport computation
from sklearn.decomposition import PCA
from pytorch_fid import fid_score
import argparse
from models.network_unet import UNetRes as net

parser = argparse.ArgumentParser()
parser.add_argument("--img", type=str, default='simpson_nb512', help='image to reconstruct')
parser.add_argument("--gpu_number", type=int, default = 0, help='gpu number use')
pars = parser.parse_args()

s = 5
s1 = s/255.
s2 = (s1)**2
cuda = True
device = "cuda:"+str(pars.gpu_number)

path_result = 'results/result_rgb/forward_mismatch/'
name = pars.img # "goldhill", "simpson_nb512", 'cameraman', 'duck', 'castle'
list_index = [2.8,2.84,2.88,2.92,3.04,3.08,3.12,3.16]
ref_ind = 3
path_model = 'Pretrained_models/celebA(woman faces)'

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
#     n_channels = 3 #RGB images
#     model = net(in_nc=n_channels+1, out_nc=n_channels, nc=[64, 128, 256, 512], nb=4, act_mode='R', downsample_mode="strideconv", upsample_mode="convtranspose")
#     path_model = 'Pretrained_models/'+pars.model_name+'.pth'
#     model.load_state_dict(torch.load(path_model), strict=True)
#     model = model.to(device)

#     def Ds(x):
#         """
#         Denoiser Drunet of level of noise s1 (std of the noise)
#         """
#         noise_map = torch.FloatTensor([s1]).repeat(1, 1, x.shape[2], x.shape[3]).to(device)
#         x = torch.cat((x, noise_map), dim=1)
#         x = x.to(device)
#         return model(x)
#     return Ds

# Metric_list = []
# Norm_1_list = []
# Norm_2_list = []

# r = ref_ind #the sampling which is taken as a reference
# im_list = torch.load(path_result+"woman_gaussian_si"+str(r)+"/"+"Samples_"+name+"_sigma1_s5.pt", map_location=device)
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

# ###
# # Computation of the Wasserstein distance between sampling
# ###

Dist_wass = []

data = np.load(path_result+"womansi"+str(ref_ind).replace('.', '_')+"/"+name+"_sigma1_s5_result.npy", allow_pickle=True)
my_dict = data.item()
im_list = my_dict['Samples']
im_list = np.array(im_list)
im_list_1 = im_list.reshape(1000, 3*256*256)

for r0 in tqdm(list_index):
    data = np.load(path_result+"womansi"+str(r0).replace('.', '_')+"/"+name+"_sigma1_s5_result.npy", allow_pickle=True)
    my_dict = data.item()
    im_list = my_dict['Samples']
    im_list = np.array(im_list)
    im_list_2 = im_list.reshape(1000, 3*256*256)
    M = ot.dist(im_list_1, im_list_2, p = 1)
    dist = ot.emd2(a = [], b = [], M = M)
    Dist_wass.append(dist)

print(Dist_wass)

fig = plt.figure(figsize = (10, 5))
plt.plot(list_index,Dist_wass)
fig.savefig(path_result+name+'Distance_wasserstein.png')
np.save(path_result+"Distance_wasserstein"+name, np.array(Dist_wass))

def correlation(x, y):
    return np.mean((x-np.mean(x))*(y-np.mean(y)))/(np.std(x)*np.std(y))
list_index = np.array(list_index)
Metric_list = list_index - 1
corr = correlation(Dist_wass, Metric_list)
print(corr)

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

# Name = ['goldhill', 'simpson_nb512', 'cameraman', 'duck', 'castle']

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