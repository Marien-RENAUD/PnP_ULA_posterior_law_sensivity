import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import linregress


s = 5
s1 = s/255.
s2 = (s1)**2
cuda = True
device = "cuda:5"

# path_result = 'results/result_ex_mixt_n_iter100000/'
# name = 'duck' # "goldhill", "simpson_nb512"
# list_index = [i for i in range(11)]
# ref_ind = 0
# path_model = 'Pretrained_models/mixt_models/DnCNN_sigma5_17layers_data_mixt'

# name = 'duck'
# Wass_dist_duck = np.load(path_result+'result_'+name+'/'+'Distance_wasserstein'+name+'.npy')
# Dist_d1_duck = np.load(path_result+'result_'+name+'/'+'Distance_d1'+name+'.npy')

# print(name)
# def correlation(x, y):
#     return np.mean((x-np.mean(x))*(y-np.mean(y)))/(np.std(x)*np.std(y))
# corr = correlation(Wass_dist_duck, Dist_d1_duck)
# print(corr)


# name = 'goldhill'
# Wass_dist_gold = np.load(path_result+'result_'+name+'/'+'Distance_wasserstein'+name+'.npy')
# Dist_d1_gold = np.load(path_result+'result_'+name+'/'+'Distance_d1'+name+'.npy')
# print(name)
# corr = correlation(Wass_dist_gold, Dist_d1_gold)
# print(corr)

# name = 'simpson_nb512'
# Wass_dist_simps = np.load(path_result+'result_simpson'+'/'+'Distance_wasserstein'+name+'.npy')
# Dist_d1_simps = np.load(path_result+'result_simpson'+'/'+'Distance_d1'+name+'.npy')

# print(name)
# corr = correlation(Wass_dist_simps, Dist_d1_simps)
# print(corr)

# name = 'cameraman'
# Wass_dist_cam = np.load(path_result+'result_'+name+'/'+'Distance_wasserstein'+name+'.npy')
# Dist_d1_cam = np.load(path_result+'result_'+name+'/'+'Distance_d1'+name+'.npy')
# print(name)
# corr = correlation(Wass_dist_cam, Dist_d1_cam)
# print(corr)

# name = 'castle'
# Wass_dist_cas = np.load(path_result+'result_'+name+'/'+'Distance_wasserstein'+name+'.npy')
# Dist_d1_cas = np.load(path_result+'result_'+name+'/'+'Distance_d1'+name+'.npy')
# print(name)
# corr = correlation(Wass_dist_cas, Dist_d1_cas)
# print(corr)

# print("total corr")
# Wass = np.concatenate([Wass_dist_duck,Wass_dist_gold,Wass_dist_simps,Wass_dist_cam, Wass_dist_cas])
# D1 = np.concatenate([Dist_d1_duck,Dist_d1_gold,Dist_d1_simps,Dist_d1_cam, Dist_d1_cas])
# corr_t = correlation(Wass, D1)
# print(corr_t)

# print('mean')
# Wass = (Wass_dist_duck+Wass_dist_gold+Wass_dist_simps+Wass_dist_cam+Wass_dist_cas)/5
# D1 = (Dist_d1_duck+Dist_d1_gold+Dist_d1_simps+Dist_d1_cam+Dist_d1_cas)/5
# corr_m = correlation(Wass, D1)
# print(corr_m)

# #Wasserstein distance between all four images see as one unique distribution
# Wass_dist = np.load(path_result+'Distance_wasserstein.npy')


# fig, ax1 = plt.subplots(figsize = (10,6))
# color = 'tab:orange'
# list_index = np.array(list_index)/10
# ax1.plot(list_index,Wass_dist_duck, color = 'b', marker = 'o', alpha = 0.8)
# ax1.plot(list_index,Wass_dist_gold, color = 'r', marker = 'o', alpha = 0.8)
# ax1.plot(list_index,Wass_dist_simps, color = 'g', marker = 'o', alpha = 0.8)
# ax1.plot(list_index,Wass_dist_cam, color = 'm', marker = 'o', alpha = 0.8)
# ax1.plot(list_index,Wass_dist_cas, color = 'c', marker = 'o', alpha = 0.8)
# ax1.set_ylabel('Wasserstein distance between samplings', color='k', fontsize=15)
# ax1.tick_params(axis='y')
# ax2 = ax1.twinx()
# color = 'tab:blue'
# ax2.plot(list_index,Dist_d1_duck, label='duck', color = 'b', marker = 'o', alpha = 0.8, linestyle = '--')
# ax2.plot(list_index,Dist_d1_gold, label='goldhill', color = 'r', marker = 'o', alpha = 0.8, linestyle = '--')
# ax2.plot(list_index,Dist_d1_simps, label='simpson', color = 'g', marker = 'o', alpha = 0.8, linestyle = '--')
# ax2.plot(list_index,Dist_d1_cam, label='cameraman', color = 'm', marker = 'o', alpha = 0.8, linestyle = '--')
# ax2.plot(list_index,Dist_d1_cas, label='castle', color = 'c', marker = 'o', alpha = 0.8, linestyle = '--')
# ax2.set_ylabel('Distance between denoisers', color='k', fontsize=15)
# ax2.tick_params(axis='y')
# ax1.set_xlabel('Proportion of MRI images in the training dataset', fontsize=15)
# fig.tight_layout()
# plt.legend()
# ax2.set_title('Correlation of : {0:.4g}'.format(corr_t))
# fig.savefig(path_result+'All.png')


# fig, ax1 = plt.subplots(figsize = (10,6))
# color = 'tab:orange'
# list_index = np.array(list_index)/10
# ax1.plot(list_index,Wass, color = color, marker = 'o', alpha = 0.8, label = 'Mean Wass dist')
# ax1.plot(list_index,Wass_dist, color = color, marker = 'o', alpha = 0.8, linestyle = '--', label = 'Wass dist for all sampling')
# ax1.set_ylabel('Wasserstein distance between samplings', color=color, fontsize=15)
# ax1.tick_params(axis='y', labelcolor=color)
# ax1.legend()
# ax2 = ax1.twinx()
# color = 'tab:blue'
# ax2.plot(list_index,D1, color = color, marker = '*', alpha = 0.8)
# ax2.set_ylabel('Distance between denoisers', color=color, fontsize=15)
# ax2.tick_params(axis='y', labelcolor=color)
# ax1.set_xlabel('Proportion of MRI images in the training dataset', fontsize=15)
# plt.title('Correlation of : {0:.4g}'.format(correlation(Wass, D1)))
# fig.tight_layout()
# fig.savefig(path_result+'Mean.png')

# print('global test')

# corr_t = correlation(Wass, D1)
# print(corr_t)

# corr_t = correlation(Wass_dist, D1)
# print(corr_t)



# print("FID")
# name = 'castle'
# print(name)
# Dist_FID_cas = np.load(path_result+'result_'+name+'/'+'Dist_FID.npy')
# corr = correlation(Dist_FID_cas, Dist_d1_cas)
# print(corr)

# name = 'goldhill'
# print(name)
# Dist_FID_gold = np.load(path_result+'result_'+name+'/'+'Dist_FID.npy')
# corr = correlation(Dist_FID_gold, Dist_d1_gold)
# print(corr)

# name = 'duck'
# print(name)
# Dist_FID_duck = np.load(path_result+'result_'+name+'/'+'Dist_FID.npy')
# corr = correlation(Dist_FID_duck, Dist_d1_duck)
# print(corr)

# print("total")
# Dist_FID = np.concatenate([Dist_FID_cas, Dist_FID_gold, Dist_FID_duck])
# Dist_d1 = np.concatenate([Dist_d1_cas, Dist_d1_gold, Dist_d1_duck])
# corr = correlation(Dist_FID, Dist_d1)
# print(corr)

# print("mean")
# Dist_FID = (Dist_FID_cas+Dist_FID_gold+Dist_FID_duck)/3
# Dist_d1 = (Dist_d1_cas+Dist_d1_gold+Dist_d1_duck)/3
# corr = correlation(Dist_FID, Dist_d1)
# print(corr)


# fig, ax1 = plt.subplots(figsize = (10,6))
# color = 'tab:orange'
# list_index = np.array(list_index)/10
# ax1.plot(list_index,Dist_FID_cas, color = 'c', marker = 'o', alpha = 0.8)
# ax1.plot(list_index,Dist_FID_gold, color = 'r', marker = 'o', alpha = 0.8)
# ax1.plot(list_index,Dist_FID_duck, color = 'b', marker = 'o', alpha = 0.8)
# ax1.set_ylabel('Wasserstein distance between samplings', color='k', fontsize=15)
# ax1.tick_params(axis='y')
# ax2 = ax1.twinx()
# color = 'tab:blue'
# ax2.plot(list_index,Dist_d1_cas, label='castle', color = 'c', marker = 'o', alpha = 0.8, linestyle = '--')
# ax2.plot(list_index,Dist_d1_gold, label='goldhill', color = 'r', marker = 'o', alpha = 0.8, linestyle = '--')
# ax2.plot(list_index,Dist_d1_duck, label='duck', color = 'b', marker = 'o', alpha = 0.8, linestyle = '--')
# ax2.set_ylabel('Distance between denoisers', color='k', fontsize=15)
# ax2.tick_params(axis='y')
# ax1.set_xlabel('Proportion of MRI images in the training dataset', fontsize=15)
# fig.tight_layout()
# plt.legend()
# ax2.set_title('Correlation of : {0:.4g}'.format(corr_t))
# fig.savefig(path_result+'FID.png')

name_list = ["duck","castle","simpson_nb512","goldhill","cameraman","02","03","04","05","06","07","09","10","11","12"]


def correlation(x, y):
    return np.mean((x-np.mean(x))*(y-np.mean(y)))/(np.std(x)*np.std(y))

path_result = 'results/result_layers50/'
list_index = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16])

fig, ax = plt.subplots(3, 5, figsize = (10,6))
indice_list = [(0,0),(0,1),(0,2),(0,3),(0,4),(1,0),(1,1),(1,2),(1,3),(1,4),(2,0),(2,1),(2,2),(2,3),(2,4)]
for i, name in enumerate(name_list):
    im_total = plt.imread("images/"+name+".png")
    if name == "duck":
        im = np.mean(im_total, axis = 2)
        im = im[400:656,550:806]
    if name == "castle":
        im = np.mean(im_total, axis = 2)
        im = im[100:356,0:256]
    if name == "simpson_nb512" or name == "goldhill":
        im = im_total[100:356,0:256]
    if name in set(["cameraman", '01', '02', '03', '04', '05', '06', '07']):
        im = im_total
    if name == '09':
        im = im_total[:256,256:]
    if name == '10':
        im = im_total[100:356,256:]
    if name == '11':
        im = im_total[:256,:256]
    if name == '12':
        im = im_total[100:356,100:356]
    ax[indice_list[i]].imshow(im, cmap = 'gray')
    ax[indice_list[i]].axis(False)
fig.savefig(path_result+'All_images.png')


Wass_dist_list = np.zeros((len(name_list), len(list_index)))
d1_dist_list = np.zeros((len(name_list), len(list_index)))
Wass_typ = np.load(path_result+"Distance_wasserstein_ref"+".npy")[1]
Wass_typ_ = np.zeros(len(Wass_typ))
for i in range(len(Wass_typ)):
    Wass_typ_[i] = float(Wass_typ[i])

for i, name in enumerate(name_list):
    Wass_dist = np.load(path_result+'Distance_wasserstein'+name+'.npy')
    Wass_dist_list[i,:] = Wass_dist - Wass_typ_[i]
    d1_dist = np.load(path_result+'Distance_d1'+name+'.npy')
    d1_dist_list[i,:] = d1_dist
    #print
    print(name)
    corr = correlation(Wass_dist, d1_dist)
    print(corr)


print('mean')
Wass = np.mean(Wass_dist_list, axis = 0)
D1 = np.mean(d1_dist_list, axis = 0)
corr_m = correlation(Wass, D1)
print(corr_m)

# Name, Dist_typ = np.load(path_result+'Distance_wasserstein_ref'+'.npy')
# Dist_typ_2 = []
# for r in Dist_typ:
#     Dist_typ_2.append(float(r))
# Dist_typ_2 = np.array(Dist_typ_2)
# Ref = np.mean(Dist_typ_2)*np.ones(list_index.shape[0])

Reg = []

for i in range(len(name_list)):
    reg = linregress(d1_dist_list[i,:], Wass_dist_list[i,:])
    Reg.append(reg)

# Generate a list of distinct colors using HSV color space
num_colors = len(name_list)
colors = plt.cm.hsv(np.linspace(0, 1, num_colors))

fig, ax1 = plt.subplots(figsize = (10,6))
color = 'tab:orange'
list_index = np.array(list_index)
alp = 0.15
for i, name in enumerate(name_list):
    ax1.plot(list_index,Wass_dist_list[i,:], color = colors[i], alpha = alp) #, label=name +', correlation = {0:.4g}'.format(Reg[i].rvalue)
ax1.plot(list_index,Wass, label='mean, correlation = {0:.4g}'.format(corr_m), color = 'k', linewidth = 2)
plt.legend()
ax1.set_ylabel('$W_1(\pi_{\epsilon, \delta}^1, \pi_{\epsilon, \delta}^2)$', color='k', fontsize=15) #Wasserstein distance

ax1.tick_params(axis='y')
ax2 = ax1.twinx()
color = 'tab:blue'
for i, name in enumerate(name_list):
    ax2.plot(list_index,d1_dist_list[i,:], color = colors[i], alpha = alp, linestyle = '--')

ax2.plot(list_index, D1, label='mean, correlation = {0:.4g}'.format(corr_m), color = 'k', linestyle = '--', linewidth = 2)

ax2.set_ylabel('-- $d_1(D_{\epsilon}^1, D_{\epsilon}^2$)', color='k', fontsize=15) #Distance between denoisers
ax2.tick_params(axis='y')
ax1.set_xlabel('Number of layers of the denoiser', fontsize=15)
fig.tight_layout()

# ax2.set_title('Correlation of : {0:.4g}'.format(corr_t))
fig.savefig(path_result+'All.png')

fig, ax = plt.subplots(figsize = (10,6))
ax.scatter(Wass, D1, c = list_index, label='mean, correlation = {0:.4g}'.format(corr_m), linestyle = '--', linewidth = 2)
ax.set_xlabel('Wasserstein distance')
ax.set_ylabel('Distance between denoisers')
plt.legend()
fig.savefig(path_result+'All_scatter.png')


fig, ax = plt.subplots(figsize = (10,6))
for i, name in enumerate(name_list):    
    resid = Wass_dist_list[i,:] - (Reg[i].slope * d1_dist_list[i,:] + Reg[i].intercept)
    ax.plot(list_index, resid, color = colors[i], alpha = alp)
reg_mean = linregress(D1, Wass)
resid = Wass - (reg_mean.slope * D1 + reg_mean.intercept)
ax.plot(list_index, resid, color = 'k', linewidth = 2)
fig.savefig(path_result+'residu.png')






