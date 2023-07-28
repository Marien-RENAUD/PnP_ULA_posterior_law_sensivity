import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import linregress


s = 5
s1 = s/255.
s2 = (s1)**2
cuda = True
device = "cuda:5"

name_list = ["woman01","woman02","woman03","woman04","woman05","woman06","woman07","woman08","woman09","woman10"]

def correlation(x, y):
    return np.mean((x-np.mean(x))*(y-np.mean(y)))/(np.std(x)*np.std(y))

path_result = 'results/result_rgb/10000/'
list_index = np.array([2,3,4,5,6,7,8])
dist_for = list_index - 1

# save the images
fig, ax = plt.subplots(2, 5, figsize = (15,6))
indice_list = [(0,0),(0,1),(0,2),(0,3),(0,4),(1,0),(1,1),(1,2),(1,3),(1,4)]
for i, name in enumerate(name_list):
    im = plt.imread("images/"+name+".jpg")
    ax[indice_list[i]].imshow(im)
    ax[indice_list[i]].axis(False)
fig.savefig(path_result+'All_images.png')

Wass_dist_list = np.zeros((len(name_list), len(list_index)))

for i, name in enumerate(name_list):
    Wass_dist = np.load(path_result+'Distance_wasserstein'+name+'.npy')
    Wass_dist_list[i,:] = Wass_dist
    #print
    print(name)
    corr = correlation(Wass_dist, dist_for)
    print(corr)


print('mean')
Wass = np.mean(Wass_dist_list, axis = 0)
corr_m = correlation(Wass, dist_for)
print(corr_m)

# Name, Dist_typ = np.load(path_result+'Distance_wasserstein_ref'+'.npy')
# Dist_typ_2 = []
# for r in Dist_typ:
#     Dist_typ_2.append(float(r))
# Dist_typ_2 = np.array(Dist_typ_2)
# Ref = np.mean(Dist_typ_2)*np.ones(list_index.shape[0])

Reg = []

for i in range(len(name_list)):
    reg = linregress(dist_for, Wass_dist_list[i,:])
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
ax1.plot(list_index,Wass, label='mean, correlation = {0:.5g}'.format(corr_m), color = 'k', linewidth = 2)
plt.legend()
ax1.set_ylabel('$W_1(\pi_{\epsilon, \delta}^1, \pi_{\epsilon, \delta}^2)$', color='k', fontsize=15) #Wasserstein distance

ax1.set_xlabel('Variance of the gaussian blur kernel', fontsize=15)
fig.tight_layout()

# ax2.set_title('Correlation of : {0:.4g}'.format(corr_t))
fig.savefig(path_result+'All.png')

fig, ax = plt.subplots(figsize = (10,6))
for i, name in enumerate(name_list):
    ax.scatter(Wass_dist_list[i,:],dist_for, c = list_index, alpha = alp) #, label=name +', correlation = {0:.4g}'.format(Reg[i].rvalue)
ax.scatter(Wass, dist_for, c = list_index, label='mean, correlation = {0:.5g}'.format(corr_m), linewidth = 2)
ax.set_xlabel('Wasserstein distance')
ax.set_ylabel('Distance between denoisers')
plt.legend()
fig.savefig(path_result+'All_scatter.png')


# fig, ax = plt.subplots(figsize = (10,6))
# for i, name in enumerate(name_list):    
#     resid = Wass_dist_list[i,:] - (Reg[i].slope * d1_dist_list[i,:] + Reg[i].intercept)
#     ax.plot(list_index, resid, color = colors[i], alpha = alp)
# reg_mean = linregress(D1, Wass)
# resid = Wass - (reg_mean.slope * D1 + reg_mean.intercept)
# ax.plot(list_index, resid, color = 'k', linewidth = 2)
# fig.savefig(path_result+'residu.png')