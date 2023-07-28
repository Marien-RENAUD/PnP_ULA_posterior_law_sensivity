from utils import *

parser = argparse.ArgumentParser()
parser.add_argument("--n_iter", type=int, default=10000, help='number of iteration in PnP-ULA')
parser.add_argument("--alpha", type=int, default=1, help='regularization parameter alpha in PnP-ULA')
parser.add_argument("--s", type=int, default=5, help='denoiser parameter')
parser.add_argument("--img", type=str, default='simpson_nb512.png', help='image to reconstruct')
parser.add_argument("--path_result", type=str, default='result', help='path to save the results : it will be save in results/path_result')
parser.add_argument("--model_name", type=str, default = None, help='name of the model for our models')
parser.add_argument("--gpu_number", type=int, default = 0, help='gpu number use')
parser.add_argument("--Lip", type=bool, default = False, help='True : the network is 1-Lip, False : no constraint')
parser.add_argument("--blur_type", type=str, default = 'uniform', help='uniform : uniform blur, gaussian : gaussian blur')
parser.add_argument("--l", type=int, default = 4, help='(2*l+1)*(2*l+1) is the size of the blur kernel. Need to verify 2l+1 < 128')
parser.add_argument("--si", type=float, default = 1., help='variance of the blur kernel in case of gaussian blur')
parser.add_argument("--num_of_layers", type=int, default = 17, help='numbers of layers in the deep neural network')
pars = parser.parse_args()

###
# PARAMETERS
###

Pb = ['deblurring'] # 'deblurring', 'inpainting
N = 256 #size of the image N*N

# Parameters for PnP-ULA
n_iter = pars.n_iter #1000
n_burn_in = int(n_iter/10)
n_inter = int(n_iter/1000)
n_inter_mmse = np.copy(n_inter)

# Denoiser parameters
s = pars.s

# Regularization parameters
alpha = pars.alpha # 1 or 0.3
c_min = 0 #-1
c_max = 1 #2

# Inverse problem prameters
sigma = 1
l = pars.l # size of the blurring kernel

# Path to save the results
path_result = 'results/' + pars.path_result
os.makedirs(path_result, exist_ok = True)

###
# IMAGE
###
path_img = 'images/'
img = pars.img
name_im = img.split('.')[0]
im = load_image_gray(path_img, img)
n_Cols, n_Rows = im.shape
#image normalization
im = im/np.max(im)

###
# Harware Parameters
###

# GPU device selection
cuda = True
device = "cuda:"+str(pars.gpu_number)
# Type
dtype = torch.float32
tensor = torch.FloatTensor
# Seed
seed = 40

# Prior regularization parameter
alphat = torch.tensor(alpha, dtype = dtype, device = device)
# Normalization of the standard deviation noise distribution
sigma1 = sigma/255.0
sigma2 = sigma1**2
sigma2t = torch.tensor(sigma2, dtype = dtype, device = device)
# Normalization of the denoiser noise level
s1 = s/255.
s2 = (s1)**2
s2t = torch.tensor(s2, dtype = dtype, device = device)
# Parameter strong convexity in the tails
lambd = 0.5/(2/sigma2 + alpha/s2)
lambdt = torch.tensor(lambd, dtype = dtype, device = device)

# Discretization step-size
delta = 1/3/(1/sigma2 + 1/lambd + alpha/s2)
deltat = torch.tensor(delta, dtype = dtype, device = device)

###
# Prior Fidelity
###

if pars.Lip:
    from model.realSN_models import DnCNN
else:
    from model.models import DnCNN
path_model = "Pretrained_models/" + pars.model_name + ".pth"
net = DnCNN(channels=1, num_of_layers=pars.num_of_layers)
model = nn.DataParallel(net,device_ids=[int(str(device)[-1])],output_device=device)#.cuda()
dicti = torch.load(path_model, map_location=torch.device(device if torch.cuda.is_available() else "cpu"))
dicti_ = {}
for keys, values in dicti.items():
    dicti_["module."+keys] = values.to(device)
model.load_state_dict(dicti_)
model.eval()

# model = load_model(model_type, s, device = device, cuda = cuda)
Ds = lambda x : torch_denoiser(x, model)
prior_grad = lambda x : alphat*(Ds(x) - x)/s2t

###
# Data Fidelity
###

# Definition of the convolution operator
if pars.blur_type == 'uniform':
    l_h = 2*l+1
    h = np.ones((1, l_h))
if pars.blur_type == 'gaussian':
    si = pars.si
    h = np.array([[np.exp(-i**2/(2*si**2)) for i in range(-l,l+1)]])
h = h/np.sum(h)
h_= np.dot(h.T,h)
h_conv = np.flip(h_) # Definition of Data-grad
h_conv = np.copy(h_conv) #Useful because pytorch cannot handle negatvie strides
hconv_torch = torch.from_numpy(h_conv).type(tensor).to(device)
hcorr_torch = torch.from_numpy(h_).type(tensor).to(device)
hconv_torch = hconv_torch.unsqueeze(0).unsqueeze(0)
hcorr_torch = hcorr_torch.unsqueeze(0).unsqueeze(0)

#forward model definition
A = lambda x: torch.nn.functional.conv2d(torch.nn.functional.pad(x, [l,l,l,l], mode = 'circular'), hconv_torch, groups=x.size(1), padding = 0)
AT = lambda x: torch.nn.functional.conv2d(torch.nn.functional.pad(x, [l,l,l,l], mode = 'circular'), hcorr_torch, groups=x.size(1), padding = 0)

#blur the blur image in torch
im_t = torch.from_numpy(np.ascontiguousarray(im)).float().unsqueeze(0).unsqueeze(0).to(device)
gen = torch.Generator(device=device)
gen.manual_seed(0) #for reproductivity
y_t = A(im_t) + torch.normal(torch.zeros(*im_t.size()).to(device), std = sigma1*torch.ones(*im_t.size()).to(device),generator=gen)

# DATA-GRAD FOR THE DEBLURRING
data_grad = lambda x: -AT(A(x) - y_t)/(sigma2t)

#initialization at the Markov Chain
init_torch = y_t

###
# Sampling
###

# Name for data storage
name = '{}_sigma{}_s{}'.format(name_im, sigma, s)

Samples_t, Mmse_t, Mmse2_t = pnpula(init = init_torch, data_grad = data_grad, prior_grad = prior_grad, delta = deltat, lambd = lambdt, seed = seed, device = device, n_iter = n_iter, n_inter = n_inter, n_inter_mmse = n_inter_mmse, path = path_result, name = name)

###
# Compututation on the results
###

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

#observation
y = y_t.cpu().detach().numpy()
y = y[0,0,:,:]
psb = PSNR(im, y, data_range = 1)
ssb = ssim(im, y, data_range = 1)

# Compute PSNR and SIM for the online MMSE
n = len(Mmse)
print(n)
PSNR_list = []
SIM_list = []
for i in range(1,n):
    PSNR_list.append(PSNR(im, np.mean(Mmse[:i], axis = 0), data_range = 1))
    SIM_list.append(ssim(im, np.mean(Mmse[:i], axis = 0), data_range = 1))

# Computation of the mean and std of the whole Markov chain
xmmse = np.mean(Mmse, axis = 0)
pmmse = PSNR(im, xmmse, data_range = 1)
smmse = ssim(im, xmmse, data_range = 1)

# Computation of the std of the Markov chain
xmmse2 = np.mean(Mmse2, axis = 0)
var = xmmse2 - xmmse**2
var = var*(var>=0) + 0*(var<0)
std = np.sqrt(var)
diff = np.abs(im-xmmse)

#save the result of the experiment
dict = {
        'Samples' : Samples,
        'Mmse' : Mmse,
        'Mmse2' : Mmse2,
        'PSNR_sample' : Psnr_sample,
        'SIM_sample' : SIM_sample,
        'PSNR_mmse' : PSNR_list,
        'SIM_list' : SIM_list,
        'image_name' : name_im,
        'observation' : y,
        'PSNR_y' : psb,
        'SIM_y' : ssb,
        'ground_truth' : im,
        'MMSE' : xmmse,
        'PSNR_MMSE' : pmmse,
        'SIM_MMSE' : smmse,
        'std' : std,
        'diff' : diff,
        'n_iter' : n_iter,
        's' : s,
        'alpha' : alpha,
        'c_min' : c_min,
        'c_max' : c_max,
        'sigma' : sigma,
        'l' : l,
        'lambda' : lambd,
        'delta' : delta,
    }

np.save(path_result+'/'+ name +'_result.npy', dict)

###
# PLOTS
###

#save the observation

plt.imsave(path_result + '/observation.png', y, cmap = 'gray')

#creation of a video of the samples
writer = imageio.get_writer(os.path.join(path_result,"samples_video"+name+".mp4"), fps=100)
for im_ in Samples:
    im_uint8 = np.clip(im_ * 255, 0, 255).astype(np.uint8)
    writer.append_data(im_uint8)
writer.close()

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

# Saving of the MMSE of the sample
plt.imsave(path_result + '/mmse_' + name + '_psnr{:.2f}_ssim{:.2f}.png'.format(pmmse, smmse), xmmse, cmap = 'gray')

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

# Saving of the standard deviation and the difference between MMSE and Ground-Truth (GT)
fig = plt.figure(figsize = (10, 5))
ax1 = fig.add_subplot(1,2,1)
ax1.imshow(std, cmap = 'gray')
ax1.axis('off')
ax1.set_title("Std of the Markov Chain")
ax2 = fig.add_subplot(1,2,2)
ax2.imshow(diff, cmap = 'gray')
ax2.axis('off')
ax2.set_title("Diff MMSE-GT")
fig.savefig(path_result+'/Std_of_the_Markov_Chain_'+name_im+'n_iter{}'.format(n_iter)+'.png')
plt.show()

# Saving of the Fourier transforme of the standard deviation, to detect possible artecfact of sampling
plt.imsave(path_result+"/Fourier_transform_std_MC_"+name_im+'n_iter{}'.format(n_iter)+".png",np.fft.fftshift(np.log(np.abs(np.fft.fft2(std))+1e-10)))