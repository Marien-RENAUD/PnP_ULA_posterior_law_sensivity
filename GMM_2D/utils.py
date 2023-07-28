from PIL import Image
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from tqdm import tqdm
from bm3d import bm3d, BM3DProfile
import scipy.linalg as linalg
from scipy.linalg import sqrtm

from scipy.interpolate import griddata
import numpy.ma as ma
from numpy.random import uniform, seed
from matplotlib import cm
import ot as ot #for optimal transport computation

from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split

from matplotlib.patches import Ellipse #to make good draw

from celluloid import Camera #to make gif
from IPython.display import HTML

def gaussian_mixt_example(name):
    """
    Return one of the three gaussian mixture constant used in the experiment
    The output is mu_list, sigma_list, pi_list
    """
    if name == 'symetric_gaussians':
        return [np.array([5,5]), np.array([-5,-5])], [np.eye(2), np.eye(2)], [0.5, 0.5]
    if name == 'cross':
        return [np.array([0,0]), np.array([0,0])], [[[2,0.5],[0.5,0.15]], [[0.15,0.5],[0.5,2.]]], [0.5, 0.5]
    if name == 'disymmetric_gaussians':
        return [np.array([3,3]), np.array([3,-3])], [np.eye(2), np.eye(2)/5], [0.5, 0.5]


def draw_gaussian(plt2 = plt, Sigma=np.asarray([[.9,0.9],[0.9,1.]]), mu=np.asarray([1.,0.]), npts = 5000, rbox = 2, color = 'g', alpha = 0.5):
    """
    Draw a 2D Gaussian distribution of covariance Sigma and mean mu, it use a random sample of size npts. The level of the quantile
    draw are [0.01, 0.1, 0.4, 0.8, 1.0]. This method is very sensible to the number npts and pretty slow.
    """
    
    def gauss(x,y,Sigma,mu):
        X=np.vstack((x,y)).T
        mat_multi=np.dot((X-mu[None,...]).dot(np.linalg.inv(Sigma)),(X-mu[None,...]).T)
        return  np.diag(np.exp(-1*(mat_multi)))

    # make up some randomly distributed data
    npts = npts
    x = uniform(-rbox, rbox, npts)
    y = uniform(-rbox, rbox, npts)
    z = gauss(x, y, Sigma=Sigma, mu=mu)
    xi = np.linspace(-rbox+.1, rbox+.1, 100)
    yi = np.linspace(-rbox+.1, rbox+.1, 100)
    ## grid the data.
    zi = griddata((x, y), z, (xi[None,:], yi[:,None]), method='cubic')
    levels = [0.01, 0.1, 0.4, 0.8, 1.0]
    # contour the gridded data, plotting dots at the randomly spaced data points.
    CS = plt2.contour(xi,yi,zi,len(levels),linewidths=1,colors=color, levels=levels, alpha= alpha)
    #plot of the mean
    plt2.scatter(mu[0], mu[1], c=color, marker= '+', alpha= alpha)

def draw_gaussian_2(ax2, Sigma=np.asarray([[.9,0.9],[0.9,1.]]), mu=np.asarray([1.,0.]), rbox = 5, color = 'g', alpha = 1, list_a = [0.01, 0.1, 0.5], label = '', linewidth = 1):
    """
    A second method to draw 2D Gaussian distribution which use directly the close form of the quantile (there are ellipse).
    Faster than the previous method and more reliable.
    """
    l, v = np.linalg.eigh(Sigma)
    l1, l2 = l
    v1, v2 = v

    for i, a in enumerate(list_a):
        if i == 0 and label != '':
            if np.sum(v1*np.array([1,0])) >= 0:
                e = Ellipse(xy = mu, width = 2*np.sqrt(-2*l1*np.log(a)), height = 2*np.sqrt(-2*l2*np.log(a)), angle = -360*np.arccos(np.sum(v1*np.array([1,0])))/(2*np.pi), edgecolor=color, fc='None', alpha = alpha, label = label, linewidth = linewidth)
            else :
                e = Ellipse(xy = mu, width = 2*np.sqrt(-2*l1*np.log(a)), height = 2*np.sqrt(-2*l2*np.log(a)), angle = 360*np.arccos(np.sum(v1*np.array([1,0])))/(2*np.pi), edgecolor=color, fc='None', alpha = alpha, label = label, linewidth = linewidth)
            ax2.add_artist(e)
        else:
            if np.sum(v1*np.array([1,0])) >= 0:
                e = Ellipse(xy = mu, width = 2*np.sqrt(-2*l1*np.log(a)), height = 2*np.sqrt(-2*l2*np.log(a)), angle = -360*np.arccos(np.sum(v1*np.array([1,0])))/(2*np.pi), edgecolor=color, fc='None', alpha = alpha, linewidth = linewidth)
            else :
                e = Ellipse(xy = mu, width = 2*np.sqrt(-2*l1*np.log(a)), height = 2*np.sqrt(-2*l2*np.log(a)), angle = 360*np.arccos(np.sum(v1*np.array([1,0])))/(2*np.pi), edgecolor=color, fc='None', alpha = alpha, linewidth = linewidth)
            ax2.add_artist(e)

def sample_gaussian(mu_list, sigma_list, pi_list, N):
    """
    Function to generate N sample a gaussian mixture of parameters mu_list (means), sigma_list (covariance matrix) and pi_list (weights)

    """
    r = len(mu_list)
    sigma_sqrt_list = [] #compute the sqrt of each covariance matrix sigma
    for i in range(r):
        sigma_sqrt_list.append(linalg.sqrtm(sigma_list[i]))
    U_sample = np.random.randn(2,int(pi_list[0]*N))
    X = mu_list[0][:,None] + np.dot(sigma_sqrt_list[0], U_sample)
    for i in range(1,r):
        U_sample = np.random.randn(2,int(pi_list[i]*N))
        X_sample = mu_list[i][:,None] + np.dot(sigma_sqrt_list[i], U_sample)
        X = np.concatenate([X,X_sample], axis = 1)
    X = np.random.permutation(X.T) #break the artificial order
    return X

def Alpha(p_list):
    p_list = np.array(p_list)
    alpha_list = p_list/np.max(p_list)
    return alpha_list
    
rbox = 7

def draw_gaussian_mixture(ax2, mu_list, sigma_list, alpha_list, rbox = rbox, color = 'k',  label = 'x|y', linewidth = 1):
    r = len(mu_list)
    i_max = np.argmax(alpha_list)
    for i in range(r):
        if i == i_max:
            draw_gaussian_2(ax2 = ax2, mu = mu_list[i], Sigma = sigma_list[i], rbox = rbox, color = color, alpha = alpha_list[i], label = label, linewidth = linewidth)
        else:
            draw_gaussian_2(ax2 = ax2, mu = mu_list[i], Sigma = sigma_list[i], rbox = rbox, color = color, alpha = alpha_list[i], linewidth = linewidth)
    ax2.legend()
    ax2.set_xlim(-rbox, rbox)
    ax2.set_ylim(-rbox, rbox)

def constantes_conditionnal_prob(A, y, sigma, mu_list, sigma_list, pi_list):
    """
    Function to compute the constante of the conditional law x|y for a forward model, y = Ax + n, where $n \sim \mathcal{N}(0,\sigma)$.
    """
    p = len(mu_list)
    sigma_inv_list = []
    for i in range(p):
        sigma_inv_list.append(np.linalg.inv(sigma_list[i]))
    sigma_cond_inv_list = []
    sigma_cond_list = []
    for i in range(p):
        sigma_cond_inv_list.append(sigma_inv_list[i] + A.T@A/sigma)
        sigma_cond_list.append(np.linalg.inv(sigma_cond_inv_list[i]))
    In = np.eye(2)
    #the parameter of the conditionnal law
    mu_cond_list = []
    p_list = np.zeros(p)
    for i in range(p):
        mu_cond_list.append(sigma_cond_list[i] @ (sigma_inv_list[i]@mu_list[i] + A@y /sigma))
        p_list[i] = pi_list[i] * np.exp(0.5*(mu_cond_list[i].T@sigma_cond_inv_list[i]@mu_cond_list[i]-mu_list[i].T@sigma_inv_list[i]@mu_list[i]-y.T@y/sigma)) / np.sqrt(np.linalg.det(sqrtm(sigma_list[i])@A.T@A@sqrtm(sigma_list[i]) + sigma * In))

    p_list = p_list / np.sum(p_list)

    return mu_cond_list, sigma_cond_list, p_list

def sample_posterior(A, y, sigma, N, mu_list, sigma_list, pi_list):
     """
     Function to sample the posterior distribution
     """
     mu_cond_list, sigma_cond_list, p_list = constantes_conditionnal_prob(A, y, sigma, mu_list, sigma_list, pi_list)
     return sample_gaussian(mu_cond_list, sigma_cond_list, p_list, N) #sampling

def gaussian_function(x, mu, sigma):
    """
    Define the Gaussian function of mean mu and covariance sigma
    """
    fac = (x-mu).T@np.linalg.inv(sigma)@(x-mu)
    return np.exp(-fac/2)

def exact_score_cond(x,y,A,sigma, mu_list, sigma_list, pi_list):
    """
    Compute the exact conditional score
    x,y: np.array (2,)
    A : np.array (2,2)
    sigma : float
    """
    [mu_1_cond, mu_2_cond], [sigma_1_cond, sigma_2_cond], [p1, p2] = constantes_conditionnal_prob(A, y, sigma, mu_list, sigma_list, pi_list)
    sigma_1_cond_inv = np.linalg.inv(sigma_1_cond)
    sigma_2_cond_inv = np.linalg.inv(sigma_2_cond)
    
    p_cond_1 = p1 * np.exp(-0.5*(x - mu_1_cond).T@sigma_1_cond_inv@(x - mu_1_cond)) / np.sqrt((2*np.pi)**2*np.linalg.det(sigma_1_cond)) 
    p_cond_2 = p2 * np.exp(-0.5*(x - mu_2_cond).T@sigma_2_cond_inv@(x - mu_2_cond)) / np.sqrt((2*np.pi)**2*np.linalg.det(sigma_2_cond))
    
    score = - (p_cond_1 * sigma_1_cond_inv @ (x - mu_1_cond) + p_cond_2 * sigma_2_cond_inv @ (x - mu_2_cond)) / (p_cond_1 + p_cond_2)
    return score

def sample_with_exact_score(N, x_0, y, delta, A, sigma, mu_list, sigma_list, pi_list):
    """
    Sample a gaussian mixture distribution with its exact score
    """
    n = x_0.shape[0]
    X = [x_0]
    for i in tqdm(range(N)):
        x_k = X[-1]
        z_k1 = np.random.randn(n)
        x_k1 = X[-1] + delta * exact_score_cond(x_k,y,A,sigma, mu_list, sigma_list, pi_list) + np.sqrt(2*delta)*z_k1
        X.append(x_k1)
    X = np.array(X)
    return X

def Theorical_MMSE(epsilon, mu_list, sigma_list, pi_list):
    """
    Return the exact denoiser for a prior distribution defined by its mean, mu_list, its covariance matrix sigma_list, and the weight pi_list
    """
    r = len(mu_list)
    Id = np.eye(2)
    sigma_inv_list = []
    for i in range(r):
        sigma_inv_list.append(np.linalg.inv(sigma_list[i]))

    def denoiser(x):
        c_list = []
        mu_mmse_list = []
        for i in range(r):
            c = np.exp(-0.5*(x-mu_list[i]).T@np.linalg.inv(np.sqrt(epsilon)*Id+sigma_list[i])@(x-mu_list[i]))
            c = c / np.sqrt(np.linalg.det(np.sqrt(epsilon)*Id+sigma_list[i]))
            c_list.append(c) 
            mu_mmse_list.append(np.linalg.inv(Id/np.sqrt(epsilon) + sigma_inv_list[i])@(x/np.sqrt(epsilon) + sigma_inv_list[i]@mu_list[i]))
        A = 0
        B = 0
        for i in range(r):
            A += c_list[i]*pi_list[i]*mu_mmse_list[i]
            B += c_list[i]*pi_list[i]
        return A/B
    return denoiser

def Wasserstein_distance(sample1, sample2):
    """
    Computed the Wasserstein distance between two samples of size (n,2). It extract randomly a sample of size (1000,2) of it.
    Solve the Earth Movers distance, so no regularization.
    """
    s1 = np.random.permutation(sample1)[:1000]
    s2 = np.random.permutation(sample2)[:1000]
    M = ot.dist(s1, s2)
    dist = ot.emd2(a = [], b = [], M = M)
    return dist
    
def denoiser_map(plt2, denoiser, prior_data, xmin = -7, xmax = 7, ymin = -7, ymax = 7, npts = 100, seed = 1234):
    """ 
    function to draw the map of the denoiser denoiser in the domain [xmin,xmax]*[ymin,ymax]. The data of the prior are also draw for lisibility.
    """
    xlen = xmax - xmin
    xmean = (xmax + xmin) / 2
    ylen = ymax - ymin
    ymean = (ymax + ymin) / 2
    np.random.seed(seed)
    X_data = (np.random.random((npts,2))-0.5)*np.array([xlen,ylen])[None,:] - np.array([xmean,ymean])

    X_hat_data = []
    for x in X_data:
        X_hat_data.append(denoiser(x))
    X_hat_data = np.array(X_hat_data)

    plt2.scatter(prior_data[:, 0], prior_data[:, 1], alpha=0.6, c='g')

    for i in range(len(X_data)):
        x = X_data[i]
        plt2.plot(x[0],x[1],'ro')

    for i in range(len(X_hat_data)):
        y = X_hat_data[i]
        plt2.plot(y[0],y[1],'bo')

    for i in range(len(X_data)):
        plt2.plot([X_data[i,0],X_hat_data[i,0]],[X_data[i,1],X_hat_data[i,1]], color = 'r', alpha = 0.5)