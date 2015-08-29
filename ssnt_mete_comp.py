from __future__ import division
import matplotlib
matplotlib.use('Agg')
import numpy as np
import csv
from scipy import stats
from scipy.stats import expon, rv_discrete, rv_continuous
import matplotlib.pyplot as plt
import mete
import mete_distributions
import macroecotools
import macroeco_distributions as md
import working_functions as wk

class ssnt_isd:
    """The ISD predicted by SSNT in the simplest form is an exponential 
    
    distribution with parameter d/g = N / E
    
    """ 
    def __init__(self, d_over_g):
        self.scale = 1 / d_over_g
        
    def pdf(self, x):
        return stats.expon.pdf(x, scale = self.scale)
    
    def cdf(self, x):
        return stats.expon.cdf(x, scale = self.scale)
    
    def ppf(self, q):
        return -self.scale * np.log(1 - np.array(q))

class ssnt_isd_bounded:
    """This is the ISD of SSNT with an lower bound at 1, 
    
    consistent with METE.
    Here the parameter of the lower-truncated exponential is d/g = N / (E - N).
    
    """ 
    def __init__(self, d_over_g):
        self.scale = 1 / d_over_g
        
    def pdf(self, x):
        return stats.expon.pdf(x, scale = self.scale, loc = 1)
    
    def cdf(self, x):
        return stats.expon.cdf(x, scale = self.scale, loc = 1)
    
    def ppf(self, q):
        return -self.scale * np.log(1 - np.array(q)) + 1

class ssnt_isd_transform():
    """This is the ISD predicted by SSNT when b & d are constant
    
    for individuals regardless of size, while g follows scaling predicted
    by metabolic theory, i.e., g(D^2/3) is constant.
    For consistency, the predicted distribution is still in the unit
    of metabolic rate (D^2), which is a Weibull distribution.
    
    """
    def __init__(self, d_over_g):
        self.k = 1/3
        self.lam = d_over_g ** 3
        
    def pdf(self, x):
        x = np.array(x)
        return self.k / self.lam * (x / self.lam) ** (self.k - 1) * np.exp(-(x / self.lam) ** self.k)
    
    def cdf(self, x,):
        x = np.array(x)
        return 1 - np.exp(-(x / self.lam) ** self.k)
    
    def ppf(self, q):
        q = np.array(q)
        return self.lam * (np.log(1 / (1 - q)) ** (1 / self.k))
    
def lik_mete_sp(n, epsilon, S, N, E, loglik = True):
    """Likelihood of a species jointly has abundance n
    
    and total energy consumption epsilon in METE.
    Inputs:
    n - abundance of given species
    epsilon - total energy consumption of the given species
    S, N, E - community-level state variables
    loglik - if the log-likelihood is to be returned

    """
    lambda2 = mete.get_lambda2(S, N, E)
    C = n * lambda2 / (np.exp(-lambda2 * n) - np.exp(-lambda2 * n * E))
    beta = mete.get_beta(S, N)
    phi_n = md.trunc_logser.pmf(n, np.exp(-beta), N)
    n_list = range(1, n)
    if loglik: 
        if n == epsilon:
            ll = n * np.log(C) - lambda2 * n * epsilon * n +np.log(phi_n)
        else:
            ll = n * np.log(C) - lambda2 * n * epsilon + (n - 1) * np.log(epsilon - n) \
               - sum(np.log(n_list)) + np.log(phi_n)
        return ll
    else: 
        if n == epsilon: 
            L = C ** n * np.exp(-lambda2 * n * n * epsilon) * phi_n
        else:
            L = C ** n * np.exp(-lambda2 * n * epsilon) * (epsilon - n) ** (n - 1) \
              / np.factorial(n-1) * phi_n
        return L

def lik_ssnt_sp(n, epsilon, S, N, E, loglik = True):
    """Likelihood of a species jointly has abundance n
    
    and total energy consumption epsilon in O'Dwyer et al.'s (2009) 
    neutral model.
    This function takes the same inputs as lik_mete_sp.
    Note that epsilon and E are some measure of metabolic rate or biomass, 
    which does not have to be 
    
    """
    b_over_d = 1 / np.exp(mete.get_beta(S, N, version = 'untruncated'))
    d_over_g = N / E
    b_over_g = b_over_d * d_over_g
    n_list = range(1, n + 1)
    if loglik:
        ll = -np.log(epsilon) - d_over_g * epsilon + n * np.log(b_over_g * epsilon) \
           - sum(np.log(n_list)) + np.log(-1 / np.log(1 - b_over_d))
        return ll
    else: 
        L = 1 / epsilon * np.exp(-d_over_g * epsilon) * (b_over_g * epsilon) ** n \
          / np.factorial(n) * (-1 / log(1 - b_over_g / d_over_g))
        return L

def lik_ssnt_ind(n, epsilon, S, N, d_over_g, loglik = True):
    """Likelihood of an individual having size epsilon and belongs to a species with abundance n"""
    b_over_d = 1 / np.exp(mete.get_beta(S, N, version = 'untruncated'))
    sad = stats.logser.pmf(n, b_over_d)
    isd = ssnt_isd_bounded(d_over_g)
    if loglik: return np.log(sad) + np.log(d_over_g) - d_over_g * (epsilon - 1)
    else: return sad * isd.pdf(epsilon)
    
def lik_mete_ind(n, epsilon, S, N, E, unit = 'mr', loglik = True):
    """Likelihood of an individual having size epsilon and belongs to a species with 
    
    abundance N in METE. Equivalent to R(n, epsilon). Input "unit" can take value 
    "mr" or "diameter", which controls the unit of size in the output.
    
    """
    psi = mete_distributions.psi_epsilon(S, N, E)
    Z_inv = psi.norm_factor * N / S
    lambda2 = psi.lambda2
    lambda1 = psi.beta - lambda2
    if loglik: 
        if unit == 'mr': return np.log(Z_inv) - lambda1 * n - lambda2 * n * epsilon
        else: return np.log(Z_inv) - lambda1 * n - lambda2 * n * (epsilon ** 2) + np.log(2) + np.log(epsilon)
    else: 
        if unit == 'mr': return Z_inv * np.exp(-lambda1 * n) * np.exp(-lambda2 * n * epsilon)
        else: return Z_inv * np.exp(-lambda1 * n) * np.exp(-lambda2 * n * (epsilon ** 2)) * 2 * epsilon

def get_lik_ind_ssnt_mete(dat_list, in_dir = './data/', out_dir = './out_files/', cutoff = 9):
    """Calculate the log-likelihood of the joint P(N, m) for SSNT and METE"""
    for dat in dat_list:
        dat_study = wk.import_raw_data(in_dir + dat + '.csv')
        for site in np.unique(dat_study['site']):
            dat_site = dat_study[dat_study['site'] == site]
            S = len(np.unique(dat_site['sp']))
            if S > cutoff:
                N = len(dat_site)
                dbh = dat_site['dbh']
                E = sum((dbh / min(dbh)) ** 2)
                sum_D = sum(dbh / min(dbh))
                sum_l_ssnt = 0
                sum_l_mete = 0
                d_over_g = N / (sum_D - N)
                for i, dbh in enumerate(dbh):
                    sp_ind = dat_site['sp'][i]
                    abd_ind = len(dat_site[dat_site['sp'] == sp_ind])
                    sum_l_ssnt += lik_ssnt_ind(abd_ind, dbh, S, N, d_over_g)
                    sum_l_mete += lik_mete_ind(abd_ind, dbh, S, N, E, unit = 'diameter')
                out = open(out_dir + 'ind_lik_comp.txt', 'a')
                print>>out, dat, site, str(sum_l_mete / S), str(sum_l_ssnt / S)
                out.close()
                
def get_ssnt_obs_pred_isd(raw_data, dataset_name, model = 'original', data_dir = './out_files/', cutoff = 9):
    """Obtain the observed dbh**2 and the values predicted by SSNT and write to file.
    
    Input:
    raw_data - data in the same format as obtained by wk.import_raw_data(), with 
        three columns site, sp, and dbh.
    dataset_name - name of the dataset for raw_data.
    model - whether the original model (ssnt_isd), the scaled model (ssnt_isd_transform), or 
        the lower truncated model (ssnt_isd_bounded) is adopted
    data_dir - directory for output file.
    cutoff - minimal number of species for a site to be included.
    
    """
    usites = np.sort(list(set(raw_data["site"])))
    if model == 'original':
        f1_write = open(data_dir + dataset_name + '_obs_pred_isd_ssnt.csv', 'wb')
    elif model == 'transform':
        f1_write = open(data_dir + dataset_name + '_obs_pred_isd_ssnt_transform.csv', 'wb')
    elif model == 'bounded':
        f1_write = open(data_dir + dataset_name + '_obs_pred_isd_ssnt_bounded.csv', 'wb')
    f1 = csv.writer(f1_write)
    
    for site in usites:
        subdat = raw_data[raw_data["site"] == site]
        dbh_raw = subdat[subdat.dtype.names[2]]
        dbh_scale = np.array(sorted(dbh_raw / min(dbh_raw)))
        S0 = len(set(subdat[subdat.dtype.names[1]]))
        N0 = len(dbh_scale)
        if S0 > cutoff:
            scaled_rank = [(x + 0.5) / len(dbh_scale) for x in range(len(dbh_scale))]
            dbh2_obs = sorted(dbh_scale ** 2)
            if model == 'original':
                d_over_g = N0 / sum(dbh2_obs)
                isd_dist = ssnt_isd(d_over_g)                
            elif model == 'transform': 
                d_over_g = N0 / sum(dbh_scale ** (2/3))
                isd_dist = ssnt_isd_transform(d_over_g)
            elif model == 'bounded':
                d_over_g = N0 / (sum(dbh_scale) - N0) # Note here the analysis is on D, not D^2, even though the output var is still named dbh2_pred
                dbh2_obs = sorted(dbh_scale)
                isd_dist = ssnt_isd_bounded(d_over_g)
            dbh2_pred = isd_dist.ppf(scaled_rank)
            
            results = np.zeros((len(dbh2_obs), ), dtype = ('S15, f8, f8'))
            results['f0'] = np.array([site] * len(dbh2_obs))
            results['f1'] = dbh2_obs
            results['f2'] = dbh2_pred
            f1.writerows(results)
    f1_write.close()

def get_ssnt_obs_pred_isd_bounded_transform(raw_data, dataset_name, data_dir = './out_files/', cutoff = 9):
    """Obtain the observed dbh**2 and the values predicted by SSNT and write to file.
    
    Input:
    raw_data - data in the same format as obtained by wk.import_raw_data(), with 
        three columns site, sp, and dbh.
    dataset_name - name of the dataset for raw_data.
    model - whether the original model (ssnt_isd), the scaled model (ssnt_isd_transform), or 
        the lower truncated model (ssnt_isd_bounded) is adopted
    data_dir - directory for output file.
    cutoff - minimal number of species for a site to be included.
    
    """
    usites = np.sort(list(set(raw_data["site"])))
    f1_write = open(data_dir + dataset_name + '_obs_pred_isd_ssnt_bounded_transform.csv', 'wb')
    f1 = csv.writer(f1_write)
    
    for site in usites:
        subdat = raw_data[raw_data["site"] == site]
        dbh_raw = subdat[subdat.dtype.names[2]]
        dbh_scale = np.array(sorted(dbh_raw / min(dbh_raw)))
        dbh_scale_tranform = dbh_scale ** (2/3)
        S0 = len(set(subdat[subdat.dtype.names[1]]))
        N0 = len(dbh_scale)
        if S0 > cutoff:
            scaled_rank = [(x + 0.5) / len(dbh_scale) for x in range(len(dbh_scale))]
            dbh2_obs = sorted(dbh_scale ** 2)
            d_over_g = N0 / (sum(dbh_scale_tranform) - N0)
            isd_dist = ssnt_isd_bounded(d_over_g)
            dbh_transform_pred = isd_dist.ppf(scaled_rank)
            dbh_pred = dbh_transform_pred ** (3/2)
            
            results = np.zeros((len(dbh_scale), ), dtype = ('S15, f8, f8'))
            results['f0'] = np.array([site] * len(dbh_scale))
            results['f1'] = dbh_scale
            results['f2'] = dbh_pred
            f1.writerows(results)
    f1_write.close()
    
def plot_obs_pred_isd(datasets, model, data_dir = './out_files/', ax = None, radius = 2):
    """Plot the observed vs predicted ISD across multiple datasets"""
    if model == 'METE':
        isd_sites, isd_obs, isd_pred = wk.get_obs_pred_from_file(datasets, data_dir, '_obs_pred_isd_dbh2.csv')
    elif model == 'SSNT':
        isd_sites, isd_obs, isd_pred = wk.get_obs_pred_from_file(datasets, data_dir, '_obs_pred_isd_ssnt.csv')
    else: isd_sites, isd_obs, isd_pred = wk.get_obs_pred_from_file(datasets, data_dir, '_obs_pred_isd_ssnt_transform.csv')
    if not ax:
        fig = plt.figure(figsize = (3.5, 3.5))
        ax = plt.subplot(111)
    wk.plot_obs_pred(isd_obs, isd_pred, radius, 1, ax = ax)
    ax.set_xlabel('Predicted DBH^2', labelpad = 4, size = 8)
    ax.set_ylabel('Observed DBH^2', labelpad = 4, size = 8)
    return ax
 
def plot_joint(dat, model, ax = 'None', cbar = False):
    """Plot the density of the predicted joint distribution P(N, M) as heatmap
    
    and empirical data points as scatter on top.
    
    Inputs:
    dat - data array with 3 columns (site, sp, and m)
    model - "METE" or "SSNT"
    ax - whether the plot is part of an existing figure
    """
    S = len(np.unique(dat['sp']))
    N = len(dat)
    E = sum(dat[dat.dtype.names[2]])
    sp_abd_list = []
    sp_m_list = []
    for sp in np.unique(dat['sp']):
        dat_sp = dat[dat['sp'] == sp]
        sp_abd_list.append(len(dat_sp))
        sp_m_list.append(sum(dat_sp[dat.dtype.names[2]]))
    res = 200 # resolution
    seq_abd = np.logspace(np.log10(min(sp_abd_list)), np.log10(max(sp_abd_list)), num = res)
    seq_m = np.logspace(np.log10(min(sp_m_list)), np.log10(max(sp_m_list)), num = res)
    
    if model == 'SSNT':
            log_p = np.array([[lik_ssnt_sp(int(round(abd)), m, S, N, E) / S for abd in seq_abd] for m in seq_m])
    else:
        log_p = np.array([[lik_mete_sp(int(round(abd)), m, S, N, E) / S for abd in seq_abd] for m in seq_m])
    
    # Transforming log_p for better visualization
    log_p_trans = [-np.log(-x) for x in log_p]
    if not ax:
        fig = plt.figure(figsize = (3.5, 3.5))
        ax = plt.subplot(111)
    heatmap = plt.imshow(log_p_trans, interpolation = 'bilinear', cmap = 'YlOrRd', aspect = 'auto', origin = 'lower', \
               extent=[0.5 * min(sp_abd_list), 1.5 * max(sp_abd_list), 0.5 * min(sp_m_list), 1.5 * max(sp_m_list)])
    # Scatter plot of empirical data
    plt.scatter(sp_abd_list, sp_m_list, s = 8, c = 'black')
    # Set up both axes on log scale
    plt.xscale('log')
    plt.yscale('log')
    ax.tick_params(axis = 'both', which = 'major', labelsize = 6)
    plt.xlabel('Abundance', fontsize = 8)
    plt.ylabel('Total metabolic rate', fontsize = 8)    
    if cbar: 
        cbar_create = plt.colorbar(heatmap, ticks = [np.min(log_p_trans), np.max(log_p_trans)])
        cbar_create.ax.set_yticklabels(['low', 'high'])
    return ax

def plot_joint_ind(dat, model, ax = 'None', cbar = False):
    """Plot the density of the predicted joint distribution P(N, m) (R(n, epsilon) in METE) as heatmap
    
    and empirical data points as scatter on top.
    
    Inputs:
    dat - data array with 3 columns (site, sp, and diameter)
    model - "METE" or "SSNT"
    ax - whether the plot is part of an existing figure
    """
    S = len(np.unique(dat['sp']))
    N = len(dat)
    dbh = dat['dbh'] / min(dat['dbh'])
    sp_abd_list = []
    sp_m_list = []
    for sp in np.unique(dat['sp']):
        dat_sp = dat[dat['sp'] == sp]
        sp_abd_list.append(len(dat_sp))
    res = 200 # resolution
    seq_abd = np.round(np.logspace(np.log10(0.9 * min(sp_abd_list)), np.log10(1.5 * max(sp_abd_list)), num = res))
    seq_d = np.logspace(np.log10(0.5 * min(dbh)), np.log10(1.5 * max(dbh)), num = res)
    if model == 'SSNT': 
        d_over_g = N / (sum(dbh) - N)
        log_p = np.array([[lik_ssnt_ind(abd, d, S, N, d_over_g) / S for abd in seq_abd] for d in seq_d])
    else:
        E = sum(dbh**2)
        log_p = np.array([[lik_mete_ind(abd, d, S, N, E, unit = 'diameter') / S for abd in seq_abd] for d in seq_d])
        
    # Transforming log_p for better visualization
    log_p_trans = [-np.log(-x) for x in log_p]
    if not ax:
        fig = plt.figure(figsize = (3.5, 3.5))
        ax = plt.subplot(111)
    heatmap = plt.imshow(log_p_trans, interpolation = 'bilinear', cmap = 'YlOrRd', aspect = 'auto', origin = 'lower', \
               extent=[0.9 * min(sp_abd_list), 1.5 * max(sp_abd_list), 0.5 * min(dbh), 1.5 * max(dbh)])
    # Scatter plot of empirical data
    ind_abd_list = []
    for ind in dat:
        sp_ind = ind['sp']
        n_sp = len(dat[dat['sp'] == sp_ind])
        ind_abd_list.append(n_sp)
    plt.scatter(ind_abd_list, dbh, s = 8, c = 'black')
    # Set up both axes on log scale
    plt.xscale('log')
    plt.yscale('log')
    ax.tick_params(axis = 'both', which = 'major', labelsize = 6)
    plt.xlabel('Abundance', fontsize = 8)
    plt.ylabel('Diameter', fontsize = 8)    
    if cbar: 
        cbar_create = plt.colorbar(heatmap, ticks = [np.min(log_p_trans), np.max(log_p_trans)])
        cbar_create.ax.set_yticklabels(['low', 'high'])
    return ax