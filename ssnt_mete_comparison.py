from __future__ import division
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv
import numpy as np
from scipy import stats
import working_functions as wk
import mete
import mete_distributions
import macroecotools as mtools
import macroeco_distributions as md

class ssnt_isd_bounded():
    """The individual-size distribution predicted by SSNT.
    
    Diameter is assumed to be lower-bounded at 1, to be consistent
    with METE.
    SSNT is applied on diameter transformed with an arbitrary
    power alpha, i.e., it is assumed that g(D^alpha) is constant.
    The predicted distribution is transformed back to diameter.
    
    """
    def __init__(self, alpha, par):
        """par is the parameter for the lower-truncated exponential
        
        distribution when the scale is D^alpha. 
        The MLE of par is N / (sum(D^alpha) - N) from data.
        
        """
        self.alpha = alpha
        self.par = par
        self.a = 1 # lower bound
        
    def pdf(self, x):
        if x < self.a: return 0
        else: return self.par * self.alpha * np.exp(-self.par * (x ** self.alpha - 1)) * (x ** (self.alpha - 1))
    
    def cdf(self, x): # cdf of D is equal to cdf of D^alpha
        if x < self.a: return 0
        else: return 1 - np.exp(-np.par * (x ** self.alpha - 1))
    
    def ppf(self, q):
        return (1 - np.log(1 - q) / self.par) ** (1 / self.alpha)

def import_likelihood_data(file_name, file_dir = './out_files/'):
    """Import file with likelihood for METE, SSNT, and transformed SSNT"""
    data = np.genfromtxt(file_dir + file_name, dtype = 'S15, S15, f15, f15, f15', 
                         names = ['study', 'site', 'METE', 'SSNT', 'SSNT_transform'], delimiter = ' ')
    return data
    
def lik_sp_abd_dbh_ssnt(sad_par, isd_dist, n, dbh_list, log = True):
    """Probability of a species having abundance n and its individuals having dbh [d1, d2, ..., d_n] in SSNT
    
    Inputs:
    sad_par - parameter of the predicted SAD (untruncated logseries)
    isd_dist - predicted distribution of the ISD
    n - abundance
    dbh_list - a list or array of length n with scaled dbh values
    """
    p_sad_log = stats.logser.logpmf(n, sad_par)
    p_dbh = [isd_dist.pdf(dbh) for dbh in dbh_list]
    if log: return p_sad_log + sum([np.log(p_ind) for p_ind in p_dbh])
    else: 
        p_iisd = 1
        for p_ind in p_dbh: p_iisd *= p_ind
        return np.exp(p_sad_log) * p_iisd
    
def lik_sp_abd_dbh_mete(sad_par, sad_upper, iisd_dist, n, dbh_list, log = True):
    """Probability of a species having abundance n and its individuals having dbh [d1, d2, ..., d_n] in METE
    
    Here unlike SSNT, P(d|n) is not equal to the ISD f(d). 
    Inputs:
    sad_par - parameter of the predicted SAD (upper-truncated logseries)
    sad_upper - upper bounded of the predicted SAD
    isd_dist - predicted iISD given n (theta_epsilon)
    n - abundance
    dbh_list - a list or array of length n with scaled dbh values
    """
    p_sad_log = md.trunc_logser.logpmf(n, sad_par, sad_upper)
    p_dbh_log = [iisd_dist.logpdf(dbh ** 2, n) + np.log(2 * dbh) for dbh in dbh_list] # Prediction of METE has to be transformed back to distribution of dbh
    if log: return p_sad_log + sum(p_dbh_log)
    else: 
        p_iisd = 1
        for p_ind in p_dbh_log: p_iisd *= np.exp(p_ind)
        return np.exp(p_sad_log) * p_iisd
    
def get_ssnt_obs_pred_sad(raw_data, dataset_name, out_dir = './out_files/', cutoff = 9):
    """Write the observed and predicted RAD to file. Note that the predicted form of the SAD
    
    (untruncated logseries) does not change with scaling of diameters.
    Inputs:
    raw_data - data in the same format as obtained by wk.import_raw_data(), with
        three columsn site, sp, and dbh.
    dataset_name - name of the dataet for raw_data.
    out_dir - directory for output file.
    cutoff - minimal number of species for a site to be included.
    
    """
    usites = np.sort(list(set(raw_data["site"])))
    f1_write = open(out_dir + dataset_name + '_obs_pred_rad_ssnt.csv', 'wb')
    f1 = csv.writer(f1_write)
    
    for i in range(0, len(usites)):
        subsites = raw_data["site"][raw_data["site"] == usites[i]]        
        subsp = raw_data["sp"][raw_data["site"] == usites[i]]
        N = len(subsp)
        S = len(set(subsp))
        subab = []
        for sp in set(subsp):
            subab.append(len(subsp[subsp == sp]))
        if S > cutoff:
            # Generate predicted values and p (e ** -beta) based on METE:
            mete_pred = mete.get_mete_rad(int(S), int(N), version = 'untruncated')
            pred = np.array(mete_pred[0])
            obsab = np.sort(subab)[::-1]
            #save results to a csv file:
            results = np.zeros((len(obsab), ), dtype = ('S15, i8, i8'))
            results['f0'] = np.array([usites[i]] * len(obsab))
            results['f1'] = obsab
            results['f2'] = pred
            f1.writerows(results)
    f1_write.close()
    
def get_ssnt_obs_pred_isd(raw_data, dataset_name, alpha, out_dir = './out_files/', cutoff = 9):
        """Write the observed (with rescaling) and predicted dbh to file.
        
        Input:
        raw_data - data in the same format as obtained by wk.import_raw_data(), with 
            three columns site, sp, and dbh.
        dataset_name - name of the dataset for raw_data.
        alpha - transformation on dbh where SSNT is applied
        out_dir - directory for output file.
        cutoff - minimal number of species for a site to be included.
        
        """
        site_list = np.unique(raw_data['site'])
        usites = np.sort(list(set(raw_data["site"])))
        out_name = dataset_name + '_' + str(round(alpha, 2)) + '.csv'
        f_write = open(out_dir + out_name, 'wb')
        f = csv.writer(f_write)
        
        for site in site_list:
            dat_site = raw_data[raw_data['site'] == site]
            S0 = len(np.unique(dat_site['sp']))
            if S0 > cutoff:
                N0 = len(dat_site)
                dbh_raw = dat_site['dbh']
                dbh_scaled = np.array(sorted(dbh_raw / min(dbh_raw)))
                par = N0 / (sum(dbh_scaled ** alpha) - N0)
                
                scaled_rank = [(x + 0.5) / len(dbh_raw) for x in range(len(dbh_raw))]
                isd_ssnt = ssnt_isd_bounded(alpha, par)
                dbh_pred = np.array([isd_ssnt.ppf(q) for q in scaled_rank])
                
                results = np.zeros((len(dbh_raw), ), dtype = ('S15, f8, f8'))
                results['f0'] = np.array([site] * len(dbh_raw))
                results['f1'] = dbh_scaled
                results['f2'] = dbh_pred
                f.writerows(results)
        f_write.close()

def get_isd_lik_three_models(dat_list, out_dir = './out_files/', cutoff = 9):
    """Function to obtain the community-level log-likelihood (standardized by the number of individuals)
    
    as well as AICc values for METE, SSNT on D, and SSNT on D**(2/3) and write to files. 
    
    """
    for dat_name in dat_list:
        dat = wk.import_raw_data('./data/' + dat_name + '.csv')
        for site in np.unique(dat['site']):
            dat_site = dat[dat['site'] == site]
            S0 = len(np.unique(dat_site['sp']))
            if S0 > cutoff:
                N0 = len(dat_site)
                dbh_scaled = dat_site['dbh'] / min(dat_site['dbh'])
                psi = mete_distributions.psi_epsilon(S0, N0, sum(dbh_scaled ** 2))
                ssnt_isd = ssnt_isd_bounded(1, N0 / (sum(dbh_scaled) - N0))
                ssnt_isd_transform = ssnt_isd_bounded(2/3, N0 / (sum(dbh_scaled ** (2/3)) - N0))
                
                lik_mete, lik_ssnt, lik_ssnt_transform = 0, 0, 0
                for dbh in dbh_scaled:
                    lik_mete += np.log(psi.pdf(dbh ** 2) * 2 * dbh) # psi is on dbh**2
                    lik_ssnt += np.log(ssnt_isd.pdf(dbh))
                    lik_ssnt_transform += np.log(ssnt_isd_transform.pdf(dbh))
                out1 = open(out_dir + 'isd_lik_three_models.txt', 'a')
                print>>out1, dat_name, site, str(lik_mete / N0), str(lik_ssnt / N0), str(lik_ssnt_transform / N0)
                out1.close()
                
                out2 = open(out_dir + 'isd_aicc_three_models.txt', 'a')
                # METE has three parameters (S0, N0, E0) for ISD, while SSNT has two (N0 and sum(dbh**alpha))
                print>>out2, dat_name, site, str(mtools.AICc(lik_mete, 3, N0)), str(mtools.AICc(lik_ssnt, 2, N0)), \
                     str(mtools.AICc(lik_ssnt_transform, 2, N0))
                out2.close()
                
def get_lik_sp_abd_dbh_three_models(dat_list, out_dir = './out_files/', cutoff = 9):
    """Obtain the summed log likelihood of each species having abundance n and its individuals having 
    
    their specific dbh values for the three models METE, SSNT on D, and SSNT on D ** (2/3).
    
    """
    for dat_name in dat_list:
        dat = wk.import_raw_data('./data/' + dat_name + '.csv')
        for site in np.unique(dat['site']):
            dat_site = dat[dat['site'] == site]
            S0 = len(np.unique(dat_site['sp']))
            if S0 > cutoff:
                N0 = len(dat_site)
                dbh_scaled = dat_site['dbh'] / min(dat_site['dbh'])
                theta = mete_distributions.theta_epsilon(S0, N0, sum(dbh_scaled ** 2))
                lambda_mete = np.exp(-mete.get_beta(S0, N0))
                lambda_ssnt = np.exp(-mete.get_beta(S0, N0, version = 'untruncated'))
                ssnt_isd = ssnt_isd_bounded(1, N0 / (sum(dbh_scaled) - N0))
                ssnt_isd_transform = ssnt_isd_bounded(2/3, N0 / (sum(dbh_scaled ** (2/3)) - N0))
                
                lik_mete, lik_ssnt, lik_ssnt_transform = 0, 0, 0
                for sp in np.unique(dat_site['sp']):
                    dbh_sp = dbh_scaled[dat_site['sp'] == sp]
                    n_sp = len(dbh_sp)
                    lik_mete += lik_sp_abd_dbh_mete(lambda_mete, N0, theta, n_sp, dbh_sp)
                    lik_ssnt += lik_sp_abd_dbh_ssnt(lambda_ssnt, ssnt_isd, n_sp, dbh_sp)
                    lik_ssnt_transform += lik_sp_abd_dbh_ssnt(lambda_ssnt, ssnt_isd_transform, n_sp, dbh_sp)
                out = open(out_dir + 'lik_sp_abd_dbh_three_models.txt', 'a')
                print>>out, dat_name, site, str(lik_mete), str(lik_ssnt), str(lik_ssnt_transform)
                out.close()

def plot_obs_pred_diameter(datasets, in_file_name, data_dir = './out_files/', ax = None, radius = 2, mete = False):
    """Plot the observed vs predicted ISD across multiple datasets"""
    isd_sites, isd_obs, isd_pred = wk.get_obs_pred_from_file(datasets, data_dir, in_file_name)
    if mete:
        isd_obs = isd_obs ** 0.5
        isd_pred = isd_pred ** 0.5
    if not ax:
        fig = plt.figure(figsize = (3.5, 3.5))
        ax = plt.subplot(111)
    wk.plot_obs_pred(isd_obs, isd_pred, radius, 1, ax = ax)
    ax.set_xlabel('Predicted diameter', labelpad = 4, size = 8)
    ax.set_ylabel('Observed diameter', labelpad = 4, size = 8)
    return ax

def plot_likelihood_comp(lik_1, lik_2, xlabel, ylabel, ax = None):
    """Plot the likelihood two models against each other.
    
    lik_1 and lik_2 are two lists/arrays of the same length, each 
    representing likelihood in each community for one model.
    
    """
    if not ax:
        fig = plt.figure(figsize = (3.5, 3.5))
        ax = plt.subplot(111)
    min_val, max_val = min(list(lik_1) + list(lik_2)), max(list(lik_1) + list(lik_2))
    if min_val < 0: axis_min = 1.1 * min_val
    else: axis_min = 0.9 * min_val
    if max_val < 0: axis_max = 0.9 * max_val
    else: axis_max= 1.1 * max_val
    plt.scatter(lik_1, lik_2, c = '#787878', edgecolors='none')
    plt.plot([axis_min, axis_max], [axis_min, axis_max], 'k-')     
    plt.xlim(axis_min, axis_max)
    plt.ylim(axis_min, axis_max)
    ax.tick_params(axis = 'both', which = 'major', labelsize = 6)
    ax.set_xlabel(xlabel, labelpad = 4, size = 8)
    ax.set_ylabel(ylabel, labelpad = 4, size = 8)
    num_above_line = len([i for i in range(len(lik_1)) if lik_1[i] < lik_2[i]])
    plt.annotate('Above the line: ' + str(num_above_line) + '/' + str(len(lik_1)), xy = (0.05, 0.85), 
                 xycoords = 'axes fraction', fontsize = 7)
    return ax

def get_sample_stats_sad_ssnt(obs, pred, p):
    """The SSNT version of get_sample_stats_sad()"""
    dat_rsquare = mtools.obs_pred_rsquare(np.log10(obs), np.log10(pred))
    dat_loglik = sum(np.log([stats.logser.pmf(x, p) for x in obs]))
    emp_cdf = mtools.get_emp_cdf(obs)
    dat_ks = max(abs(emp_cdf - np.array([stats.logser.cdf(x, p) for x in obs])))
    return dat_rsquare, dat_loglik, dat_ks

def bootstrap_SAD_SSNT(dat_name, cutoff = 9, Niter = 500):
    """Compare the goodness of fit of the empirical SAD to 
    
    that of the boostrapped samples from the proposed SSNT distribution.
    Note that both versions of the SSNT predict the same form for the SAD and 
    thus are not distinguished here.
    Inputs:
    dat_name - name of study
    cutoff - minimum number of species required to run - 1
    Niter - number of bootstrap samples
    """
    dat = wk.import_raw_data('./data/' + dat_name + '.csv')
    site_list = np.unique(dat['site'])
    dat_obs_pred = wk.import_obs_pred_data('./out_files/' + dat_name + '_obs_pred_rad_ssnt.csv')
        
    for site in site_list:
        out_list_rsquare, out_list_loglik, out_list_ks = [dat_name, site], [dat_name, site], [dat_name, site]
        dat_site = dat[dat['site'] == site]
        S_list = set(dat_site['sp'])
        S0 = len(S_list)
        if S0 > cutoff:
            N0 = len(dat_site)
            beta = mete.get_beta(S0, N0, version = 'untruncated')
            
            dat_site_obs_pred = dat_obs_pred[dat_obs_pred['site'] == site]
            dat_site_obs = dat_site_obs_pred['obs']
            dat_site_pred = dat_site_obs_pred['pred']
            
            emp_rsquare, emp_loglik, emp_ks = get_sample_stats_sad_ssnt(dat_site_obs, dat_site_pred, np.exp(-beta))
            out_list_rsquare.append(emp_rsquare)
            out_list_loglik.append(emp_loglik)
            out_list_ks.append(emp_ks)
            
            for i in range(Niter):
                sample_i = sorted(stats.logser.rvs(np.exp(-beta), size = S0), reverse = True)
                sample_rsquare, sample_loglik, sample_ks = get_sample_stats_sad_ssnt(sample_i, dat_site_pred, np.exp(-beta))
                out_list_rsquare.append(sample_rsquare)
                out_list_loglik.append(sample_loglik)
                out_list_ks.append(sample_ks)
  
            wk.write_to_file('./out_files/SAD_bootstrap_SSNT_rsquare.txt', ",".join(str(x) for x in out_list_rsquare))
            wk.write_to_file('./out_files/SAD_bootstrap_SSNT_loglik.txt', ",".join(str(x) for x in out_list_loglik))
            wk.write_to_file('./out_files/SAD_bootstrap_SSNT_ks.txt', ",".join(str(x) for x in out_list_ks))
