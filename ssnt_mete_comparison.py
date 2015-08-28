from __future__ import division
import matplotlib
matplotlib.use('Agg')
import os
import matplotlib.pyplot as plt
import csv
import numpy as np
from numpy.lib.recfunctions import append_fields
from scipy import stats
import working_functions as wk
import mete
import mete_distributions
import mete_agsne as agsne
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
        else: return 1 - np.exp(-self.par * (x ** self.alpha - 1))
    
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

def get_obs_pred_iisd_sdr(raw_data, dataset_name, alpha, out_dir = './out_files/', cutoff = 9):
    """This is the SSNT version of get_obs_pred_intradist() in module "working_functions". 
    
    To be consistent with METE, the SDR is recorded in the unit of D^2 (metabolic rate). 
    Keyword arguments:
    raw_data : numpy structured array with 3 columns: 'site','sp','dbh'
    dataset_name : short code that will indicate the name of the dataset in
                    the output file names
    out_dir : directory in which to store output
    cutoff : minimum number of species required to run - 1.
    n_cutoff: minimal number of individuals within a species to be included in iISD -1 
    
    """
    site_list = np.unique(raw_data['site'])
    f1_write = open(out_dir + dataset_name + '_' + str(round(alpha, 2)) + '_obs_pred_sdr.csv', 'wb')
    f1 = csv.writer(f1_write)
    f2_write = open(out_dir + dataset_name + '_' + str(round(alpha, 2)) + '_obs_pred_iisd.csv', 'wb')
    f2 = csv.writer(f2_write)

    for site in site_list:
        dat_site = raw_data[raw_data['site'] == site]
        S0 = len(np.unique(dat_site['sp']))
        if S0 > cutoff:
            sdr_obs, sdr_pred, iisd_obs, iisd_pred = [], [], [], []
            N0 = len(dat_site)
            dbh_raw = dat_site['dbh']
            dbh_scaled = np.array(dbh_raw / min(dbh_raw))
            par = N0 / (sum(dbh_scaled ** alpha) - N0)
            iisd_ssnt = ssnt_isd_bounded(alpha, par)
            
            for sp in np.unique(dat_site['sp']):
                dbh_sp = dbh_scaled[dat_site['sp'] == sp]
                scaled_rank_sp  =[(x + 0.5) / len(dbh_sp) for x in range(len(dbh_sp))]
                dbh_pred_sp = [iisd_ssnt.ppf(q) for q in scaled_rank_sp]
                sdr_obs.append(sum([dbh ** 2 for dbh in dbh_sp]) / len(dbh_sp))
                sdr_pred.append(2 / par ** 2 + 2 / par + 1) # Note that this now only works for alpha=1
                iisd_obs.extend(sorted(dbh_sp))
                iisd_pred.extend(sorted(dbh_pred_sp))

            results1 = np.zeros((len(sdr_obs), ), dtype = ('S15, f8, f8'))
            results1['f0'] = np.array([site] * len(sdr_obs))
            results1['f1'] = np.array(sdr_obs)
            results1['f2'] = np.array(sdr_pred)
            f1.writerows(results1)
            
            results2 = np.zeros((len(iisd_obs), ), dtype = ('S15, f8, f8'))
            results2['f0'] = np.array([site] * len(iisd_obs))
            results2['f1'] = np.array(iisd_obs)
            results2['f2'] = np.array(iisd_pred)
            f2.writerows(results2)
    f1_write.close()
    f2_write.close()
            
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

def plot_obs_pred_diameter(datasets, in_file_name, data_dir = './out_files/', ax = None, radius = 2, mete = False, title = None):
    """Plot the observed vs predicted diamters across multiple datasets. Applies to both ISD and iISD."""
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
    if title: plt.title(title, fontsize = 10)
    return ax

def plot_obs_pred_sad_sdr(datasets, in_file_name, data_dir = "./out_files/", ax = None, radius =2, title = None, axis_lab = 'abundance'):
    """Plot the observed vs predicted SAD or SDR for each species for multiple datasets."""
    sites, obs, pred = wk.get_obs_pred_from_file(datasets, data_dir, in_file_name)
    if not ax:
        fig = plt.figure(figsize = (3.5, 3.5))
        ax = plt.subplot(111)
    wk.plot_obs_pred(obs, pred, radius, 1, ax = ax)
    ax.set_xlabel('Predicted ' + axis_lab, labelpad = 4, size = 8)
    ax.set_ylabel('Observed ' + axis_lab, labelpad = 4, size = 8)
    if title: plt.title(title, fontsize = 10)
    return ax

def plot_likelihood_comp(lik_1, lik_2, xlabel, ylabel, annotate = True, ax = None):
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
    if annotate:
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

def get_sample_stats_isd_ssnt(obs, pred, dist):
    """Equivalent to get_sample_stats_isd() in module working_functions"""
    dat_rsquare = mtools.obs_pred_rsquare(np.log10(obs), np.log10(pred))
    dat_loglik = sum(np.log([dist.pdf(x) for x in obs]))
    dat_cdf = mtools.get_emp_cdf(obs)
    dat_ks = max(abs(dat_cdf - np.array([dist.cdf(x) for x in obs])))
    return dat_rsquare, dat_loglik, dat_ks

def bootstrap_ISD_SDR_iISD_SSNT(dat_name, alpha = 1, cutoff = 9, Niter = 500):
    """Compare the goodness of fit of the size-related patterns (ISD, iISD & SDR) to 
    
    that of the boostrapped samples from the proposed SSNT distributions.
    
    Inputs:
    dat_name - name of study
    cutoff - minimum number of species required to run - 1
    Niter - number of bootstrap samples
    """
    dat = wk.import_raw_data('./data/' + dat_name + '.csv')
    site_list = np.unique(dat['site'])
    dat_obs_pred_isd = wk.import_obs_pred_data('./out_files/' + dat_name + '_' + str(round(alpha, 2)) + '.csv')
    dat_obs_pred_sdr = wk.import_obs_pred_data('./out_files/' + dat_name + '_' + str(round(alpha, 2)) + '_obs_pred_sdr.csv')
    dat_obs_pred_iisd = wk.import_obs_pred_data('./out_files/' + dat_name + '_' + str(round(alpha, 2)) + '_obs_pred_iisd.csv')                                            

    if not os.path.exists('./out_files/iISD_bootstrap_ks/SSNT_' + str(round(alpha, 2)) + '/'):
        os.makedirs('./out_files/iISD_bootstrap_ks/SSNT_' + str(round(alpha, 2)) + '/')
                
    for site in site_list:
        dat_site = dat[dat['site'] == site]
        S_list = np.unique(dat_site['sp'])
        S0 = len(S_list)
        if S0 > cutoff:
            N0 = len(dat_site)
            dbh_raw = dat_site['dbh']
            dbh_scaled = np.array(dbh_raw / min(dbh_raw))
            par = N0 / (sum(dbh_scaled ** alpha) - N0)
            isd_ssnt = ssnt_isd_bounded(alpha, par)
             
            dat_site_obs_pred_isd = dat_obs_pred_isd[dat_obs_pred_isd['site'] == site]
            dat_site_obs_pred_sdr = dat_obs_pred_sdr[dat_obs_pred_sdr['site'] == site]
            dat_site_obs_pred_iisd = dat_obs_pred_iisd[dat_obs_pred_iisd['site'] == site]
            
            emp_isd_rsquare, emp_isd_loglik, emp_isd_ks = get_sample_stats_isd_ssnt(dat_site_obs_pred_isd['obs'], \
                                                                                    dat_site_obs_pred_isd['pred'], isd_ssnt)
            emp_sdr_rsquare = mtools.obs_pred_rsquare(np.log10(dat_site_obs_pred_sdr['obs']), \
                                                      np.log10(dat_site_obs_pred_sdr['pred']))
            emp_iisd_rsquare = mtools.obs_pred_rsquare(np.log10(dat_site_obs_pred_iisd['obs']), \
                                                       np.log10(dat_site_obs_pred_iisd['pred']))
           
            out_list_isd_rsquare = [dat_name, site, emp_isd_rsquare]
            out_list_isd_loglik = [dat_name, site, emp_isd_loglik]
            out_list_isd_ks = [dat_name, site, emp_isd_ks]
            out_list_sdr_rsquare = [dat_name, site, emp_sdr_rsquare]
            out_list_iisd_rsquare = [dat_name, site, emp_iisd_rsquare]
            
            emp_ks_list = []
            n_list = []
            for i, sp in enumerate(S_list):
                dbh_site_sp = dbh_scaled[dat_site['sp'] == sp]
                n_sp = len(dbh_site_sp)
                n_list.append(n_sp)
                
                emp_cdf = mtools.get_emp_cdf(dbh_site_sp)
                ks_sp = max(abs(emp_cdf - np.array([isd_ssnt.cdf(x) for x in dbh_site_sp])))
                emp_ks_list.append(ks_sp)
            
            out_ks_site = './out_files/iISD_bootstrap_ks/SSNT_' + str(round(alpha, 2)) + \
                '/iISD_bootstrap_ks_' + dat_name + '_' + str(round(alpha, 2)) + '_' + site + '.txt'
            wk.write_to_file(out_ks_site, ",".join(str(x) for x in n_list)) 
            wk.write_to_file(out_ks_site, ",".join(str(x) for x in emp_ks_list)) 
            
            for i in range(Niter):
                # Generate a sample from the predicte ISD
                rand_q = stats.uniform.rvs(size = N0)
                sim_ISD = np.array([isd_ssnt.ppf(x) for x in rand_q])
                sim_isd_rsquare, sim_isd_loglik, sim_isd_ks = get_sample_stats_isd_ssnt(np.sort(sim_ISD), \
                                                                                                   dat_site_obs_pred_isd['pred'], isd_ssnt)
                out_list_isd_rsquare.append(sim_isd_rsquare)
                out_list_isd_loglik.append(sim_isd_loglik)
                out_list_isd_ks.append(sim_isd_ks)
                
                sim_sdr_list, sim_ks_list, sim_iisd_list = [], [], []
                for i, sp in enumerate(S_list):
                    dbh_site_sp_sim = sim_ISD[dat_site['sp'] == sp]
                    sim_sdr_list.append(sum([x ** 2 for x in dbh_site_sp_sim]) / len(dbh_site_sp_sim))
                    sim_iisd_list.extend(sorted(dbh_site_sp_sim))
                    sim_cdf = mtools.get_emp_cdf(dbh_site_sp_sim)
                    ks_sp_sim = max(abs(sim_cdf - np.array([isd_ssnt.cdf(x) for x in dbh_site_sp_sim])))
                    sim_ks_list.append(ks_sp_sim)
                    
                wk.write_to_file(out_ks_site, ",".join(str(x) for x in sim_ks_list))
                out_list_iisd_rsquare.append(mtools.obs_pred_rsquare(np.log10(sim_iisd_list), \
                                                                     np.log10(dat_site_obs_pred_iisd['pred'])))
                out_list_sdr_rsquare.append(mtools.obs_pred_rsquare(np.log10(sim_sdr_list), \
                                                                    np.log10(dat_site_obs_pred_sdr['pred'])))
            
            wk.write_to_file('./out_files/ISD_bootstrap_rsquare_' + str(round(alpha, 2)) + '.txt', ",".join(str(x) for x in out_list_isd_rsquare))
            wk.write_to_file('./out_files/ISD_bootstrap_loglik_' + str(round(alpha, 2)) + '.txt', ",".join(str(x) for x in out_list_isd_loglik))
            wk.write_to_file('./out_files/ISD_bootstrap_ks_' + str(round(alpha, 2)) + '.txt', ",".join(str(x) for x in out_list_isd_ks))
            wk.write_to_file('./out_files/SDR_bootstrap_rsquare_' + str(round(alpha, 2)) + '.txt', ",".join(str(x) for x in out_list_sdr_rsquare))
            wk.write_to_file('./out_files/iISD_bootstrap_rsquare_' + str(round(alpha, 2)) + '.txt', ",".join(str(x) for x in out_list_iisd_rsquare))
            
def plot_bootstrap(alpha = 1):
    """Similar to create_Fig_E2() in working_functions.
    
    Add input "alpha" to adapt to output files for different transformations.
    
    """
    fig = plt.figure(figsize = (7, 14))
    sad_r2 = wk.import_bootstrap_file('./out_files/SAD_bootstrap_SSNT_rsquare.txt', Niter = 200)
    ax_1 = plt.subplot(421)
    wk.plot_hist_quan(sad_r2, ax = ax_1)
    plt.xlabel('Quantile', fontsize = 8)
    plt.ylabel('Frequency', fontsize = 8)
    plt.title(r'SAD, $R^2$', fontsize = 10)
 
    sad_ks = wk.import_bootstrap_file('./out_files/SAD_bootstrap_SSNT_ks.txt', Niter = 200)
    ax_2 = plt.subplot(422)
    wk.plot_hist_quan(sad_ks, dat_type = 'ks', ax = ax_2)
    plt.xlabel('Quantile', fontsize = 8)
    plt.ylabel('Frequency', fontsize = 8)
    plt.title('SAD, K-S Statistic', fontsize = 10)
 
    isd_r2 = wk.import_bootstrap_file('./out_files/ISD_bootstrap_rsquare_' + str(round(alpha, 2)) + '.txt', Niter = 200)
    ax_3 = plt.subplot(423)
    wk.plot_hist_quan(isd_r2, ax = ax_3)
    plt.xlabel('Quantile', fontsize = 8)
    plt.ylabel('Frequency', fontsize = 8)
    plt.title(r'ISD, $R^2$', fontsize = 10)
 
    isd_ks = wk.import_bootstrap_file('./out_files/ISD_bootstrap_ks_' + str(round(alpha, 2)) + '.txt', Niter = 200)
    ax_4 = plt.subplot(424)
    wk.plot_hist_quan(isd_ks, dat_type = 'ks', ax = ax_4)
    plt.xlabel('Quantile', fontsize = 8)
    plt.ylabel('Frequency', fontsize = 8)
    plt.title('ISD, K-S Statistic', fontsize = 10)

    iisd_r2 = wk.import_bootstrap_file('./out_files/iISD_bootstrap_rsquare_' + str(round(alpha, 2)) + '.txt', Niter = 200)
    ax_5 = plt.subplot(425)
    wk.plot_hist_quan(iisd_r2, ax = ax_5)
    plt.xlabel('Quantile', fontsize = 8)
    plt.ylabel('Frequency', fontsize = 8)
    plt.title(r'iISD, $R^2$', fontsize = 10)
   
    ax_6 = plt.subplot(426)
    wk.plot_hist_quan_iisd_ks('./out_files/iISD_bootstrap_ks/SSNT_' + str(round(alpha, 2)) + '/', ax = ax_6)
    plt.xlabel('Quantile', fontsize = 8)
    plt.ylabel('Frequency', fontsize = 8)
    plt.title('iISD, K-S Statistic', fontsize = 10)
    
    sdr_r2 = wk.import_bootstrap_file('./out_files/SDR_bootstrap_rsquare_' + str(round(alpha, 2)) + '.txt', Niter = 200)
    ax_7 = plt.subplot(427)
    wk.plot_hist_quan(sdr_r2, ax = ax_7)
    plt.xlabel('Quantile', fontsize = 8)
    plt.ylabel('Frequency', fontsize = 8)
    plt.title(r'SDR, $R^2$', fontsize = 10)
   
    plt.subplots_adjust(wspace = 0.29, hspace = 0.29)
    plt.savefig('Bootstrap_SSNT_' + str(round(alpha, 2)) + '_200.pdf', dpi = 600)

def clean_data_agsne(raw_data_site, cutoff_genera = 4, cutoff_sp = 9, max_removal = 0.1):
    """Further cleanup of data, removing individuals with undefined genus. 
    
    Inputs:
    raw_data_site - structured array generated by wk.import_raw_data(), with three columns 'site', 'sp', and 'dbh', for a single site
    min_genera - minimal number of genera required for analysis
    min_sp - minimal number of species
    max_removal - the maximal proportion of individuals removed with undefined genus
    
    Output:
    a structured array with four columns 'site', 'sp', 'genus', and 'dbh'
    
    """
    counter = 0
    genus_list = []
    row_to_remove = []
    for i, row in enumerate(raw_data_site):
        sp_split = row['sp'].split(' ')
        genus = sp_split[0]
        if len(sp_split) > 1 and genus[0].isupper() and genus[1].islower() and (not any(char.isdigit() for char in genus)):
            genus_list.append(genus)
        else: 
            row_to_remove.append(i)
            counter += 1
    if counter / len(raw_data_site) <= max_removal:
        raw_data_site = np.delete(raw_data_site, np.array(row_to_remove), axis = 0)
        gen_col = np.array(genus_list)
        out = append_fields(raw_data_site, 'genus', gen_col, usemask = False)
        if len(np.unique(out['sp'])) > cutoff_sp and len(np.unique(out['genus'])) > cutoff_genera: 
            return out
        else: return None
    else: return None

def get_GSNE(raw_data_site):
    """Obtain the state variables given data for a single site, returned by clean_data_genera()."""
    G = len(np.unique(raw_data_site['genus']))
    S = len(np.unique(raw_data_site['sp']))
    N = len(raw_data_site)
    E = sum((raw_data_site['dbh'] / min(raw_data_site['dbh'])) ** 2)
    return G, S, N, E

def get_agsne_obs_pred_sad(raw_data_site, dataset_name, out_dir = './out_files/'):
    """Write the observed and AGSNE-predicted SAD to file. Here it is assumed that the input data (raw_data_site)
    
    has already gone through screening and cleaning (thus cutoff is no longer needed).
    Inputs:
    raw_data_site - data in the same format as obtained by clean_data_genera(), with
        four columns site, sp, dbh, and genus, and only for one site.
    dataset_name - name of the dataet for raw_data_site.
    out_dir - directory for output file.
    
    """
    G, S, N, E = get_GSNE(raw_data_site)
    pred = agsne.get_mete_agsne_rad(G, S, N, E)
    obs = np.sort([len(raw_data_site[raw_data_site['sp'] == sp]) for sp in np.unique(raw_data_site['sp'])])[::-1]
    results = np.zeros((S, ), dtype = ('S15, i8, i8'))
    results['f0'] = np.array([raw_data_site['site'][0]] * S)
    results['f1'] = obs
    results['f2'] = pred    
    
    f1_write = open(out_dir + dataset_name + '_obs_pred_rad_agsne.csv', 'ab')
    f1 = csv.writer(f1_write)
    f1.writerows(results)
    f1_write.close()

def get_agsne_obs_pred_isd(raw_data_site, dataset_name, out_dir = './out_files/'):
    """Write the observed and AGSNE-predicted ISD to file. Here it is assumed that the input data (raw_data_site)
    
    has already gone through screening and cleaning (thus cutoff is no longer needed).
    For inputs see get_agsne_obs_pred_sad(). 
    
    """
    G, S, N, E = get_GSNE(raw_data_site)
    pred = np.array(agsne.get_mete_agsne_isd(G, S, N, E)) ** 0.5 # Note the prediction is for metabolic rate, or D^2
    obs = np.sort(raw_data_site['dbh'] / min(raw_data_site['dbh']))[::-1]
    results = np.zeros((N, ), dtype = ('S15, f8, f8'))
    results['f0'] = np.array([raw_data_site['site'][0]] * N)
    results['f1'] = obs
    results['f2'] = pred    
    
    f1_write = open(out_dir + dataset_name + '_obs_pred_isd_agsne.csv', 'ab')
    f1 = csv.writer(f1_write)
    f1.writerows(results)
    f1_write.close()

def get_agsne_obs_pred_sdr(raw_data_site, dataset_name, out_dir = './out_files/'):
    """Write the observed and AGSNE-predicted size-density relationship to file. Here it is assumed that the input data (raw_data_site)
    
    has already gone through screening and cleaning (thus cutoff is no longer needed).
    For inputs see get_agsne_obs_pred_sad(). 
    
    """
    G, S, N, E = get_GSNE(raw_data_site)
    lambda1, beta, lambda3 = agsne.get_agsne_lambdas(G, S, N, E)
    theta = mete_distributions.theta_agsne([G, S, N, E], [lambda1, beta, lambda3, agsne.agsne_lambda3_z(lambda1, beta, S) / lambda3])
    
    pred, obs = [], []
    scaled_d2 = (raw_data_site['dbh'] / min(raw_data_site['dbh'])) ** 2
    for sp in np.unique(raw_data_site['sp']):
        n = len(raw_data_site[raw_data_site['sp'] == sp]) # Number of individuals within species
        genus_sp = raw_data_site['genus'][raw_data_site['sp'] == sp][0]
        m = len(np.unique(raw_data_site['sp'][raw_data_site['genus'] == genus_sp])) # Number of specis within genus
        pred.append(theta.expected(m, n))
        obs.append(np.mean(scaled_d2[raw_data_site['sp'] == sp]))
    
    results = np.zeros((S, ), dtype = ('S15, f8, f8'))
    results['f0'] = np.array([raw_data_site['site'][0]] * S)
    results['f1'] = obs
    results['f2'] = pred    
    f1_write = open(out_dir + dataset_name + '_obs_pred_sdr_agsne.csv', 'ab')
    f1 = csv.writer(f1_write)
    f1.writerows(results)
    f1_write.close()
    