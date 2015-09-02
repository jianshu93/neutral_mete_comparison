from __future__ import division
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import ssnt_mete_comparison as smc
import working_functions as wk
import mete_distributions as medis
import numpy as np
import multiprocessing

dat_list = ['ACA', 'BCI', 'BVSF', 'CSIRO', 'FERP', 'Lahei', 'LaSelva', 'NC', 'Oosting', 'Serimbu', 
            'WesternGhats', 'Cocoli', 'Luquillo', 'Sherman', 'Shirakami']

# Obtain pred-obs for three patterns * four models 
dat_list_keep = []
dat_site_list = []
for dat_name in dat_list:
    dat = wk.import_raw_data('./data/' + dat_name + '.csv')
    for site in np.unique(dat['site']):
        dat_site = dat[dat['site'] == site]
        dat_clean = smc.clean_data_agsne(dat_site)
        if dat_clean is not None:
            dat_list_keep.append(dat_name)
            dat_site_list.append([dat_name, site])
            
            smc.get_obs_pred_sad(dat_clean, dat_name, 'ssnt')
            smc.get_obs_pred_sad(dat_clean, dat_name, 'asne')
            smc.get_obs_pred_sad(dat_clean, dat_name, 'agsne')
            
            smc.get_obs_pred_isd(dat_clean, dat_name, 'ssnt_0')
            smc.get_obs_pred_isd(dat_clean, dat_name, 'ssnt_1')
            smc.get_obs_pred_isd(dat_clean, dat_name, 'asne')
            smc.get_obs_pred_isd(dat_clean, dat_name, 'agsne')            
            
            smc.get_obs_pred_sdr(dat_clean, dat_name, 'ssnt_0')
            smc.get_obs_pred_sdr(dat_clean, dat_name, 'ssnt_1')
            smc.get_obs_pred_sdr(dat_clean, dat_name, 'asne')
            smc.get_obs_pred_sdr(dat_clean, dat_name, 'agsne')            

# Plot the predicted versus observed values for the three patterns and the four models
smc.plot_obs_pred_four_models(dat_list)

# Boostrap analyses
model_list = ['ssnt_0', 'ssnt_1', 'asne', 'agsne']
for model in model_list:
    def model_boot_sad(name_site_combo):
        smc.bootstrap_SAD(name_site_combo, model, Niter = 500)
    pool = multiprocessing.Pool(8)  # Assuming that there are 8 cores
    pool.map(model_boot_sad, dat_site_list)
    pool.close()
    pool.join()

for model in model_list:
    def model_boot_isd(name_site_combo):
        smc.bootstrap_ISD(name_site_combo, model, Niter = 500)
    pool = multiprocessing.Pool(8)  # Assuming that there are 8 cores
    pool.map(model_boot_isd, dat_site_list)
    pool.close()
    pool.join()

for model in model_list:
    def model_boot_sdr(name_site_combo):
        smc.bootstrap_SDR(name_site_combo, model, Niter = 500)
    pool = multiprocessing.Pool(8)  # Assuming that there are 8 cores
    pool.map(model_boot_sdr, dat_site_list)
    pool.close()
    pool.join()

# Plot results from bootstrap analysis for four models
for model in model_list:
    smc.plot_bootstrap(dat_list, model)

# Obtain and plot likelihood comparison among models
dat_list_keep = []
dat_site_list = []
for dat_name in dat_list:
    dat = wk.import_raw_data('./data/' + dat_name + '.csv')
    for site in np.unique(dat['site']):
        dat_site = dat[dat['site'] == site]
        dat_clean = smc.clean_data_agsne(dat_site)
        if dat_clean is not None:
            smc.get_lik_sp_abd_dbh_four_models(dat_clean, dat_name)

smc.plot_likelihood_comp()
