"""
==============================================
Phase-Amplitude Coupling tutorial on sEEG data
==============================================

In this example, we illustrate how to conduct a PAC analysis on real data. The
data used here are taken from :cite:`combrisson2017intentions`. The task is a
center-out motor task where the subject have to reach a target on a screen
with the mouse. A single trial consists in three periods :

    * From [-1000, 0]ms it is the baseline period (REST)
    * From [0, 1500]ms the subject have to prepare the movement (MOTOR
      PLANNING period)
    * From [1500, 4000]ms the subject perform the movement to reach the target
      on the screen (MOTOR EXECUTION period)

The recorded electrophysiological data comes from an epileptic subject with
electrodes deep inside the brain (intracranial EEG or
stereoelectroencephalography). Here, the provided data contains only a single
recording contact for one subject. This contact is located in the premotor
cortex.

When working with phase-amplitude coupling, there are basically three questions
you should try to answer :

    1. What is the range of phase frequencies supporting this coupling?
    2. Where in time this coupling occurs?
    3. What is the range of amplitude frequencies supporting this coupling?

In this tutorial we propose to answer to those three questions using following
structure :

    1. Compute the inter-trial coherence (ITC) and see if the phases are
       realigned at a given time point (:class:`tensorpac.utils.ITC`)
    2. Compute the Power Spectrum Density (PSD) to try to find the phase
       frequency range (:class:`tensorpac.utils.PSD`)
    3. Compute the Event-Related PAC (ERPAC) to have a first idea of where the
       coupling occurs in time, especially if it start at a given time point
       (:class:`tensorpac.EventRelatedPac`)
    4. Realign time-frequency representations based on the starting time-point
       found with the ERPAC and see if the gamma burst follow the rhythm
       imposed by a phase. This step should also gives and idea of the
       amplitude frequency range (:class:`tensorpac.utils.PeakLockedTF`)
    5. Compute the comodulogram and statistics in order to test if the
       hypothetic coupling can be considered as statistically different from
       PAC that could be obtained by chance (:class:`tensorpac.Pac`)
"""
import os
import urllib.request
import urllib

import numpy as np
from scipy.io import loadmat

from tensorpac import Pac, EventRelatedPac
from tensorpac.utils import PeakLockedTF, PSD, ITC

import matplotlib.pyplot as plt


###############################################################################
# Download the data
###############################################################################
# Lets first start by downloading the data. The file should be relatively
# small to download (1.18M). The file is going to be saved in the same folder
# where this script is launched. The data are downloaded only if the file is
# not present in the current folder


filename = os.path.join(os.getcwd(), 'seeg_data_pac.npz')
if not os.path.isfile(filename):
    print('Downloading the data')
    url = "https://www.dropbox.com/s/dn51xh7nyyttf33/seeg_data_pac.npz?dl=1"
    urllib.request.urlretrieve(url, filename=filename)

arch = np.load(filename)
data = arch['data']       # data of a single sEEG contact
sf = float(arch['sf'])    # sampling frequency
times = arch['times']     # time vector

print(f"DATA: (n_trials, n_times)={data.shape}; SAMPLING FREQUENCY={sf}Hz; "
      f"TIME VECTOR: n_times={len(times)}")


###############################################################################
# Plot the raw data
###############################################################################
# In this section, we simply plot the mean of the raw data across the 160
# trials

# function for adding the sections rest / planning / execution to each figure
def add_motor_condition(y_text, fontsize=14, color='k', ax=None):
    x_times = [-.5, 0.750, 2.250]
    x_conditions = ['REST', 'MOTOR\nPLANNING', 'MOTOR\nEXECUTION']
    if ax is None: ax = plt.gca()  # noqa
    plt.sca(ax)
    plt.axvline(0., lw=2, color=color)
    plt.axvline(1.5, lw=2, color=color)
    for x_t, t_t in zip(x_times, x_conditions):
        plt.text(x_t, y_text, t_t, color=color, fontsize=fontsize, ha='center',
                 va='center', fontweight='bold')


###############################################################################

plt.figure(figsize=(8, 6))
plt.plot(times, data.mean(0))
plt.autoscale(axis='x', tight=True)
plt.title("Mean raw data across trials of a premotor sEEG site", fontsize=18)
plt.xlabel('Times (in seconds)', fontsize=15)
plt.ylabel('V', fontsize=15)
plt.ylim(-600, 800)
add_motor_condition(700.)

plt.show()


###############################################################################
# Compute and plot Inter Trial Coherence
###############################################################################
# The Inter Trial Coherence (ITC) returns a factor that indicates how much
# the phases are consistent across trials. Here, we compute the ITC for
# multiple phase frequencies

itc = ITC(data, sf, f_pha=(2, 20, 1, .2))

###############################################################################

itc.plot(times=times, cmap='plasma', fz_labels=15, fz_title=18)
add_motor_condition(18, color='white')
plt.show()

###############################################################################
# For this sEEG site, we can see that the very low frequency phase (~3Hz) are
# realigned at the beginning of the execution period (~1500ms)


###############################################################################
# Compute and plot the Power Spectrum Density
###############################################################################
# Then, we compute the Power Spectrum Density (PSD) over all of the time-points
# and plot the mean PSD over the 160 trials

psd = PSD(data, sf)

###############################################################################

plt.figure(figsize=(8, 6))
ax = psd.plot(confidence=95, f_min=5, f_max=100, log=True, grid=True)
plt.axvline(8, lw=2, color='red')
plt.axvline(12, lw=2, color='red')
plt.show()

###############################################################################
# From the PSD above, we can see a clear peak around 10hz that could indicate
# an alpha <-> gamma coupling. This peak is essentially comprised between
# [8, 12]Hz. This range of frequencies is then gonig to be used to see if there
# is indeed an alpha <-> gamma coupling (Aru et al. :cite:`aru2015untangling`)

###############################################################################
# Compute and plot the Event-Related PAC
###############################################################################
# To go one step further we can use the Event-Related PAC (ERPAC) in order to
# isolate the gamma range that is coupled with the alpha phase such as when, in
# time, this coupling occurs. Here, we compute the ERPAC using the
# Gaussian-Copula mutual information # (Ince et al. 2017
# :cite:`ince2017statistical`), between the alpha [8, 12]Hz and several gamma
# amplitudes, at each time point.


rp_obj = EventRelatedPac(f_pha=[8, 12], f_amp=(30, 160, 30, 2))
erpac = rp_obj.filterfit(sf, data, method='gc', smooth=100)

###############################################################################

plt.figure(figsize=(8, 6))
rp_obj.pacplot(erpac.squeeze(), times, rp_obj.yvec, xlabel='Time',
               ylabel='Amplitude frequency (Hz)',
               title='Event-Related PAC occurring for alpha phase',
               fz_labels=15, fz_title=18)
add_motor_condition(135, color='white')
plt.show()

###############################################################################
# As you can see from the image above, there is an increase of alpha <-> gamma
# (~90Hz) coupling that is occurring especially during the planning phase (i.e
# between [0, 1500]ms)


###############################################################################
# Align time-frequency map based on alpha phase peak
###############################################################################
# to confirm the previous result showing a potential alpha <-> gamma coupling
# occuring during the planning phase, we can realign time-frequency
# representations (TFR) based on the alpha peak at the beginning of the
# planning phase (i.e at time code 0s)

peak = PeakLockedTF(data, sf, 0., times=times, f_pha=[8, 12],
                    f_amp=(5, 160, 30, 2))

###############################################################################

plt.figure(figsize=(8, 8))
ax_1, ax_2 = peak.plot(zscore=True, baseline=(250, 750), cmap='Spectral_r',
                       vmin=-1, vmax=2)
add_motor_condition(135, color='black', ax=ax_1)
plt.tight_layout()
plt.show()

###############################################################################
# From the TFR bellow we can see the relative to baseline gamma increase such
# as the beta desynchronization during the execution period, which is typical
# for a motor site. Once realign on alpha phase, we can also see that gamma
# burst are regularly spaced, following the alpha rhythm, especially the gamma
# in [40, 120]Hz. this confirm that indeed, there is the presence of
# alpha <-> gamma PAC occurring during the planning phase


###############################################################################
# Compute and compare PAC that is occurring during rest, planning and execution
###############################################################################
# we now know that an alpha [8, 12]Hz <-> gamma (~90Hz) should occur
# specifically during the planning phase. An other way to inspect this result
# is to compute the PAC, across time-points, during the rest, motor planning
# and motor execution periods. Bellow, we first extract several phases and
# amplitudes, then we compute the Gaussian-Copula PAC inside the three motor
# periods


p_obj = Pac(idpac=(6, 0, 0), f_pha=(6, 15, 4, .2), f_amp=(50, 120, 20, 2))
# extract all of the phases and amplitudes
pha_p = p_obj.filter(sf, data, ftype='phase')
amp_p = p_obj.filter(sf, data, ftype='amplitude')
# define time indices where rest, planning and execution are defined
time_rest = slice(0, 1000)
time_prep = slice(1000, 2500)
time_exec = slice(2500, 4000)
# compute PAC inside rest, planning, and execution
pac_rest = p_obj.fit(pha_p[..., time_rest], amp_p[..., time_rest]).mean(-1)
pac_prep = p_obj.fit(pha_p[..., time_prep], amp_p[..., time_prep]).mean(-1)
pac_exec = p_obj.fit(pha_p[..., time_exec], amp_p[..., time_exec]).mean(-1)


###############################################################################

vmax = np.max([pac_rest.max(), pac_prep.max(), pac_exec.max()])
kw = dict(vmax=vmax, vmin=.04, cmap='viridis')
plt.figure(figsize=(14, 4))
plt.subplot(131)
p_obj.comodulogram(pac_rest, title="PAC Rest [-1, 0]s", **kw)
plt.subplot(132)
p_obj.comodulogram(pac_prep, title="PAC Planning [0, 1.5]s", **kw)
plt.ylabel('')
plt.subplot(133)
p_obj.comodulogram(pac_exec, title="PAC Execution [1.5, 3]s", **kw)
plt.ylabel('')
plt.tight_layout()
plt.show()


###############################################################################
# From the three comodulograms above, you can see that, during the planning
# period there is an alpha [8, 12]Hz <-> gamma [80, 100]Hz that is not
# present during the rest and execution periods


###############################################################################
# Test if the alpha-gamma PAC is significant during motor planning
###############################################################################
# finally, here, we are going to test if the peak PAC that is occurring during
# the planning period is significantly different for a surrogate distribution.
# To this end, and as recommended by Aru et al. 2015, :cite:`aru2015untangling`
# the surrogate distribution is obtained by cutting an amplitude at a random
# time-point and then swap the two blocks of amplitudes (Bahramisharif et al.
# 2013, :cite:`bahramisharif2013propagating`). This procedure is then repeated
# multiple times (e.g 200 or 1000 times) in order to obtained the distribution.
# Finally, the p-value is inferred by computing the proportion exceeded by the
# true coupling. In addition, the correction for multiple comparison is
# obtained using the maximum statistics.

# still using the Gaussian-Copula PAC but this time, we also select the method
# for computing the permutations
p_obj.idpac = (6, 2, 0)
# compute pac and 200 surrogates
pac_prep = p_obj.fit(pha_p[..., time_prep], amp_p[..., time_prep], n_perm=50)
# get the p-values
pvalues = p_obj.infer_pvalues(p=0.05, mcp='maxstat')

###############################################################################

# sphinx_gallery_thumbnail_number = 7
plt.figure(figsize=(8, 6))
title = (r"Significant alpha$\Leftrightarrow$gamma coupling occurring during "
         "the motor planning phase\n(p<0.05, corrected for multiple "
         "comparisons)")
# plot the non-significant pac in gray
pac_prep_ns = pac_prep.mean(-1).copy()
pac_prep_ns[pvalues < .05] = np.nan
p_obj.comodulogram(pac_prep_ns, cmap='gray', vmin=np.nanmin(pac_prep_ns),
                   vmax=np.nanmax(pac_prep_ns), colorbar=False)
# plot the significant pac in color
pac_prep_s = pac_prep.mean(-1).copy()
pac_prep_s[pvalues >= .05] = np.nan
p_obj.comodulogram(pac_prep_s, cmap='Spectral_r', vmin=np.nanmin(pac_prep_s),
                   vmax=np.nanmax(pac_prep_s), title=title)
plt.gca().invert_yaxis()
plt.show()
