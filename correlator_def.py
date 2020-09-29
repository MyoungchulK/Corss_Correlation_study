import numpy as np
import os

# lag pad maker whan all wf are in same range of the pad
def lag_pad_maker(p_time_len, t_width):

    # lag pad
    lag_pad = np.arange(-p_time_len+1, p_time_len) * t_width
  
    return lag_pad

# numpy correlation w/o normalization
def cross_correlation(p_wf1, p_wf2):

    corr = np.correlate(p_wf1, p_wf2, 'full')

    return corr

def cross_correlation_w_bias_normalization(p_wf1, p_wf2, p_time_len):

    # rms product
    rms1 = np.nanstd(p_wf1)
    rms2 = np.nanstd(p_wf2)

    # mean subtracted wf -> subtraction only need to happen non zero value. If not, edge of correlation will be nasty
    p_wf1_mean = np.nanmean(p_wf1)
    p_wf1 = np.ma.masked_equal(p_wf1, 0) - p_wf1_mean
    p_wf1.mask = False
    del p_wf1_mean

    p_wf2_mean = np.nanmean(p_wf2)
    p_wf2 = np.ma.masked_equal(p_wf2, 0) - p_wf2_mean
    p_wf2.mask = False
    del p_wf2_mean

    # normal correlation procedure
    corr = cross_correlation(p_wf1, p_wf2)

    #normalization
    corr /= p_time_len
    corr /= (rms1 * rms2)
    del rms1, rms2

    return corr

def cross_correlation_w_unbias_normalization(p_wf1, p_wf2, p_time_len):

    # 01 wf array
    p_wf_01 = np.copy(p_wf1)
    p_wf_01[p_wf_01 != 0] = 1

    p_wf_02 = np.copy(p_wf2)
    p_wf_02[p_wf_02 != 0] = 1

    # rms product
    rms1 = np.nanstd(p_wf1)
    rms2 = np.nanstd(p_wf2)

    # mean subtracted wf -> subtraction only need to happen non zero value. If not, edge of correlation will be nasty
    p_wf1_mean = np.nanmean(p_wf1)
    p_wf1 = np.ma.masked_equal(p_wf1, 0)
    # lets still masked array for counting
    p_wf1_len = np.ma.count(p_wf1)
    p_wf1 -= p_wf1_mean
    p_wf1.mask = False
    del p_wf1_mean

    p_wf2_mean = np.nanmean(p_wf2)
    p_wf2 = np.ma.masked_equal(p_wf2, 0)
    # lets still masked array for counting
    p_wf2_len = np.ma.count(p_wf2)
    p_wf2 -= p_wf2_mean
    p_wf2.mask = False
    del p_wf2_mean    

    # get the longest wf length from combination
    p_wf_len = np.concatenate((p_wf1_len, p_wf2_len), axis=None)
    p_wf_len = np.nanmax(p_wf_len)
    del p_wf1_len, p_wf2_len

    # correlation. why it doesn't have AXIS option!!!
    corr = np.correlate(p_wf1, p_wf2, 'full')     
    corr01 = np.correlate(p_wf_01, p_wf_02, 'full')
    del p_wf_01, p_wf_02

    # normalization
    corr /= p_time_len
    corr /= (rms1 * rms2)
    del rms1, rms2

    # additional unbias normalization
    corr *= p_wf_len
    corr /= corr01
    #corr[corr01<2] = 0.
    del p_wf_len

    # removing nan and inf
    corr[np.isnan(corr)] = 0. #convert x/nan result
    corr[np.isinf(corr)] = 0. #convert nan/nan result

    return corr, corr01
    
