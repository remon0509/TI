import numpy as np
import pandas as pd

def ATR(close, high, low, n=14, pad_with_na=True):
    prev_close = np.zeros(len(close))
    prev_close[1:] = close[:-1]

    true_range = (np.vstack([high - low, abs(high - prev_close), abs(low - prev_close)]).T).max(axis=1)[1:]
    first_atr = true_range[:n].mean()
    atr_ls = []
    for i in range(len(true_range)):
        if i < n - 1:
            pass
        else:
            atr_t = (atr_prev*(n - 1) + true_range[i])/n if i > (n - 1) else first_atr
            atr_ls.append(atr_t)
            atr_prev = atr_t
    if pad_with_na:
        pad = np.zeros(n)
        pad[:] = np.nan
        atr_ls = np.append(pad, atr_ls)
    return atr_ls


def SMA(close, n, pad_with_na=True):
    # close: np.array()
    tmp = np.zeros((len(close), n))
    for i in range(n):
        tmp[i:, i] = close[:-i] if i != 0 else close
    sma = tmp[(n - 1):].mean(axis=1)
    if pad_with_na:
        pad = np.zeros(n - 1)
        pad[:] = np.nan
        sma = np.append(pad, sma)
    return sma



def EMA(close, n, pad_with_na=True, clear_na=True):
    # close: np.array()
    if clear_na:
        extra_pad = len(close) - len(close[~np.isnan(close)])
        close = close[~np.isnan(close)]
    else:
        extra_pad = 0

    alpha = 2/(n + 1)
    ema = []
    first_ema = close[:n].mean()
    for i in range(n - 1, len(close)):
        ema_t = alpha*close[i] + (1 - alpha)*ema_prev if i != (n - 1) else first_ema
        ema.append(ema_t)
        ema_prev = ema_t
    if pad_with_na:
        pad = np.zeros(n - 1 + extra_pad)
        pad[:] = np.nan
        ema = np.append(pad, ema)
    return np.array(ema)


def MOM(close, n, pad_with_na=True):
    mom = close[n:] - close[:-n]
    if pad_with_na:
        pad = np.zeros(n)
        pad[:] = np.nan
        mom = np.append(pad, mom)
    return mom


def MD(t_price, n, sma, pad_with_na=True):
    # t_price: np.array()
    md = []
    for i in range(len(t_price)):
        if i + n <= len(t_price):
            diff = np.absolute(t_price[i: i + n] - sma[i + n - 1])
            md.append(diff.mean())
    md = np.array(md)
    if pad_with_na:
        pad = np.zeros(n - 1)
        pad[:] = np.nan
        md = np.append(pad, md)
    return md


def CCI(t_price, sma_n, md_n):
    sma = SMA(t_price, n=sma_n)
    md = MD(t_price, md_n, sma)
    constant = 1/0.015
    cci = constant*((t_price - sma)/ md)
    return cci


def MACD(close, first_n, second_n):
    first_ema = EMA(close, n=first_n)
    second_ema = EMA(close, n=second_n)
    return first_ema - second_ema



def SMI(close, high, low, first_n, second_n, third_n):
    # close, high, low: np.array()
    # first_n: %K period
    # second_n: %K EMA smoothing period
    # third_n: %K double EMA smoothing period

    # Step 1: get HH and LL, first (n - 1) values are np.nan
    tmp_HH = np.zeros((len(high), first_n))
    tmp_HH[:] = -np.inf
    tmp_LL = np.zeros((len(low), first_n))
    tmp_LL[:] = np.inf

    for j in range(first_n):
        tmp_HH[j:, j] = high[:-j] if j != 0 else high
        tmp_LL[j:, j] = low[:-j] if j != 0 else low
    pad = np.zeros(first_n - 1)
    pad[:] = np.nan
    HH = tmp_HH.max(axis=1)[(first_n - 1):]
    HH = np.append(pad, HH)
    LL = tmp_LL.min(axis=1)[(first_n - 1):]
    LL = np.append(pad, LL)

    # Step 2: get nrm and den
    nrm = EMA(EMA(close - ((LL + HH)/2), n=second_n), n=third_n)
    den = EMA(EMA((HH - LL)/2, n=second_n), n=third_n)

    # Step 3: final step
    smi = 100 * nrm/den
    return smi


def ROC(close, n, pad_with_na=True):
    roc = 100*(close[n:] - close[:-n])/close[:-n]
    if pad_with_na:
        pad = np.zeros(n)
        pad[:] = np.nan
        roc = np.append(pad, roc)
    return roc


def William_R(close, high, low, n, pad_with_na=True):
    # work out HH and LL
    tmp_HH = np.zeros((len(high), n))
    tmp_LL = np.zeros((len(low), n))
    tmp_HH[:] = -np.inf
    tmp_LL[:] = np.inf
    for j in range(n):
        tmp_HH[j:, j] = high[:-j] if j != 0 else high
        tmp_LL[j:, j] = low[:-j] if j != 0 else low
    HH = tmp_HH.max(axis=1)
    LL = tmp_LL.min(axis=1)
    wr = 100 * (HH - close)/(HH - LL) 
    return -wr


def k_avg_label(close, w, k):
    # w: look back w days including current day
    # k: k future days

    # Step 1: get sigma
    tmp = np.zeros((len(close), w))
    tmp[:] = np.nan
    for i in range(w):
        tmp[i:, i] = close[:-i] if i != 0 else close
    w_mean = tmp.mean(axis=1)
    sigma = np.sqrt(((tmp.T - w_mean).T**2).sum(axis=1)/w)

    # Step 2: future k mean
    tmp = np.zeros((len(close), k))
    tmp[:] = np.nan
    for i in range(k):
        tmp[i:, i] = close[:-i] if i != 0 else close
    k_mean = tmp.mean(axis=1)
    k_pad = np.zeros(k)
    k_pad[:] = np.nan
    k_mean = np.append(k_mean[k:], k_pad)

    # Step 3: labeling
    diff = close - k_mean
    conditions = [
                (diff <= -sigma),
                (-sigma < diff) & (diff < 0),
                (0 <= diff) & (diff < sigma),
                (sigma <= diff),
                np.isnan(sigma) == True,
                np.isnan(diff) == True
                
    ]
    values = ['fall plus', 'fall', 'rise', 'rise plus', np.nan, np.nan]
    label = np.select(conditions, values)
    
    return diff, sigma, label
