'''
Author: David Radke
Date: April 26, 2021
'''

import re, datetime, time, csv
import matplotlib.pyplot as plt
import pandas as pd
import urllib.request as rq
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import numpy as np
import json
import os
import copy


def loadData(url):
    dataset = rq.urlopen(url)
    dataset = dataset.read()
    dataset = json.loads(dataset)

    comp_price = pd.DataFrame(dataset)
    comp_price = comp_price.sort_values('BLOCK_HOUR')
    return comp_price

def loadMarketAvg(path):
    df = pd.read_csv(path)
    lows = np.array(df["Low"])
    highs = np.array(df["High"])
    avg = np.flip(np.mean(np.stack((lows, highs), axis=0).T,axis=1))
    return avg

def meanCOMP(alcx_price):
    start, end, step = 0, 24, 24
    mean_alcx = []
    while end <= alcx_price.size:
        mean_alcx.append(np.mean(alcx_price[start:end]))
        start+=step
        end+=step
    return np.array(mean_alcx)

def plotHist(hist_diffs, window, shift):
    fig = plt.figure(figsize=(10, 6))

    plt.hist(hist_diffs, bins='auto')
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlabel("Mean: "+str(round(np.mean(hist_diffs),4))+"; STD: "+str(round(np.std(hist_diffs),4)), fontsize=16)
    plt.title("Cost Distribution: Window "+str(window)+", Shift "+str(shift), fontsize=16)
    plt.show()



def plotCompare(mean_alcx, market_avg, diffs, window,shift=0):
    fig = plt.figure(figsize=(10, 6))
    mean_diffs = np.mean(diffs)
    hist_diffs = copy.deepcopy(diffs) / window
    diffs_norm = (diffs - np.amin(diffs)) / (np.amax(diffs) - np.amin(diffs))
    for i, val in enumerate(diffs_norm):
        plt.axvline(i, c='b', alpha=val,lw=10)

    plt.plot(np.arange(mean_alcx.shape[0]), mean_alcx, c='r', linewidth=3, label='COMP')
    plt.plot(np.arange(market_avg.shape[0]), market_avg, c='g', linewidth=3, label='CCI 30')
    plt.xlabel("Days Since Inception", fontsize=16)
    plt.ylabel("Normalized Price", fontsize=16)
    plt.text(mean_alcx.shape[0]-50, 0.0, "Mean Diff: "+str(round(mean_diffs/window,4)), fontsize=15, bbox=dict(facecolor='w', edgecolor='k', pad=3.0))
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=15, loc='upper left')
    if shift == 0:
        plt.title("DTW ("+str(window)+" day window) with CCI30 and COMP Price", fontsize=18)
    else:
        plt.title("DTW ("+str(window)+" day window) with CCI30 and COMP Price ("+str(shift)+" day shift)", fontsize=18)
    plt.show()

    plotHist(hist_diffs, window, shift)

def doDTW(mean_alcx, market_avg, window=7,shift=0):
    end = 0
    diffs = []
    while (end+shift) <= market_avg.size:
        if (end-window) < 0:
            start = 0
        else:
            start = (end-window)
        focal = mean_alcx[start:end]
        target = market_avg[start+shift:end+shift]
        d, path = fastdtw(focal, target, dist=euclidean)
        diffs.append(d)
        end+=1
    return np.array(diffs)
