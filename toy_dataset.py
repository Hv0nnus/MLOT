#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# This code is part of the paper :
# IJCAI 2020 paper "Metric Learning in Optimal Transport for Domain Adaptation"
# Written by Tanguy Kerdoncuff
# If there is any bug, don't hesitate to send me a mail to my personal email:
# tanguy.kerdoncuff@laposte.net

import numpy as np
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import argparse


from pyotda import ot  # This is a local import that use code that is currently not available on POT
from sklearn.decomposition import PCA
import sklearn


def getAlgoToUse(algoName, XS, XT, YS, distribution_estimation):
    """
    Create the algorithm, and set its parameters to use
    """
    if algoName == "MLOT_id":
        source_in_target_subspace = XS
        target_in_target_subspace = XT
        ML_init = "full_identity"

        reg_e = 0.005
        reg_cl = 0
        reg_l = 0.1
        max_iter = 10
        max_inner_iter_grad = 1
        max_inner_iter_sink = 10
        margin = 1
        sinkhorn_type = "sinkhorn_class"
        verbose_mlot = False

        algo = ot.da.SinkhornMLTransport(reg_e=reg_e,
                                         reg_cl=reg_cl,
                                         reg_l=reg_l,
                                         norm="median",
                                         max_iter=max_iter,
                                         max_inner_iter_grad=max_inner_iter_grad,
                                         max_inner_iter_sink=max_inner_iter_sink,
                                         verbose=verbose_mlot,
                                         dimension=2,
                                         ML_init=ML_init,
                                         margin=margin)

        algo.fit(source_in_target_subspace, YS, target_in_target_subspace)
        XSa, XTa = algo.transform(Xs=source_in_target_subspace), target_in_target_subspace

    elif algoName == "MLOT":
        ML_init = "full_identity"

        reg_e = 0.005
        reg_cl = 0
        reg_l = 0.1
        max_iter = 3  # 3
        max_inner_iter_grad = 1
        max_inner_iter_sink = 10
        margin = 1
        sinkhorn_type = "sinkhorn_class"
        verbose_mlot = False

        pcaT = sklearn.decomposition.PCA(1).fit(XT)
        Vt = np.transpose(pcaT.components_)

        source_in_target_subspace = XS
        target_in_target_subspace = XT.dot(Vt.dot(np.transpose(Vt)))

        algo = ot.da.SinkhornMLTransport(reg_e=reg_e,
                                         reg_cl=reg_cl,
                                         reg_l=reg_l,
                                         norm="median",
                                         max_iter=max_iter,
                                         max_inner_iter_grad=max_inner_iter_grad,
                                         max_inner_iter_sink=max_inner_iter_sink,
                                         verbose=verbose_mlot,
                                         dimension=2,
                                         ML_init=ML_init,
                                         margin=margin)

        algo.fit(source_in_target_subspace, YS, target_in_target_subspace)
        XSa, XTa = algo.transform(Xs=source_in_target_subspace), target_in_target_subspace

    elif algoName == "OTDA":
        algo = ot.da.SinkhornLpl1Transport(reg_e=0.05, reg_cl=0.1, norm="median", max_iter=10,
                                           max_inner_iter=1000)
        algo.fit(XS, YS, XT, distribution_estimation=distribution_estimation)
        XSa, XTa = algo.transform(Xs=XS), XT

    else:
        XSa, XTa, algo = XS, XT, None
    return XSa, XTa, algo


def make_image_dataset():
    np.random.seed(456)
    x = 3
    y = 10
    c4 = 100
    c1 = 0.1  # 0.1
    nb = 10
    ms_p, ms_n = [-x, 0], [x, 0]
    mt_p, mt_n = [-x, y], [x, -y]
    # covs_p, covs_n = [[c1, 0], [0, c4]], [[c1, 0], [0, c4]]
    # covt_p, covt_n = [[c1, 0], [0, 1]], [[c1, 0], [0, 1]]
    covs_p, covs_n = [[c1, 0], [0, c4]], [[c1, 0], [0, c4]]
    covt_p, covt_n = [[c1 * 50, 0], [0, 50]], [[c1 * 50, 0], [0, 50]]
    Xs_p = np.random.multivariate_normal(ms_p, covs_p, nb)
    Xs_n = np.random.multivariate_normal(ms_n, covs_n, nb)
    Xs = np.concatenate((Xs_p, Xs_n), axis=0)
    ys = np.concatenate((np.zeros(nb), np.ones(nb)), axis=0)

    Xt_p = np.random.multivariate_normal(mt_p, covt_p, nb)
    Xt_n = np.random.multivariate_normal(mt_n, covt_n, nb)
    Xt = np.concatenate((Xt_p, Xt_n), axis=0)
    yt = np.concatenate((np.zeros(nb), np.ones(nb)), axis=0)
    return Xs, ys, Xt, yt


def drawPoints(X, Y, b, r, m, z, label):
    markerSize = 50
    plt.scatter(X[:, 0], X[:, 1], c=Y, label=label, edgecolor='black',
                linewidth=1, marker=m, s=[markerSize] * len(X), zorder=z,
                cmap=ListedColormap([b, r]))  #


def finalizePlot(ax, xMin, xMax, yMin, yMax):
    ax.set_xlim(xMin, xMax)
    ax.set_ylim(yMin, yMax)
    ax.legend(loc=0)


def plot_data(XS, XT, algo, YS, YT, l2, c11, c12, c21, c22, c13, c23):
    if algo is not None:
        G = algo.coupling_
        nbPerSample = 20
        cls = np.argsort(-G)[:, :nbPerSample]
        mx = G.max()
        for i in range(XS.shape[0]):
            color = c12
            if YS[i] == l2:
                color = c22
            for j in range(nbPerSample):
                alpha = G[i, cls[i, j]] / mx
                alpha /= 3
                plt.plot([XS[i, 0], XT[cls[i, j], 0]],
                         [XS[i, 1], XT[cls[i, j], 1]],
                         alpha=alpha, color=color, zorder=0)
    drawPoints(XS[YS == 0], YS[YS == 0], c11, c21, "o", 1, "Class 1")
    drawPoints(XS[YS == 1], YS[YS == 1], c21, c11, "s", 1, "Class 2")
    drawPoints(XT, YT, c13, c23, "+", 2, "Target")


def main(algoName):
    """
    :param algoName: Algo to use, MLOT or OTDA. Chose NA for initial point an legend.
    :return: Save and display an image of the toy dataset with the Optimal Transport for each algorithm.
    """

    c11 = "#0000FF"
    c12 = "#0044BB"
    c13 = "#444488"
    c21 = "#FF0000"
    c22 = "#BB4400"
    c23 = "#884444"

    matplotlib.rcParams['font.size'] = 18

    XS, YS, XT, YT = make_image_dataset()
    labels = np.unique(YS)
    l1 = labels[0]
    l2 = labels[1]

    distribution_estimation = None

    XSa, XTa, algo = getAlgoToUse(algoName, XS, XT, YS, distribution_estimation)

    fig, ax = plt.subplots()
    if algoName[:4] == "MLOT":
        if algoName == "MLOT_id":
            XS_L = XS @ algo.Ls
            XT_u = XT
        elif algoName == "MLOT":
            XS_L = XS @ algo.Ls
            XT_u = XTa
        plot_data(XS_L, XT_u, algo, YS, YT, l2, c11, c12, c21, c22, c13, c23)
    else:
        plot_data(XS, XT, algo, YS, YT, l2, c11, c12, c21, c22, c13, c23)

    # set the x-spine (see below for more info on `set_position`)
    ax.spines['left'].set_position('zero')

    # turn off the right spine/ticks
    ax.spines['right'].set_color('none')
    ax.yaxis.tick_left()

    # set the y-spine
    ax.spines['bottom'].set_position('zero')

    # turn off the top spine/ticks
    ax.spines['top'].set_color('none')
    ax.xaxis.tick_bottom()

    ax.set_xticks([5])
    ax.set_xticklabels([1], position=(0, 5.5))

    ax.set_yticks([6, 12, 18])
    ax.set_yticklabels(["", "", 3])

    if algoName == "NA":
        ax.legend(loc=0, prop={'size': 13})
        leg = ax.get_legend()
        leg.legendHandles[0].set_color(c11)
        leg.legendHandles[1].set_color(c21)
        leg.legendHandles[2].set_color('black')

    plt.savefig("./PDF/" + algoName + ".pdf", bbox_inches="tight")
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Toy dataset parameters')
    parser.add_argument('--algoName', type=str, default='NA', choices=["NA", "OTDA", "MLOT_id", "MLOT"])
    args = parser.parse_args()
    main(algoName=args.algoName)