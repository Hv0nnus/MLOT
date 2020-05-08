#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# This code is part of the paper :
# IJCAI 2020 paper "Metric Learning in Optimal Transport for Domain Adaptation"
# Written by Tanguy Kerdoncuff
# If there is any bug, don't hesitate to send me a mail to my personal email:
# tanguy.kerdoncuff@laposte.net
# This is inspired from a code of Leo Gautheron.

import os
import time
import random
import numpy as np
import sklearn
import argparse
import pickle
from scipy.io import loadmat
from sklearn import preprocessing
from sklearn import neighbors
from sklearn.metrics import euclidean_distances
from sklearn.decomposition import PCA

from pyotda import ot
# This is a local import that use code that is currently not available on POT


def getLabel(trainData, trainLabels, testData, type_classifier="1NN"):
    """
    :param trainData:
    :param trainLabels:
    :param testData:
    :param type_classifier: Only nNN and SVM_x implemented. With x a float and n an integer.
    :return: The prediction of the label of testData using the train data to learn a classifier
    """
    if "NN" in type_classifier:
        clf = sklearn.neighbors.KNeighborsClassifier(int(type_classifier[0:-2]))
        clf.fit(trainData, trainLabels)
        prediction = clf.predict(testData)
    elif "SVM" in type_classifier:
        C = float(type_classifier.split("_")[1])
        trainData, trainLabels = sklearn.utils.shuffle(trainData, trainLabels)
        clf = sklearn.linear_model.SGDClassifier(max_iter=2000, tol=10 ** (-4), alpha=C)
        clf.fit(trainData, trainLabels)
        prediction = clf.predict(testData)
    return prediction


def generateSubset2(X, Y, p):
    """
    This function should not be used on target true label because the proportion of classes are not available.
    :param X: Features
    :param Y: Labels
    :param p: Percentage of data kept.
    :return: Subset of X and Y with same proportion of classes.
    """
    idx = []
    for c in np.unique(Y):
        idxClass = np.argwhere(Y == c).ravel()
        random.shuffle(idxClass)
        idx.extend(idxClass[0:int(p * len(idxClass))])
    return X[idx], Y[idx]


def get_param_optimal(name, Sx, Sy, Tx, Ty, param, Sname, Tname, type_classifier="1NN"):
    """
    This is the function to call to compute the cross validation to find the best set of hyper parameters.
    This function respect the unsupervised setting and do not use Ty (expect for the method that deliberatly cheat to
    create a baseline (Tused)).
    :param name: Name of the method to cross validate
    :param Sx: Source features
    :param Sy: Source labels
    :param Tx: Target features
    :param Ty: Target labels
    :param param: list of parameters for different methods
    :param Sname: Source dataset name
    :param Tname: Target dataset name
    :param type_classifier:
    :return: Save a pickle with the name of method, the source and target dataset name.
    """
    param_train = dict(param)
    time_before_loop = time.time()
    print("\nTrain : ", name)

    # This is the number of iteration to each set of hyperparameter to avoid randomness
    nb_train = param["numberIteration"]

    # The next lines will define the range of cross validation.
    d_loop = [20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 250, 300, 400, 500]
    max_d = min(min(Sx.shape), min(Tx.shape))
    d_loop_aux = []
    for d in range(len(d_loop)):
        if d_loop[d] < max_d:
            d_loop_aux.append(d_loop[d])
    d_loop = d_loop_aux

    # 50 loop of MLOT seems enought in our case.
    max_iter_loop = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 18, 20, 25, 30, 40, 50]

    if "OTSAML" in name:
        max_iter_loop = [1000]
        reg_pca_loop = [0.0001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100]
        lr_loop = [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]
    else:
        reg_pca_loop = [-1]
        lr_loop = [-1]
    reg_e_loop = [0.05, 0.07, 0.09, 0.1, 0.3, 0.5, 0.7, 1, 1.2, 1.5, 1.7, 2, 3]
    reg_cl_loop = [0, 0.01, 0.03, 0.05, 0.07, 0.09, 0.1, 0.3, 0.5, 0.7, 1, 1.2, 1.5, 1.7, 2, 3]
    reg_l_loop = [0.001, 0.01, 0.1, 1, 10, 100]
    if "JDOT" in name:  # Following the paper
        reg_l_loop = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1]
        reg_e_loop = [1, 5, 10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000]
    margin_loop = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 100]
    number_iteration_cross_val = 0
    while time.time() - time_before_loop < 3600 * args.time_cross_val and number_iteration_cross_val < 1000:
        np.random.seed(4896 * number_iteration_cross_val + 5272)
        param_train["d"] = d_loop[np.random.randint(len(d_loop))]
        param_train["max_iter"] = max_iter_loop[np.random.randint(len(max_iter_loop))]
        param_train["reg_e"] = reg_e_loop[np.random.randint(len(reg_e_loop))]
        param_train["reg_cl"] = reg_cl_loop[np.random.randint(len(reg_cl_loop))]
        param_train["reg_l"] = reg_l_loop[np.random.randint(len(reg_l_loop))]
        param_train["reg_pca"] = reg_pca_loop[np.random.randint(len(reg_pca_loop))]
        param_train["lr"] = lr_loop[np.random.randint(len(lr_loop))]
        param_train["margin"] = margin_loop[np.random.randint(len(margin_loop))]
        result = []
        target_result = []
        # Some method can sometime crash, this will be printed.
        try:
            for i in range(nb_train):
                # Firt adaptation
                if name == "JDOTSVM" or name == "JDOTSVMe":
                    Tay_pred = adaptData(algo=name,
                                         Sx=Sx, Sy=Sy,
                                         Tx=Tx, Ty=Ty,
                                         param=param_train)
                else:
                    Sa, Ta, Say, Tay = adaptData(algo=name,
                                                 Sx=Sx, Sy=Sy,
                                                 Tx=Tx, Ty=Ty,
                                                 param=param_train)
                    Tay_pred = getLabel(Sa, Say, Ta, type_classifier=type_classifier)

                # If we use the adaptation from target to source with SA
                if args.SA:
                    param_train_sa = dict(param_train)
                    param_train_sa["d"] = 70
                    Taa, Saa, Taay, Saay = adaptData(algo="SA",
                                                     Sx=Tx, Sy=Tay_pred,
                                                     Tx=Sx, Ty=Sy,
                                                     param=param_train_sa)
                # If we use the adaptation from target to source with the same method.
                else:
                    Taa, Saa, Taay, Saay = adaptData(algo=name,
                                                     Sx=Tx, Sy=Tay_pred,
                                                     Tx=Sx, Ty=Sy,
                                                     param=param_train)
                # To check the robustness of the set of hyper parameter the prediction is tested 10 times with a subset
                # of the predicted data.
                for j in range(10):
                    Taa_sub, Taay_sub = generateSubset2(Taa, Taay, p=0.5)
                    Sy_pred = getLabel(Taa_sub, Taay_sub, Saa, type_classifier=type_classifier)
                    # Real cross validation result.
                    result.append(100 * float(sum(Sy_pred == Saay)) / len(Sy_pred))

                # This line cheat during the cross validation but is saved as a baseline.
                if name == "JDOTSVM" or "JDOTSVMe":
                    target_result.append(100 * float(sum(Tay_pred == Ty)) / len(Tay_pred))
                else:
                    target_result.append(100 * float(sum(Tay_pred == Tay)) / len(Tay_pred))

            temps_dict = dict(param_train)
            temps_dict["target_result"] = target_result
            temps_dict["result"] = result
            print(number_iteration_cross_val, ": result", np.mean(result),
                  "target result", np.mean(target_result),
                  "d", param_train["d"],
                  "max_iter", param_train["max_iter"],
                  "reg_e", param_train["reg_e"],
                  "reg_l", param_train["reg_l"],
                  "reg_cl", param_train["reg_cl"],
                  "reg_pca", param_train["reg_pca"],
                  "lr", param_train["lr"],
                  "margin", param_train["margin"])
            # Open and close the pickle every time to avoid potential bug.
            pickle_out = open("pickle/" + args.pickle_name + "/" + args.pickle_name +
                              name + Sname + Tname + ".pickle", "ab")
            pickle.dump(temps_dict, pickle_out)
            pickle_out.close()
        except:
            print("Error with this setting :",
                  "d", param_train["d"],
                  "max_iter", param_train["max_iter"],
                  "reg_e", param_train["reg_e"],
                  "reg_l", param_train["reg_l"],
                  "reg_cl", param_train["reg_cl"],
                  "reg_pca", param_train["reg_pca"],
                  "lr", param_train["lr"],
                  "margin", param_train["margin"])
        time.sleep(1.)  # Allow us to stop the program with ctrl-C
        number_iteration_cross_val += 1
        # Special case were there is no hyperparameters to tune.
        if name in ["NA", "CORAL", "Tused"] and number_iteration_cross_val > 1:
            print("No param to tune, the pickle has been saved")
            break
    print("Time for the cross validation:", time.time() - time_before_loop, "s")
    return param_train


def adaptData(algo, Sx, Sy, Tx, Ty, param=None):
    """
    Main function of the code that launch a method.
    :param algo: Name of the method to use.
    :param Sx: Source features.
    :param Sy: Source labels.
    :param Tx: Target features.
    :param Ty: Target labels.
    :param param: List of parameters needed for each method.
    :return: The adapted data source and target. It also return the labels unchanged.
    """
    if algo == "Tused":
        # Cheating method that use the target dataset to learn the classifier.
        # This can be usefull for a baseline that we probably can't beat in domain adaptation.
        Sy = Ty
        sourceAdapted = Tx
        targetAdapted = Tx
    if algo == "NA":
        # No Adaptation
        sourceAdapted = Sx
        targetAdapted = Tx

    elif algo == "SA":
        # Subspace Alignment, described in:
        # Unsupervised Visual Domain Adaptation Using Subspace Alignment, 2013,
        # Fernando et al.

        pcaS = sklearn.decomposition.PCA(n_components=param["d"], svd_solver=param["svd_solver"]).fit(Sx)
        pcaT = sklearn.decomposition.PCA(n_components=param["d"], svd_solver=param["svd_solver"]).fit(Tx)

        XS = np.transpose(pcaS.components_)
        XT = np.transpose(pcaT.components_)
        Xa = XS.dot(np.transpose(XS)).dot(XT)

        sourceAdapted = Sx.dot(Xa)
        targetAdapted = Tx.dot(XT)

    elif algo == "TCA":
        # Domain adaptation via transfer component analysis. IEEE TNN 2011
        d = param["d"]  # subspace dimension
        Ns = Sx.shape[0]
        Nt = Tx.shape[0]
        L_ss = (1. / (Ns * Ns)) * np.full((Ns, Ns), 1)
        L_st = (-1. / (Ns * Nt)) * np.full((Ns, Nt), 1)
        L_ts = (-1. / (Nt * Ns)) * np.full((Nt, Ns), 1)
        L_tt = (1. / (Nt * Nt)) * np.full((Nt, Nt), 1)
        L_up = np.hstack((L_ss, L_st))
        L_down = np.hstack((L_ts, L_tt))
        L = np.vstack((L_up, L_down))
        X = np.vstack((Sx, Tx))
        K = np.dot(X, X.T)  # linear kernel
        H = (np.identity(Ns + Nt) - 1. / (Ns + Nt) * np.ones((Ns + Nt, 1)) *
             np.ones((Ns + Nt, 1)).T)
        inv = np.linalg.pinv(np.identity(Ns + Nt) + K.dot(L).dot(K))
        D, W = np.linalg.eigh(inv.dot(K).dot(H).dot(K))
        W = W[:, np.argsort(-D)[:d]]  # eigenvectors of d highest eigenvalues
        sourceAdapted = np.dot(K[:Ns, :], W)  # project source
        targetAdapted = np.dot(K[Ns:, :], W)  # project target

    elif algo == "CORAL":
        # Return of Frustratingly Easy Domain Adaptation. AAAI 2016
        from scipy.linalg import sqrtm
        Cs = np.cov(Sx, rowvar=False) + np.eye(Sx.shape[1])
        Ct = np.cov(Tx, rowvar=False) + np.eye(Tx.shape[1])
        Ds = Sx.dot(np.linalg.inv(np.real(sqrtm(Cs))))  # whitening source
        Ds = Ds.dot(np.real(sqrtm(Ct)))  # re-coloring with target covariance
        sourceAdapted = Ds
        targetAdapted = Tx

    elif algo == "OT":
        # Optimal Transport with class regularization described in:
        # Domain adaptation with regularized optimal transport, 2014.
        # Courty et al.
        transp3 = ot.da.SinkhornLpl1Transport(reg_e=param["reg_e"], reg_cl=0, norm="median",
                                              max_iter=1, max_inner_iter=100, log=False,
                                              tol=10 ** -7)
        transp3.fit(Xs=Sx, ys=Sy, Xt=Tx)

        sourceAdapted = transp3.transform(Xs=Sx)
        targetAdapted = Tx

    elif algo == "OTDA":
        # Optimal Transport with class regularization described in:
        # Domain adaptation with regularized optimal transport, 2014.
        # Courty et al.
        transp3 = ot.da.SinkhornLpl1Transport(reg_e=param["reg_e"], reg_cl=param["reg_cl"], norm="median")
        transp3.fit(Xs=Sx, ys=Sy, Xt=Tx)

        sourceAdapted = transp3.transform(Xs=Sx)
        targetAdapted = Tx

    elif algo == "OTDA_pca":
        # Variant of the SA method + Optimal Transport.
        # A PCA is apply to each method separatly. The transformation is apply twice (XS @ Xs.T) so that the data are
        # still in the same dimension space. This can be seen has a projection.
        pcaS = sklearn.decomposition.PCA(param["d"], svd_solver=param["svd_solver"]).fit(Sx)
        pcaT = sklearn.decomposition.PCA(param["d"], svd_solver=param["svd_solver"]).fit(Tx)

        XS = np.transpose(pcaS.components_)
        XT = np.transpose(pcaT.components_)

        source_in_target_subspace = Sx.dot(XS.dot(np.transpose(XS)))
        target_in_target_subspace = Tx.dot(XT.dot(np.transpose(XT)))
        # print("source_in_target_subspace", source_in_target_subspace)
        # print("param[d]", param["d"])
        # print("XS.dot(np.transpose(XS))", XS.dot(np.transpose(XS)))

        transp = ot.da.SinkhornLpl1Transport(reg_e=param["reg_e"], reg_cl=param["reg_cl"], norm="median")
        transp.fit(Xs=source_in_target_subspace, ys=Sy, Xt=target_in_target_subspace)

        sourceAdapted = transp.transform(source_in_target_subspace)
        targetAdapted = target_in_target_subspace

    elif algo[:4] == "MLOT":
        ML_init_temps = param["ML_init"]
        # pcaS = sklearn.decomposition.PCA(min(param["d"], Sx.shape[0], Sx.shape[1]),
        # svd_solver=param["svd_solver"]).fit(Sx)
        pcaT = sklearn.decomposition.PCA(min(param["d"], Tx.shape[0], Tx.shape[1]), svd_solver=param["svd_solver"]).fit(
            Tx)
        # XS = np.transpose(pcaS.components_)
        XT = np.transpose(pcaT.components_)

        if algo == "MLOT_id":
            # The pca is not applied
            source_in_target_subspace = Sx
            target_in_target_subspace = Tx
        elif algo == "MLOT":
            # The pca is apply only on the target dataset at this point, this can be seen as a preprocess.
            # The source PCA is apply during the SinkhornMLTranport fit.
            source_in_target_subspace = Sx
            target_in_target_subspace = Tx.dot(XT.dot(np.transpose(XT)))
            param["ML_init"] = "SS"
        transp3 = ot.da.SinkhornMLTransport(reg_e=param["reg_e"],
                                            reg_cl=param["reg_cl"],
                                            reg_l=param["reg_l"],
                                            norm="median",
                                            max_iter=param["max_iter"],
                                            max_inner_iter_grad=param["max_inner_iter_grad"],
                                            max_inner_iter_sink=param["max_inner_iter_sink"],
                                            svd_solver=param["svd_solver"],
                                            verbose=param["verbose"],
                                            dimension=param["d"],
                                            ML_init=param["ML_init"],
                                            margin=param["margin"],
                                            mini_batch_size=5000)
        param["ML_init"] = ML_init_temps
        transp3.fit(Xs=source_in_target_subspace,
                    ys=Sy,
                    Xt=target_in_target_subspace,
                    yt=Ty)
        if param["new_space"]:
            sourceAdapted = transp3.transform(Xs=source_in_target_subspace)
            targetAdapted = target_in_target_subspace
        else:
            transp3.xt_ = Tx
            sourceAdapted = transp3.transform(Xs=source_in_target_subspace)
            targetAdapted = Tx

    elif algo == "LMNN":
        # Large Margin Nearest Neighbor
        from pyotda.ot import lmnn_original
        LMNN = lmnn_original.LargeMarginNearestNeighbor(k=3, mu=0.5,
                                                        margin=param["margin"],
                                                        nFtsOut=param["d"],
                                                        maxCst=int(1e7),
                                                        randomState=None,
                                                        maxiter=param["max_iter"])
        LMNN.fit(X=Sx, y=Sy)

        sourceAdapted = Sx @ (LMNN.L_).T @ (LMNN.L_)
        targetAdapted = Tx

    elif algo == "JDOTSVMe":
        # This is the version of JDOT used. Sinkhorn + linear SVM classifier.
        from JDOT import jdot
        from sklearn import preprocessing
        lb = preprocessing.LabelBinarizer()
        lb.fit(Sy)
        Sy_01 = lb.transform(Sy)
        # WARNING : we use SVM method as NN method is not immediately implemented
        clf_jdot, dic = jdot.jdot_svm(X=Sx, y=Sy_01, Xtest=Tx, ytest=[],
                                      gamma_g=1,
                                      numIterBCD=param["max_iter"],  # To stay fair, this will also be cross validate
                                      alpha=param["reg_l"],  # from 10-5 to 1.
                                      lambd=1e1,  # Used for the classifier
                                      method='sinkhorn',
                                      reg_sink=param["reg_e"],
                                      ktype='linear')
        return dic["ypred"]

    elif algo == "JDOTe":
        # WARNING : here we use the transport plan learned to adapt the source
        # and we do not use the prediction.
        from JDOT import jdot
        from sklearn import preprocessing
        lb = preprocessing.LabelBinarizer()
        lb.fit(Sy)
        Sy_01 = lb.transform(Sy)
        # WARNING : we use SVM method as NN method is not immediately implemented
        clf_jdot, dic = jdot.jdot_svm(X=Sx, y=Sy_01, Xtest=Tx, ytest=[],
                                      gamma_g=1,
                                      numIterBCD=param["max_iter"],  # To stay fair, this will also be cross validate
                                      alpha=param["reg_l"],  # from 10-5 to 1.
                                      lambd=1e1,  # Used for the classifier
                                      method='sinkhorn',
                                      reg_sink=param["reg_e"],
                                      ktype='linear')
        transp = dic["G"] / np.sum(dic["G"], 1)[:, None]  # Barycentric mapping
        sourceAdapted = transp @ Tx  # WARNING : here we use the transport plan learned to adapt the source
        # and we do not use the prediction.
        targetAdapted = Tx

    elif algo == "JDOTSVM":
        from JDOT import jdot
        from sklearn import preprocessing
        lb = preprocessing.LabelBinarizer()
        lb.fit(Sy)
        Sy_01 = lb.transform(Sy)
        # WARNING : we use SVM method as NN method is not immediately implemented
        clf_jdot, dic = jdot.jdot_svm(X=Sx, y=Sy_01, Xtest=Tx, ytest=[],
                                      gamma_g=1,
                                      numIterBCD=param["max_iter"],  # To stay fair, this will also be cross validate
                                      alpha=param["reg_l"],  # from 10-5 to 1.
                                      lambd=1e1,  # Used for the classifier
                                      method='emd',
                                      reg_sink=1,
                                      ktype='linear')
        return dic["ypred"]

    elif algo == "JDOT":
        # WARNING : here we use the transport plan learned to adapt the source
        # and we do not use the prediction.
        from JDOT import jdot
        from sklearn import preprocessing
        lb = preprocessing.LabelBinarizer()
        lb.fit(Sy)
        Sy_01 = lb.transform(Sy)
        # WARNING : we use SVM method as NN method is not immediately implemented
        clf_jdot, dic = jdot.jdot_svm(X=Sx, y=Sy_01, Xtest=Tx, ytest=[],
                                      gamma_g=1,
                                      numIterBCD=param["max_iter"],  # To stay fair, this will also be cross validate
                                      alpha=param["reg_l"],  # from 10-5 to 1.
                                      lambd=1e1,  # Used for the classifier
                                      method='emd',
                                      reg_sink=1,
                                      ktype='linear')
        transp = dic["G"] / np.sum(dic["G"], 1)[:, None]  # Barycentric mapping
        sourceAdapted = transp @ Tx  # WARNING : here we use the transport plan learned to adapt the source
        # and we do not use the prediction.
        targetAdapted = Tx

    # ------------------------ Method not implemented in the paper -------------------------------------------
    #     elif algo == "LMNNOTDA":
    #         Sx_lmnn = Sx
    #
    #         transp3 = ot.da.SinkhornLpl1Transport(reg_e=param["reg_e"], reg_cl=param["reg_cl"], norm="median")
    #         transp3.fit(Xs=Sx_lmnn, ys=Sy, Xt=Tx)
    #
    #         sourceAdapted = transp3.transform(Xs=Sx)
    #         targetAdapted = Tx
    elif "OTSAML" in algo:  # OTSAMLidl OTSAMLl OTSAMLidnl OTSAMLnl
        # This version of MLOT with pytorch is still in progress.
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from sklearn.utils.validation import check_random_state
        from sklearn.metrics.pairwise import euclidean_distances
        from sklearn.utils import gen_batches
        from scipy import sparse

        # The cuda version might not be always implemented has this method is in progress.
        cuda = param["cuda"]
        rule = param["rule"]
        detach = param["detach"]

        def cost_matrix(features_source, features_target):
            """
            :param features_source:
            :param features_target:
            :return: Returns the matrix of $|x_i-y_j|^2$.
            """
            if int(torch.__version__[0]) < 1:  # 0.4
                XX = torch.einsum('ij,ij->i', (features_source, features_source))[:, None]
                YY = torch.einsum('ij,ij->i', (features_target, features_target))[None, :]
            else:
                XX = torch.einsum('ij,ij->i', features_source, features_source)[:, None]
                YY = torch.einsum('ij,ij->i', features_target, features_target)[None, :]
            distances = torch.matmul(features_source, features_target.t())
            distances *= -2
            distances += XX
            distances += YY
            all_C = distances
            return all_C

        def normalised(C, rule="median", detach=False):
            """
            :param C: The matrix of cost
            :param rule: The normalization can be done in various way. mean median and max are implemented.
            :param detach: If True, the gradient will be propagate trough the normalizer.
            :return: The matrix cost normalise to stabilize the sinkhorn algorithm. This modification change the
            Wasserstein disance by a factor norm_C but doesn't not change the optimal mapping which is the only thing
            important Domain Adaptation.
            """
            if rule == "mean":
                norm_C = torch.mean(C)
            elif rule == "median":
                norm_C = torch.median(C)
            elif rule == "max":
                norm_C = torch.max(C)
            else:
                assert False
            if detach:
                norm_C = norm_C.detach()
            return C / (norm_C + 10 ** -7)

        class Transport(nn.Module):
            def __init__(self, d, Sx, Tx, y):
                """
                :param d: The number of dimension of the PCA
                :param Sx: Source features
                :param Tx: Target features
                :param y: Source labels
                """
                super(Transport, self).__init__()
                self.Sx = Sx
                self.Tx = Tx
                self.Sx_tensor = torch.tensor(Sx).float()
                self.Tx_tensor = torch.tensor(Tx).float()
                self.nb_features = Sx.shape[1]
                if cuda:
                    self.Sx_tensor = self.Sx_tensor.cuda()
                    self.Tx_tensor = self.Tx_tensor.cuda()
                if "OTSAMLid" in algo:
                    self.SS_t_init_tensor = torch.eye(self.nb_features, self.nb_features, requires_grad=False).float()
                    self.TT_t_init_tensor = torch.eye(self.nb_features, self.nb_features, requires_grad=False).float()
                elif "OTSAML" in algo:
                    pcaS = sklearn.decomposition.PCA(d, svd_solver=param["svd_solver"]).fit(Sx)
                    pcaT = sklearn.decomposition.PCA(d, svd_solver=param["svd_solver"]).fit(Tx)

                    S = np.transpose(pcaS.components_)
                    T = np.transpose(pcaT.components_)
                    SS_t_init = S.dot(np.transpose(S))
                    TT_t_init = T.dot(np.transpose(T))

                    self.SS_t_init_tensor = torch.tensor(SS_t_init, requires_grad=False).float()
                    self.TT_t_init_tensor = torch.tensor(TT_t_init, requires_grad=False).float()

                if cuda:
                    self.SS_t_init_tensor = self.SS_t_init_tensor.cuda()
                    self.TT_t_init_tensor = self.TT_t_init_tensor.cuda()

                # lmnn
                self.Xt = Tx
                self.Xs = Sx
                self.y = y
                self.k_ = 3
                self.mu = 0.5
                self.nFtsOut_ = np.shape(self.Xs)[1]
                self.maxCst = int(1e7)
                self.margin = param["margin"]
                self.randomState = None
                self.maxiter = param["max_iter"]
                self.loss_func = None
                self.mini_batch_size = 10
                self.time = time.time()
                self.verbose = param["verbose"]

                # Store the appearing classes and the class index for each sample
                self.labels_, self.y_ = np.unique(y, return_inverse=True)
                self.classes_ = np.arange(len(self.labels_))
                # print(self.classes_)
                # Check that the number of neighbors is achievable for all classes
                min_class_size = np.bincount(self.y_).min()
                max_neighbors = min_class_size - 1
                self.k_ = min(self.k_, max_neighbors)
                self.targets_ = torch.tensor(np.empty((self.Xs.shape[0], self.k_), dtype=int))
                # print(self.targets_.shape, self.targets_)

                # layer NN
                # getattr()
                # self.fc2_s.weight.data.copy_(self.SS_t_init_tensor.t())
                if "nl" in algo:
                    self.fc1_s = nn.Linear(self.nb_features, self.nb_features, bias=True)
                    # self.fc1_s.weight.data.copy_(torch.zeros(self.nb_features, self.nb_features))
                    # self.fc1_s.bias.data.copy_(torch.zeros(self.nb_features))
                    self.fc2_s = nn.Linear(self.nb_features, self.nb_features, bias=True)
                    # self.fc2_s.weight.data.copy_(torch.zeros(self.nb_features, self.nb_features))
                    # self.fc2_s.bias.data.copy_(torch.zeros(self.nb_features))
                    self.fc3_s = nn.Linear(self.nb_features, self.nb_features, bias=True)
                    # self.fc3_s.weight.data.copy_(torch.zeros(self.nb_features, self.nb_features))
                    # self.fc3_s.bias.data.copy_(torch.zeros(self.nb_features))
                self.fc4_s = nn.Linear(self.nb_features, self.nb_features, bias=True)
                self.fc4_s.weight.data.copy_(torch.zeros(self.nb_features, self.nb_features))
                self.fc4_s.bias.data.copy_(torch.zeros(self.nb_features))

                if "nl" in algo:
                    self.fc1_t = nn.Linear(self.nb_features, self.nb_features, bias=True)
                    # self.fc1_t.weight.data.copy_(torch.zeros(self.nb_features, self.nb_features))
                    # self.fc1_t.bias.data.copy_(torch.zeros(self.nb_features))
                    self.fc2_t = nn.Linear(self.nb_features, self.nb_features, bias=True)
                    # self.fc2_t.weight.data.copy_(torch.zeros(self.nb_features, self.nb_features))
                    # self.fc2_t.bias.data.copy_(torch.zeros(self.nb_features))
                    self.fc3_t = nn.Linear(self.nb_features, self.nb_features, bias=True)
                    # self.fc3_t.weight.data.copy_(torch.zeros(self.nb_features, self.nb_features))
                    # self.fc3_t.bias.data.copy_(torch.zeros(self.nb_features))
                self.fc4_t = nn.Linear(self.nb_features, self.nb_features, bias=True)
                self.fc4_t.weight.data.copy_(torch.zeros(self.nb_features, self.nb_features))
                self.fc4_t.bias.data.copy_(torch.zeros(self.nb_features))

            def f_s(self, X_s_start):
                if "nl" in algo:
                    X_s = self.fc1_s(X_s_start)
                    X_s = (torch.sigmoid(X_s) - 0.5) * 2
                    X_s = self.fc2_s(X_s)
                    X_s = (torch.sigmoid(X_s) - 0.5) * 2
                    X_s = self.fc3_s(X_s)
                    X_s = (torch.sigmoid(X_s) - 0.5) * 2
                    X_s = self.fc4_s(X_s)
                elif "l" in algo:
                    X_s = self.fc4_s(X_s_start)
                else:
                    raise Exception('Wrong name of method')
                init = X_s_start @ self.SS_t_init_tensor
                return X_s + init, init
                # return init, init

            def f_t(self, X_t_start):
                if "nl" in algo:
                    X_t = self.fc1_t(X_t_start)
                    X_t = (torch.sigmoid(X_t) - 0.5) * 2
                    X_t = self.fc2_t(X_t)
                    X_t = (torch.sigmoid(X_t) - 0.5) * 2
                    X_t = self.fc3_t(X_t)
                    X_t = (torch.sigmoid(X_t) - 0.5) * 2
                    X_t = self.fc4_t(X_t)
                elif "l" in algo:
                    X_t = self.fc4_t(X_t_start)
                else:
                    raise Exception('Wrong name of method')
                init = X_t_start @ self.TT_t_init_tensor
                # return init, init
                return X_t + init, init

            def compute_neighbour(self, LXs_numpy):
                for class_ in self.classes_:
                    # print(class_)
                    class_ind, = np.where(np.equal(self.y_, class_))
                    dist = euclidean_distances(LXs_numpy[class_ind], squared=True)
                    np.fill_diagonal(dist, np.inf)
                    nghIdx = np.argpartition(dist, self.k_ - 1, axis=1)
                    nghIdx = nghIdx[:, :self.k_]
                    # argpartition doesn't guarantee sorted order, so we sort again but
                    # only the k neighbors
                    rowIdx = np.arange(len(class_ind))[:, None]
                    nghIdx = nghIdx[rowIdx, np.argsort(dist[rowIdx, nghIdx])]
                    self.targets_[class_ind] = torch.tensor(class_ind[nghIdx])
                    # print(self.targets_)
                # print("after the loop")
                # self.targets_ = torch.tensor(self.targets_)
                # print(self.targets)
                n, k = self.targets_.shape
                self.rows = torch.tensor(np.repeat(np.arange(n), k))
                self.cols = self.targets_.view(-1)
                # print(self.rows, self.cols)
                # targets_sparse = sparse.csr_matrix((np.ones(n * k),
                #                                     (self.rows, self.cols)), shape=(n, n))
                # self.laplacian = sum_outer_products(targets_sparse)
                #
                # if cuda:
                #     self.laplacian = self.laplacian.cuda()

            def _find_impostors(self, LXs_np, LXs_torch, margin_radii):
                n = LXs_np.shape[0]
                impostors = sparse.csr_matrix((n, n), dtype=np.int8)
                for class_ in self.classes_[:-1]:
                    imp1, imp2 = [], []
                    ind_in, = np.where(np.equal(self.y_, class_))
                    ind_out, = np.where(np.greater(self.y_, class_))
                    # Subdivide idx_out x idx_in to chunks of a size that is
                    # fitting in memory
                    ii, jj = self._find_impostors_batch(LXs_np[ind_out], LXs_np[ind_in],
                                                        margin_radii[ind_out],
                                                        margin_radii[ind_in])
                    if len(ii):
                        imp1.extend(ind_out[ii])
                        imp2.extend(ind_in[jj])
                        new_imps = sparse.csr_matrix(([1] * len(imp1), (imp1, imp2)),
                                                     shape=(n, n), dtype=np.int8)
                        impostors = impostors + new_imps
                imp1, imp2 = impostors.nonzero()
                if impostors.nnz > self.maxCst:  # subsample constraints if too many
                    randomState = check_random_state(self.randomState)
                    ind_subsample = randomState.choice(impostors.nnz,
                                                       self.maxCst, replace=False)
                    imp1, imp2 = torch.tensor(imp1[ind_subsample]), torch.tensor(imp2[ind_subsample])
                    if cuda:
                        imp1 = imp1.cuda()
                        imp2 = imp2.cuda()
                dist = torch.zeros(len(imp1))
                if cuda:
                    dist = dist.cuda()
                for chunk in gen_batches(len(imp1), 500):
                    dist[chunk] = torch.sum((LXs_torch[imp1[chunk]] - LXs_torch[imp2[chunk]]) ** 2, dim=1)
                if cuda:
                    return torch.tensor(imp1, dtype=torch.long).cuda(), torch.tensor(imp2,
                                                                                     dtype=torch.long).cuda(), dist
                else:
                    return torch.tensor(imp1, dtype=torch.long), torch.tensor(imp2, dtype=torch.long), dist

            def _find_impostors_batch(self, x1, x2, t1, t2, batch_size=500):
                n = len(t1)
                imp1, imp2 = [], []
                for chunk in gen_batches(n, batch_size):
                    dist_out_in = euclidean_distances(x1[chunk], x2, squared=True)
                    i1, j1 = np.where(dist_out_in < t1[chunk, None])
                    i2, j2 = np.where(dist_out_in < t2[None, :])
                    if len(i1):
                        imp1.extend(i1 + chunk.start)
                        imp2.extend(j1)
                    if len(i2):
                        imp1.extend(i2 + chunk.start)
                        imp2.extend(j2)
                return imp1, imp2

            def loss_grad(self, LXs_torch, LXs_np):
                n = self.Sx_tensor.shape[0]
                # Compute distances to target neighbors under L (plus margin 1)
                dist_tn = torch.zeros((n, self.k_))
                if cuda:
                    dist_tn = dist_tn.cuda()
                for k in range(self.k_):
                    # print("FIRST")
                    # print(self.targets_[:, k])
                    # print(LXs_torch)
                    dist_tn[:, k] = torch.sum((LXs_torch - LXs_torch[self.targets_[:, k]]) ** 2,
                                              dim=1) + self.margin
                margin_radii = dist_tn[:, -1]
                # Compute distances to impostors under L
                imp1, imp2, dist_imp = self._find_impostors(LXs_np, LXs_torch, margin_radii.cpu().detach().numpy())

                loss = 0
                for k in reversed(range(self.k_)):
                    loss1 = torch.max(dist_tn[imp1, k] - dist_imp)
                    loss2 = torch.max(dist_tn[imp2, k] - dist_imp)
                    loss = loss + torch.sum(loss1 ** 2) + torch.sum(loss2 ** 2)
                # loss = (1 - self.mu) * (self.gradStatic * (Ls.matmul(Ls.t()))).sum() + self.mu * loss
                outer_prod = ((LXs_torch[self.rows] - LXs_torch[self.cols]) ** 2).sum()
                # print(outer_prod)
                loss = (1 - self.mu) * outer_prod + self.mu * loss
                return loss

            def forward(self, gamma):
                s_transform, init_s = self.f_s(self.Sx_tensor)
                t_transform, init_t = self.f_t(self.Tx_tensor)
                self.source_in_source_subspace = s_transform.detach().cpu().numpy()
                self.target_in_target_subspace = t_transform.detach().cpu().numpy()
                C = cost_matrix(features_source=s_transform,
                                features_target=t_transform)
                Cs = torch.sum((init_s - s_transform) ** 2, dim=1)
                Ct = torch.sum((init_t - t_transform) ** 2, dim=1)
                C = normalised(C, rule=rule, detach=detach)
                # print(Cs)
                regularisation_s = normalised(Cs, rule=rule, detach=detach)
                regularisation_t = normalised(Ct, rule=rule, detach=detach)
                # print(regularisation_s)
                # print(s_transform.shape)
                # print(self.laplacian.shape)
                # print("1.8", time.time() - time_init)
                if True:
                    self.compute_neighbour(self.source_in_source_subspace)
                else:
                    self.compute_neighbour(self.Sx)
                # print("1.9", time.time() - time_init)
                lmnn = self.loss_grad(s_transform, self.source_in_source_subspace) / (self.k_ * self.Xs.shape[0])
                if param["verbose"]:
                    print("Dist", torch.sum(gamma * C).item())
                    print("Regularisation_s", torch.mean(regularisation_s))
                    print("Regularisation_t", torch.mean(regularisation_t))
                    print("lmnn", lmnn)
                    print("Sum", torch.sum(gamma * C) +
                          (torch.mean(regularisation_s) + torch.mean(regularisation_t)) * param["reg_pca"] +
                          lmnn * float(param["reg_l"]))
                # return lmnn
                # print("1.95", time.time() - time_init)
                return torch.sum(gamma * C) + \
                       (torch.mean(regularisation_s) + torch.mean(regularisation_t)) * param["reg_pca"] + \
                       lmnn * float(param["reg_l"])

        transport = Transport(d=param["d"],
                              Sx=Sx,
                              Tx=Tx,
                              y=Sy)

        if cuda:
            transport.cuda()
        transport.train(True)
        # optimizer = optim.SGD(transport.parameters(), lr=param["lr"], momentum=0)
        optimizer = optim.Adam(transport.parameters(), lr=param["lr"])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99)
        sink = ot.da.SinkhornLpl1Transport(reg_e=param["reg_e"], reg_cl=param["reg_cl"], norm="median")
        time_init = time.time()
        epoch = 0
        continue_while = True
        old_loss = torch.tensor([10 ** 10]).float()
        while continue_while:
            # print("1")
            # print("0.0", time.time() - time_init)
            if param["verbose"]:
                print("epoch", epoch)
                # print("L_s", transport.SS_t.weight)
            source_in_source_subspace = transport.f_s(transport.Sx_tensor)[0].detach().cpu().numpy()
            target_in_target_subspace = transport.f_t(transport.Tx_tensor)[0].detach().cpu().numpy()
            optimizer.zero_grad()
            # print("1.0", time.time() - time_init)
            sink.fit(Xs=source_in_source_subspace,
                     ys=Sy,
                     Xt=target_in_target_subspace)
            # print("1.5", time.time() - time_init)
            gamma = sink.coupling_
            # print(getAccuracy(sink.transform(Xs=source_in_source_subspace), Sy,
            #                   target_in_target_subspace, Ty))
            for _ in range(param["max_inner_iter_grad"]):
                optimizer.zero_grad()
                if cuda:
                    loss = transport(torch.tensor(gamma).float().cuda())
                else:
                    loss = transport(torch.tensor(gamma).float())
                # print("1.7", time.time() - time_init)
                if param["verbose"]:
                    print("Loss", loss * param["lr"])
                    # for name, param_transp in transport.named_parameters():
                    #     if param_transp.requires_grad:
                    #         print(name, param_transp.data)
                treshold = True
                if treshold:
                    if param["verbose"]:
                        print("torch.abs(loss - old_loss)", torch.abs(loss - old_loss))
                    if torch.abs(old_loss - loss) < (10 ** -3):  # 3
                        continue_while = False
                        break
                if epoch >= param["max_iter"] - 1:  # dist < 0.1 or
                    continue_while = False
                    break
                # print("2.0", time.time() - time_init)
                loss.backward()
                # print("3.0", time.time() - time_init)
                # print(transport.fc1.weight, transport.TT_t.weight)
                optimizer.step()
                old_loss = loss
                # print(transport.fc1.weight, transport.TT_t.weight)
            epoch += 1
            scheduler.step(epoch=epoch)

        transport.train(False)
        if param["new_space"]:
            sourceAdapted = sink.transform(Xs=transport.source_in_source_subspace)
            targetAdapted = transport.target_in_target_subspace
        else:
            sink.xt_ = Tx
            sourceAdapted = sink.transform(Xs=transport.source_in_source_subspace)
            targetAdapted = Tx

    elif algo == "SA_only_target":
        # Variant of the SA method + Optimal Transport
        pcaS = sklearn.decomposition.PCA(param["d"], svd_solver=param["svd_solver"]).fit(Sx)
        pcaT = sklearn.decomposition.PCA(param["d"], svd_solver=param["svd_solver"]).fit(Tx)

        XS = np.transpose(pcaS.components_)  # source subspace matrix
        XT = np.transpose(pcaT.components_)  # target subspace matrix
        Xa = XT @ np.transpose(XT) @ XS @ np.transpose(XS)
        source_in_target_subspace = Sx
        target_in_target_subspace = Tx.dot(Xa)

        transp = ot.da.SinkhornLpl1Transport(reg_e=param["reg_e"], reg_cl=param["reg_cl"], norm="median")
        transp.fit(Xs=source_in_target_subspace, ys=Sy, Xt=target_in_target_subspace)

        sourceAdapted = transp.transform(source_in_target_subspace)
        targetAdapted = target_in_target_subspace

    elif algo == "SA_only_source":
        # Variant of the SA method + Optimal Transport
        pcaS = sklearn.decomposition.PCA(param["d"], svd_solver=param["svd_solver"]).fit(Sx)
        pcaT = sklearn.decomposition.PCA(param["d"], svd_solver=param["svd_solver"]).fit(Tx)

        XS = np.transpose(pcaS.components_)
        XT = np.transpose(pcaT.components_)
        Xa = XS @ np.transpose(XS) @ XT @ np.transpose(XT)
        source_in_target_subspace = Sx.dot(Xa)
        target_in_target_subspace = Tx

        transp = ot.da.SinkhornLpl1Transport(reg_e=param["reg_e"], reg_cl=param["reg_cl"], norm="median")
        transp.fit(Xs=source_in_target_subspace, ys=Sy, Xt=target_in_target_subspace)

        sourceAdapted = transp.transform(source_in_target_subspace)
        targetAdapted = target_in_target_subspace

    elif algo == "SA_both_source":
        # Variant of the SA method + Optimal Transport
        pcaS = sklearn.decomposition.PCA(param["d"], svd_solver=param["svd_solver"]).fit(Sx)
        pcaT = sklearn.decomposition.PCA(param["d"], svd_solver=param["svd_solver"]).fit(Tx)

        XS = np.transpose(pcaS.components_)
        XT = np.transpose(pcaT.components_)
        Xa = XS @ np.transpose(XS) @ XT

        source_in_target_subspace = Sx.dot(Xa)
        target_in_target_subspace = Tx.dot(Xa)

        transp = ot.da.SinkhornLpl1Transport(reg_e=param["reg_e"], reg_cl=param["reg_cl"], norm="median")
        transp.fit(Xs=source_in_target_subspace, ys=Sy, Xt=target_in_target_subspace)

        sourceAdapted = transp.transform(source_in_target_subspace)
        targetAdapted = target_in_target_subspace

    elif algo == "SA_both_target":
        # Variant of the SA method + Optimal Transport
        pcaS = sklearn.decomposition.PCA(param["d"], svd_solver=param["svd_solver"]).fit(Sx)
        pcaT = sklearn.decomposition.PCA(param["d"], svd_solver=param["svd_solver"]).fit(Tx)

        XS = np.transpose(pcaS.components_)
        XT = np.transpose(pcaT.components_)
        Xa = XT @ np.transpose(XT) @ XS
        source_in_target_subspace = Sx.dot(Xa)
        target_in_target_subspace = Tx.dot(Xa)

        transp = ot.da.SinkhornLpl1Transport(reg_e=param["reg_e"], reg_cl=param["reg_cl"], norm="median")
        transp.fit(Xs=source_in_target_subspace, ys=Sy, Xt=target_in_target_subspace)

        sourceAdapted = transp.transform(source_in_target_subspace)
        targetAdapted = target_in_target_subspace

    elif algo == "SAOT":
        # SA method + Optimal Transport
        pcaS = sklearn.decomposition.PCA(param["d"], svd_solver=param["svd_solver"]).fit(Sx)
        pcaT = sklearn.decomposition.PCA(param["d"], svd_solver=param["svd_solver"]).fit(Tx)

        XS = np.transpose(pcaS.components_)
        XT = np.transpose(pcaT.components_)
        Xa = XS.dot(np.transpose(XS)).dot(XT)
        source_in_target_subspace = Sx.dot(Xa)
        target_in_target_subspace = Tx.dot(XT)
        transp = ot.da.SinkhornLpl1Transport(reg_e=param["reg_e"], reg_cl=param["reg_cl"], norm="median")
        transp.fit(Xs=source_in_target_subspace, ys=Sy, Xt=target_in_target_subspace)

        sourceAdapted = transp.transform(source_in_target_subspace)
        targetAdapted = target_in_target_subspace

    elif algo == "SAOT_L1l2":
        # SA method + L1l2 Optimal Transport regularisation.
        pcaS = sklearn.decomposition.PCA(param["d"], svd_solver=param["svd_solver"]).fit(Sx)
        pcaT = sklearn.decomposition.PCA(param["d"], svd_solver=param["svd_solver"]).fit(Tx)

        XS = np.transpose(pcaS.components_)
        XT = np.transpose(pcaT.components_)
        Xa = XS.dot(np.transpose(XS)).dot(XT)
        source_in_target_subspace = Sx.dot(Xa)
        target_in_target_subspace = Tx.dot(XT)

        transp = ot.da.SinkhornL1l2Transport(reg_e=param["reg_e"], reg_cl=param["reg_cl"], norm="median")
        transp.fit(Xs=source_in_target_subspace, ys=Sy, Xt=target_in_target_subspace)

        sourceAdapted = transp.transform(source_in_target_subspace)
        targetAdapted = target_in_target_subspace

    elif algo == "SOT":
        # Variant of the SA method + Optimal Transport
        pcaS = PCA(param["d"], svd_solver=param["svd_solver"]).fit(Sx)
        pcaT = PCA(param["d"], svd_solver=param["svd_solver"]).fit(Tx)

        XS = np.transpose(pcaS.components_)
        XT = np.transpose(pcaT.components_)
        source_in_target_subspace = Sx.dot(XS)
        target_in_target_subspace = Tx.dot(XT)

        transp = ot.da.SinkhornLpl1Transport(reg_e=param["reg_e"], reg_cl=param["reg_cl"], norm="median")
        transp.fit(Xs=source_in_target_subspace, ys=Sy, Xt=target_in_target_subspace)

        sourceAdapted = transp.transform(source_in_target_subspace)
        targetAdapted = target_in_target_subspace

    elif algo == "pytorch":
        # Beginning of the implementation of Sinkhorn with auto grad. However this method does not include the
        # class regularization of OTDA which is a huge disavantage.
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from wassdistance import layers
        # [Wasserstein distances](https://dfdazac.github.io/sinkhorn.html).

        source_in_target_subspace = Sx
        target_in_target_subspace = Tx
        source_in_target_subspace_tensor = torch.tensor(source_in_target_subspace).float().cuda()
        target_in_target_subspace_tensor = torch.tensor(target_in_target_subspace).float().cuda()
        Sy_tensor, Ty_tensor = torch.tensor(Sy), torch.tensor(Ty)

        sinkhorn = layers.SinkhornDistance(eps=param["reg_e"],
                                           max_iter=100,
                                           thresh=10 ** -7)

        class AlexNet(nn.Module):
            def __init__(self, num_feature_in, num_feature_out):
                super(AlexNet, self).__init__()
                self.l1 = nn.Linear(num_feature_in, num_feature_in, bias=False)
                self.l1.weight.data.copy_(torch.eye(num_feature_in, num_feature_out))

            def forward(self, x):
                x = self.l1(x)
                return x

        network = AlexNet(num_feature_in=source_in_target_subspace_tensor.shape[1],
                          num_feature_out=target_in_target_subspace_tensor.shape[1]).cuda()
        print(torch.mean(network.l1.weight.data))
        network.train(True)
        optimizer = optim.SGD(network.parameters(), lr=0.1, momentum=0.9)
        for epoch in range(1):
            optimizer.zero_grad()
            dist, gamma, C = sinkhorn(x=network(source_in_target_subspace_tensor),
                                      y=target_in_target_subspace_tensor)
            # print(gamma)
            if epoch % 10 == 0:
                print(epoch, dist)
            if dist < 0.1:
                break
            dist.backward()
            optimizer.step()
        print(torch.mean(network.l1.weight.data))
        sourceAdapted = gamma.detach().cpu().numpy() @ target_in_target_subspace
        targetAdapted = target_in_target_subspace

    return sourceAdapted, targetAdapted, Sy, Ty


def getAccuracy(trainData, trainLabels, testData, testLabels, type_classifier="1NN"):
    """
    :param trainData:
    :param trainLabels:
    :param testData:
    :param testLabels:
    :param type_classifier:
    :return: The accuracy of the test data train with the train data. Only NN and SVM are implemented
    """
    prediction = getLabel(trainData, trainLabels, testData, type_classifier)
    return 100 * float(sum(prediction == testLabels)) / len(testData)


def main(featuresToUse, numberIteration, adaptationAlgoUsed, type_classifier,
         d, reg_e, reg_cl, reg_l, max_iter, new_space, margin, ML_init,
         max_inner_iter_sink, max_inner_iter_grad, verbose,
         save_pickle, cross_val, test_exp, which_dataset, specific_comparaison, pickle_name,
         cuda, rule, detach, svd_solver, lr=0, reg_pca=0):
    """
    Main function of the code that is launch at the beginning of the code.
    :return: Print result and save files in pickle format if needed.
    """

    if featuresToUse in ["surf", "CaffeNet4096", "decaf6"]:
        domainNames = ['amazon', 'caltech10', 'dslr', 'webcam']
    elif featuresToUse in ["office31fc6", "office31fc7"]:
        domainNames = ['amazon', 'dslr', 'webcam']
    tests = []
    data = {}

    for sourceDomain in domainNames:
        possible_data = loadmat(os.path.join(".", "DATA", featuresToUse,
                                             sourceDomain + '.mat'))
        if featuresToUse == "surf":
            # Normalize the surf histograms
            feat = (possible_data['fts'].astype(float) /
                    np.tile(np.sum(possible_data['fts'], 1),
                            (np.shape(possible_data['fts'])[1], 1)).T)
        elif featuresToUse == "decaf6":
            feat = (possible_data['feas'].astype(float) /
                    np.tile(np.sum(possible_data['feas'], 1),
                            (np.shape(possible_data['feas'])[1], 1)).T)
        elif featuresToUse in ["office31fc6", "office31fc7"]:
            feat = (possible_data['fts'].astype(float) /
                    np.tile(np.sum(possible_data['fts'], 1),
                            (np.shape(possible_data['fts'])[1], 1)).T)
        else:
            feat = possible_data['fts'].astype(float)

        # Z-score
        feat = preprocessing.scale(feat)
        if featuresToUse == "mnist_usps":
            labels = possible_data['Y_src'].ravel()
        else:
            labels = possible_data['labels'].ravel()

        data[sourceDomain] = [feat, labels]

        # The next part can be used to select a subset of the source dataset but it is not used later because this
        # setup seems to not be follow anymore.
        for targetDomain in domainNames:
            if sourceDomain != targetDomain:
                perClassSource = 20
                if sourceDomain == 'dslr':  # or sourceDomain == 'webcam':
                    perClassSource = 8
                tests.append([sourceDomain, targetDomain, perClassSource])

    # Select a subset of the dataset available if needed.
    if which_dataset >= 0:
        tests = [tests[which_dataset]]

    meansAcc = {}
    stdsAcc = {}
    totalTime = {}
    param = {"d": d, "reg_e": reg_e, "reg_cl": reg_cl, "reg_l": reg_l, "reg": 1, "threshold": 95,
             "max_iter": max_iter, "new_space": new_space, "margin": margin, "ML_init": ML_init,
             "verbose": verbose, "max_inner_iter_sink": max_inner_iter_sink,
             "max_inner_iter_grad": max_inner_iter_grad, "numberIteration": numberIteration, "cuda": cuda,
             "rule": rule, "detach": detach, "svd_solver": svd_solver, "lr": lr, "reg_pca": reg_pca}

    param_aux = {}
    for name in adaptationAlgoUsed:
        meansAcc[name] = []
        stdsAcc[name] = []
        totalTime[name] = 0
    my_dict = {}
    return_results = {}

    # if svd solver is set to full, there is no more randomness and the number of iteration is set to 1.
    if param["svd_solver"] == "full":
        param["numberIteration"] = 1
    numberIteration_temp = param["numberIteration"]

    # Loop over each couple of dataset.
    for test in tests:
        Sname = test[0]
        Tname = test[1]
        print(Sname.upper()[:1] + '->' + Tname.upper()[:1], end=" ")
        Sx = data[Sname][0]
        Sy = data[Sname][1]
        Tx = data[Tname][0]
        Ty = data[Tname][1]

        # -------------------- Cross validation ------------------------------------
        if cross_val:
            for name in adaptationAlgoUsed:
                # list of method that are not random
                if name in ["MLOT_id", "OT", "OTDA", "NA", "Tused", "TCA", "CORAL", "JDA", "LMNN", "JDOT", "JDOTSVM",
                            "JDOTe", "JDOTSVMe"]:
                    param["numberIteration"] = 1
                else:
                    param["numberIteration"] = numberIteration_temp
                # launch the cross validation
                param_aux[name] = get_param_optimal(name, Sx, Sy, Tx, Ty, param, Sname, Tname)
            # allow to skip the test part.
            continue

        # --------------------Test ------------------------------------------------
        Sx = data[Sname][0]
        Sy = data[Sname][1]
        Tx = data[Tname][0]
        Ty = data[Tname][1]

        results = {}
        times = {}
        for name in adaptationAlgoUsed:
            results[name] = []
            times[name] = []

        for name in adaptationAlgoUsed:
            startTime = time.time()
            # list of method that are not random
            if name in ["MLOT_id", "OT", "OTDA", "NA", "Tused", "TCA", "CORAL", "JDA", "LMNN", "JDOT", "JDOTSVM",
                        "JDOTe", "JDOTSVMe"]:
                param["numberIteration"] = 1
            else:
                param["numberIteration"] = numberIteration_temp

            for iteration in range(param["numberIteration"]):
                np.random.seed(iteration * 45 + 4988612)
                random.seed(iteration * 65 + 8965321)

                # Adapt the data
                if name == "JDOTSVM" or name == "JDOTSVMe":
                    pred_jdot = adaptData(name, Sx, Sy, Tx, Ty, param)
                else:
                    subSa, Ta, subSay, Tay = adaptData(name, Sx, Sy, Tx, Ty, param)

                # This will save all the data adaptated which can become really huge in some dataset.
                # This can be usefull for comparing different classifier.
                if save_pickle:
                    assert not name == "JDOTSVM"
                    assert not name == "JDOTSVMe"
                    dict_index = name + " " + str(iteration) + " " + Sname.upper()[:1] + "_" + Tname.upper()[:1]
                    my_dict[dict_index + " subSa"] = subSa
                    my_dict[dict_index + " subSay"] = subSay
                    my_dict[dict_index + " Ta"] = Ta
                    my_dict[dict_index + " Tay"] = Tay
                if name == "JDOTSVM" or name == "JDOTSVMe":
                    results[name].append(100 * float(sum(pred_jdot == Ty)) / len(Ty))
                else:
                    results[name].append(getAccuracy(subSa, subSay, Ta, Tay, type_classifier=type_classifier))
                times[name].append(time.time() - startTime)
                if specific_comparaison == "None":
                    print(".", end="")

        if specific_comparaison == "None":
            print("")
        return_results[Sname.upper()[:1] + '->' + Tname.upper()[:1]] = {}
        for name in adaptationAlgoUsed:
            meanAcc = np.mean(results[name])
            stdAcc = np.std(results[name])
            meansAcc[name].append(meanAcc)
            stdsAcc[name].append(stdAcc)
            totalTime[name] += sum(times[name])
            if specific_comparaison == "None":
                print("     {:4.1f}".format(meanAcc) + "  {:3.1f}".format(stdAcc) +
                      "  {:6}".format(name) + " {:6.2f}s".format(sum(times[name])))
            return_results[Sname.upper()[:1] + '->' + Tname.upper()[:1]][name] = {"mean": meanAcc, "std": stdAcc}
        if test_exp:
            # Only one run in test mode
            break

    if save_pickle:
        pickle_out = open("pickle_SVM/" + pickle_name + ".pickle", "wb")
        pickle.dump(my_dict, pickle_out)
        pickle_out.close()
    print("")
    print("Mean results and total time")
    for name in adaptationAlgoUsed:
        meanMean = np.mean(meansAcc[name])
        meanStd = np.mean(stdsAcc[name])
        print("     {:4.1f}".format(meanMean) + "  {:3.1f}".format(meanStd) +
              "  {:6}".format(name) + " {:6.2f}s".format(totalTime[name]))
    return return_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MLOT')
    parser.add_argument('--featuresToUse', type=str, default='surf',
                        choices=["CaffeNet4096", "surf", "decaf6", "office31fc6", "office31fc7"])
    parser.add_argument('--numberIteration', type=int, default=10,
                        help="Number of iterations of each method, this is usefull for random methods.")
    parser.add_argument('--adaptationAlgoUsed', type=str, default='MLOT',
                        help="Name of the method that should be used. Method should be separate by comma if more than \
                             one method is needed (SA,TCA,NA)")
    parser.add_argument('--type_classifier', type=str, default='1NN',
                        help="Final classifier used, only NN and SVM is implemented. 4NN will search for the 4\
                         NearestNeighbors. SVM_10.1 will launch the SVM with a margin of 10.1 ")
    parser.add_argument('--d', type=int, default=70,
                        help="Dimension for PCA")
    parser.add_argument('--which_dataset', type=int, default=-1,
                        help="The dataset to use in the feature list. -1 mean all dataset should be run.")
    parser.add_argument('--reg_e', type=float, default=0.2,
                        help="Entropy regularisation of the sinkhorn method")
    parser.add_argument('--reg_cl', type=float, default=0.2,
                        help="Class regularisation from OTDA")
    parser.add_argument('--reg_l', type=float, default=1,
                        help="Metric Learning regularisation term")
    parser.add_argument('--lr', type=float, default=0.001,
                        help="Learning rate of the gradient descent of the pytorch method")
    parser.add_argument('--reg_pca', type=float, default=1,
                        help="Regularisation to force to not go far from PCA init, used only in the pytorch version")
    parser.add_argument('--max_iter', type=int, default=10,
                        help="number max of iterations of MLOT")
    parser.add_argument('--max_inner_iter_grad', type=int, default=1,
                        help="number max of iteration of the gradient in ML")
    parser.add_argument('--margin', type=float, default=10,
                        help="Margin used in LMNN (Large Margin Nearest Neighbors)")
    parser.add_argument('--pickle_name', type=str, default="test",
                        help="Pickle_name to save pickle. It will have different location depending if the cross \
                        validation or the -s option or the specific comparaison is used")
    parser.add_argument('--time_cross_val', type=float, default=1,
                        help="Time of cross_validation in hour for 1 couple of dataset")
    parser.add_argument('--svd_solver', type=str, default="auto", choices=["auto", "full", "arpack", "randomized"],
                        help="Solver of the PCA")

    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Will print more information about the run")
    parser.add_argument("-n", "--new_space", action="store_true",
                        help="Should be used most of the time, compute the final classifier in the modify target \
                             space and not in the intial target space.")
    parser.add_argument("-s", "--save_pickle", action="store_true",
                        help="Will save the data adapted, this can be huge for some dataset.")
    parser.add_argument("-sa", "--SA", action="store_true",
                        help="Will use the SA algorithm during the cross validation to go back from the target to the \
                        source")
    parser.add_argument("-c", "--cross_val", action="store_true",
                        help="Will launch the cross validation and skip the test.")
    parser.add_argument('--specific_comparaison', type=str, choices=["None", "OTDAvsMLOT"], default="None",
                        help="If OTDAvsMLOT is set, it will run the code to save a pickle with the comparison of \
                        the method that are in adaptationAlgoUsed")
    parser.add_argument("-t", "--test_exp", action="store_true",
                        help="Will run only one iteration")

    parser.add_argument("-cu", "--cuda", action="store_true",
                        help="Put the Neural Network on GPU (only Pytorch version)")
    parser.add_argument('--rule', type=str, choices=["median", "max", "mean"], default="median",
                        help="Rule of normalization for the cost matrix of the transport")
    parser.add_argument("-d", "--detach", action="store_true",
                        help="Will detach the normalization term of the cost matrix, it will be considered has a \
                        constant for pytorch")
    args = parser.parse_args()

    print(args)
    if args.save_pickle:
        print("Name of the pickle file associated with this run: ", args.pickle_name)
    if args.specific_comparaison == "None":
        main(featuresToUse=args.featuresToUse,
             numberIteration=args.numberIteration,
             adaptationAlgoUsed=args.adaptationAlgoUsed.split(","),
             type_classifier=args.type_classifier,
             d=args.d,
             reg_e=args.reg_e, reg_cl=args.reg_cl, reg_l=args.reg_l,
             max_iter=args.max_iter,
             max_inner_iter_sink=10,
             max_inner_iter_grad=args.max_inner_iter_grad,
             new_space=args.new_space,
             margin=args.margin,
             ML_init="full_identity",
             verbose=args.verbose,
             save_pickle=args.save_pickle,
             cross_val=args.cross_val,
             test_exp=args.test_exp,
             which_dataset=args.which_dataset,
             specific_comparaison=args.specific_comparaison,
             pickle_name=args.pickle_name,
             lr=args.lr,
             svd_solver=args.svd_solver,
             reg_pca=args.reg_pca,
             cuda=args.cuda,
             rule=args.rule,
             detach=args.detach)
    elif args.specific_comparaison == "OTDAvsMLOT":
        # This part of the code reproduce the figure to compare OTDA and MLOT on a range of parameters.
        dict_total = None
        counter = 0
        if args.reg_cl == 0:
            reg_cl_list = [0]
        else:
            reg_cl_list = [0, 0.05, 0.1, 0.5, 1.0]
        reg_e_list = [0.1, 0.2, 0.3, 0.4, 0.5, 1, 5.0]
        # too high value of reg_cl with low value of reg_e seems to make sinkhorn not stable.
        for reg_e in reg_e_list:
            for reg_cl in reg_cl_list:
                if reg_e == 0.2 and reg_cl == 10:
                    # Not stable
                    continue
                time_loop = time.time()
                print("\nreg_e :", reg_e, "reg_cl", reg_cl)
                counter += 1
                result = main(featuresToUse=args.featuresToUse,
                              numberIteration=args.numberIteration,
                              adaptationAlgoUsed=args.adaptationAlgoUsed.split(","),
                              type_classifier=args.type_classifier,
                              d=args.d,
                              reg_e=reg_e, reg_cl=reg_cl,
                              reg_l=args.reg_l,
                              max_iter=args.max_iter,
                              max_inner_iter_sink=10,
                              max_inner_iter_grad=args.max_inner_iter_grad,
                              new_space=args.new_space,
                              margin=args.margin,
                              ML_init="full_identity",
                              verbose=args.verbose,
                              save_pickle=args.save_pickle,
                              cross_val=args.cross_val,
                              test_exp=args.test_exp,
                              which_dataset=args.which_dataset,
                              pickle_name=args.pickle_name,
                              specific_comparaison=args.specific_comparaison,
                              svd_solver=args.svd_solver,
                              lr=args.lr,
                              reg_pca=args.reg_pca,
                              cuda=args.cuda,
                              rule=args.rule,
                              detach=args.detach)
                pickle_out = open(args.pickle_name + args.specific_comparaison +
                                  args.featuresToUse + ".pickle", "ab")
                pickle.dump({"reg_e": reg_e, "reg_l": args.reg_l, "reg_cl": reg_cl, "margin": args.margin,
                             "max_iter": args.max_iter, "d": args.d}, pickle_out)
                pickle.dump(result, pickle_out)
                pickle_out.close()
                if dict_total is None:
                    dict_total = result
                else:
                    for key in dict_total:
                        for key_algo in dict_total[key]:
                            dict_total[key][key_algo]["mean"] += result[key][key_algo]["mean"]
                            dict_total[key][key_algo]["std"] += result[key][key_algo]["std"]
                print("Time for one run:", time.time() - time_loop, "s")

        total_mean = {}
        total_std = {}
        for key_algo in dict_total[list(dict_total.keys())[0]]:
            total_mean[key_algo] = []
            total_std[key_algo] = []
        print("\nMean for each dataset:")
        for key in dict_total:
            print(key)
            for key_algo in dict_total[key]:
                total_mean[key_algo].append(dict_total[key][key_algo]["mean"] / counter)
                total_std[key_algo].append(dict_total[key][key_algo]["std"] / counter)
                print("     {:4.1f}".format(dict_total[key][key_algo]["mean"] / counter) +
                      "  {:3.1f}".format(dict_total[key][key_algo]["std"] / counter) +
                      "  {:6}".format(key_algo))
        print("\nTotal mean:")
        for key_algo in dict_total[list(dict_total.keys())[0]]:
            print("     {:4.1f}".format(np.mean(total_mean[key_algo])) + "  {:3.1f}".format(
                np.mean(total_std[key_algo])) +
                  "  {:6}".format(key_algo))
