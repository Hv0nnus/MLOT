#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Distance metric learning for large margin nearest neighbor classification
# Kilian Q.Weinberger, Lawrence K. Saul
# Journal of Machine Learning Research
# 2009

from sklearn.utils.validation import check_array
from sklearn.utils.validation import check_random_state
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.utils import gen_batches
import sklearn
from scipy import optimize
import numpy as np
from scipy import sparse
from .utils import dist, cost_normalization
import time


def sum_outer_products(X, weights, remove_zero=False):
    weights_sym = weights + weights.T
    if remove_zero:
        _, cols = weights_sym.nonzero()
        ind = np.unique(cols)
        weights_sym = weights_sym.tocsc()[:, ind].tocsr()[ind, :]
        X = X[ind]

    n = weights_sym.shape[0]
    diag = sparse.spdiags(weights_sym.sum(axis=0), 0, n, n)
    laplacian = diag.tocsr() - weights_sym
    sodw = X.T.dot(laplacian.dot(X))
    return sodw


def creat_mini_batch(ys, mini_batch_size, Xt_len):
    assert mini_batch_size <= len(ys)
    unique, counts = np.unique(ys, return_counts=True)
    # mini_batch_size = mini_batch_size * len(unique)
    number_batch = min(int(len(ys) / mini_batch_size), np.min(counts))
    # print(number_batch, len(ys), mini_batch_size, counts)
    #     counts = np.round(mini_batch_size*counts/len(ys))
    S = [0] * len(unique)
    for cl in range(len(unique)):
        s = np.random.permutation(np.where(ys == unique[cl])[0])
        s = s[:len(s) - len(s) % number_batch]
        s = np.split(s, number_batch)
        S[cl] = np.array(s)
    S = np.concatenate(S, axis=1)

    # If we want to use small part of Xs
    t = np.random.permutation(Xt_len)
    t = t[:len(t) - len(t) % number_batch]
    t = np.split(t, number_batch)

    # If we use all Xt
    # t = [np.arange(Xt_len) for _ in range(number_batch)]
    return S, t


def gradient_L(gamma, L, Xs, Xt, which_norm="malhanobis"):
    time1 = time.time()
    if which_norm == "malhanobis":
        # print("5.1", time.time() - time1)
        LZ = np.expand_dims(np.dot(Xs, L), axis=1) - np.expand_dims(np.dot(Xt, L), axis=0)
        normLZ = np.sqrt(np.sum(LZ**2, axis=2))
        gamma = np.divide(gamma, normLZ)
        # print("5.2", time.time() - time1)
        gamma2Z = np.expand_dims(gamma, axis=2) * (np.expand_dims(Xs, axis=1) - np.expand_dims(Xt, axis=0))
        # print("5.3", time.time() - time1)
        gradient_transport = np.expand_dims(LZ, axis=3) * np.expand_dims(gamma2Z, axis=2)
        # print("5.4", time.time() - time1)
        gradient_transport = np.sum(gradient_transport, axis=(0, 1))
        # print("5.5", time.time() - time1)

    elif which_norm == "L_only_on_Xs":
        Lxt_xs = np.expand_dims(Xs @ L, axis=1) - np.expand_dims(Xt, axis=0)
        gamma = np.divide(gamma, np.sqrt(np.sum(Lxt_xs**2, axis=2)))
        gamma2Lxt_xs = np.expand_dims(gamma, axis=2) * Lxt_xs
        gradient_transport = Xs[:, np.newaxis, :, np.newaxis] * np.expand_dims(gamma2Lxt_xs, axis=2)
        gradient_transport = np.sum(gradient_transport, axis=(0, 1))

    elif which_norm == "malhanobisM":
        Z = np.expand_dims(Xs, axis=1) - np.expand_dims(Xt, axis=0)
        # print("5.2", time.time() - time1)
        gamma2Z = 2 * np.expand_dims(gamma, axis=2) * Z
        # print("5.3", time.time() - time1)
        gradient_transport = np.expand_dims(Z, axis=3) * np.expand_dims(gamma2Z, axis=2)
        # print("5.4", time.time() - time1)
        gradient_transport = np.sum(gradient_transport, axis=(0, 1))
        # print("5.5", time.time() - time1)

    return gradient_transport


class LargeMarginNearestNeighbor():
    def __init__(self, Xt, Xs, y,
                 k=3, mu=0.5, nFtsOut=None, maxCst=int(1e7), margin=1,
                 randomState=None, maxiter=100, loss_func=None, grad_func=None, mini_batch_size=10,
                 reg_l=10 ** -5,
                 verbose=False,
                 which_norm="malhanobis",
                 ML_init=False):
        self.Xt = Xt
        self.Xs = Xs
        self.y = y
        self.k_ = k
        self.mu = mu
        self.nFtsOut_ = nFtsOut
        self.maxCst = maxCst
        self.margin = margin
        self.randomState = randomState
        self.maxiter = maxiter
        self.loss_func = loss_func
        self.grad_func = grad_func
        self.mini_batch_size = mini_batch_size
        self.time = time.time()
        self.reg_l = reg_l
        self.verbose = verbose
        self.which_norm = which_norm

        # Store the appearing classes and the class index for each sample
        self.labels_, self.y_ = np.unique(y, return_inverse=True)
        self.classes_ = np.arange(len(self.labels_))
        # Check that the number of neighbors is achievable for all classes
        min_class_size = np.bincount(self.y_).min()
        max_neighbors = min_class_size - 1
        self.k_ = min(self.k_, max_neighbors)
        if ML_init == "PCA":
            pcaS = sklearn.decomposition.PCA(max(self.nFtsOut_//2, self.Xs.shape[1]//2), svd_solver="auto").fit(self.Xs)
            pcaT = sklearn.decomposition.PCA(max(self.nFtsOut_//2, self.Xs.shape[1]//2), svd_solver="auto").fit(self.Xt)
            self.L_ = np.concatenate((pcaS.components_, pcaT.components_), axis=0)
            self.nFtsOut_ = np.shape(self.L_)[0]
        elif ML_init == "old_ML":
            # Initialize matrix L
            cov_ = np.cov(self.Xs, rowvar=False)
            evals, evecs = np.linalg.eigh(cov_)
            self.L_ = np.fliplr(evecs).T  # Initialize to eigenvectors
            if self.nFtsOut_ is None:
                self.nFtsOut_ = self.L_.shape[0]
            nFtsIn = self.Xs.shape[1]
            if self.nFtsOut_ > nFtsIn:
                self.nFtsOut_ = nFtsIn
            if self.L_.shape[0] > self.nFtsOut_:
                self.L_ = self.L_[:self.nFtsOut_]
        elif ML_init == "SSTT":
            pcaS = sklearn.decomposition.PCA(self.nFtsOut_, svd_solver="auto").fit(self.Xs)
            pcaT = sklearn.decomposition.PCA(self.nFtsOut_, svd_solver="auto").fit(self.Xt)
            XS = np.transpose(pcaS.components_)
            XT = np.transpose(pcaT.components_)
            self.L_ = np.transpose(XS @ np.transpose(XS) @ XT @ np.transpose(XT))
            self.nFtsOut_ = np.shape(self.L_)[0]
        elif ML_init == "SS":
            pcaS = sklearn.decomposition.PCA(self.nFtsOut_, svd_solver="auto").fit(self.Xs)
            XS = np.transpose(pcaS.components_)
            self.L_ = np.transpose(XS @ np.transpose(XS))
            self.nFtsOut_ = np.shape(self.L_)[0]
        elif ML_init == "identity":
            self.L_ = np.eye(self.nFtsOut_, self.Xt.shape[1])
        elif ML_init == "full_identity":
            self.L_ = np.eye(self.Xt.shape[1])
        self.nFtsOut_ = np.shape(self.L_)[0]

        # Find target neighbors (fixed)
        self.targets_ = np.empty((self.Xs.shape[0], self.k_), dtype=int)
        for class_ in self.classes_:
            class_ind, = np.where(np.equal(self.y_, class_))
            dist = euclidean_distances(self.Xs[class_ind], squared=True)
            np.fill_diagonal(dist, np.inf)
            nghIdx = np.argpartition(dist, self.k_ - 1, axis=1)
            nghIdx = nghIdx[:, :self.k_]
            # argpartition doesn't guarantee sorted order, so we sort again but
            # only the k neighbors
            rowIdx = np.arange(len(class_ind))[:, None]
            nghIdx = nghIdx[rowIdx, np.argsort(dist[rowIdx, nghIdx])]
            self.targets_[class_ind] = class_ind[nghIdx]
        # Compute gradient component of target neighbors (constant)
        n, k = self.targets_.shape
        rows = np.repeat(np.arange(n), k)
        cols = self.targets_.flatten()
        targets_sparse = sparse.csr_matrix((np.ones(n * k),
                                            (rows, cols)), shape=(n, n))

        self.gradStatic = sum_outer_products(self.Xs, targets_sparse)

    def fit(self, transp):
        self.transp = transp

        # Call optimizer
        L, loss, details = optimize.fmin_l_bfgs_b(
            func=self._loss_grad, x0=self.L_, maxiter=self.maxiter)
        if self.verbose:
            print(details)
        # Reshape result from optimizer
        self.L_ = L.reshape(self.nFtsOut_, L.size // self.nFtsOut_)
        return self

    def transform(self, X=None):
        if X is None:
            X = self.Xs
        else:
            X = check_array(X)
        return X.dot(self.L_.T)

    def _loss_grad(self, L):
        # print("4", time.time() - self.time)
        n, nFtsIn = self.Xs.shape
        self.L_ = L.reshape(self.nFtsOut_, nFtsIn)
        Lx = self.transform()

        # Compute distances to target neighbors under L (plus margin 1)
        dist_tn = np.zeros((n, self.k_))
        for k in range(self.k_):
            dist_tn[:, k] = np.sum(np.square(Lx - Lx[self.targets_[:, k]]),
                                   axis=1) + self.margin
        margin_radii = dist_tn[:, -1]
        # print(margin_radii)

        # Compute distances to impostors under L
        imp1, imp2, dist_imp = self._find_impostors(Lx, margin_radii)

        loss = 0
        A0 = sparse.csr_matrix((n, n))
        for k in reversed(range(self.k_)):
            loss1 = np.maximum(dist_tn[imp1, k] - dist_imp, 0)
            act, = np.where(loss1 != 0)
            A1 = sparse.csr_matrix((2 * loss1[act], (imp1[act], imp2[act])),
                                   (n, n))
            loss2 = np.maximum(dist_tn[imp2, k] - dist_imp, 0)
            act, = np.where(loss2 != 0)
            A2 = sparse.csr_matrix((2 * loss2[act], (imp1[act], imp2[act])),
                                   (n, n))
            vals = np.squeeze(np.asarray(A2.sum(0) + A1.sum(1).T))
            A0 = A0 - A1 - A2 + sparse.csr_matrix(
                (vals, (range(n), self.targets_[:, k])), (n, n))
            loss = loss + np.sum(loss1 ** 2) + np.sum(loss2 ** 2)
        grad_new = sum_outer_products(self.Xs, A0, remove_zero=True)
        df = self.L_.dot((1 - self.mu) * self.gradStatic + self.mu * grad_new)
        df *= 2
        loss = ((1 - self.mu) * (self.gradStatic * (self.L_.T.dot(self.L_))).sum() +
                self.mu * loss)
        if self.reg_l < 100:
            mini_batch_list = creat_mini_batch(self.y, self.mini_batch_size, len(self.Xt))
            W = np.zeros(np.shape(self.L_))
            for index_mini_batch in range(len(mini_batch_list[1])):
                mini_batch_s, mini_batch_t = mini_batch_list[0][index_mini_batch], mini_batch_list[1][index_mini_batch]
                # print("5", time.time() - self.time)
                W += gradient_L(gamma=self.transp[mini_batch_s, :][:, mini_batch_t],
                                L=np.transpose(self.L_),
                                Xs=self.Xs[mini_batch_s, :],
                                Xt=self.Xt[mini_batch_t, :],
                                which_norm=self.which_norm)
                break
            if self.which_norm == "malhanobis":
                C = euclidean_distances(self.Xs @ np.transpose(self.L_),
                                        self.Xt @ np.transpose(self.L_),
                                        squared=False)
            elif self.which_norm == "L_only_on_Xs":
                C = euclidean_distances(self.Xs @ np.transpose(self.L_),
                                        self.Xt,
                                        squared=False)

            loss_transp = np.sum(C * self.transp)
            median_C = float(np.median(C))

        else:
            # print("Warning. Only ML is used here.")
            loss_transp = 0
            W = np.zeros(np.shape(self.L_))
            median_C = 1
        if self.verbose:
            print("GRAD et LOSS")
            print("grad_", np.mean((self.reg_l * df))**2)
            print("grad_transp", np.mean(W**2))
            print("norm L", np.linalg.norm(self.L_))
            print("loss", self.reg_l * loss)
            print("loss_transp", loss_transp)
        return self.reg_l * loss + loss_transp / median_C, (self.reg_l * df + W / median_C).flatten()

    def _find_impostors(self, Lx, margin_radii):
        n = Lx.shape[0]
        impostors = sparse.csr_matrix((n, n), dtype=np.int8)
        for class_ in self.classes_[:-1]:
            imp1, imp2 = [], []
            ind_in, = np.where(np.equal(self.y_, class_))
            ind_out, = np.where(np.greater(self.y_, class_))
            # Subdivide idx_out x idx_in to chunks of a size that is
            # fitting in memory
            ii, jj = self._find_impostors_batch(Lx[ind_out], Lx[ind_in],
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
            imp1, imp2 = imp1[ind_subsample], imp2[ind_subsample]
        dist = np.zeros(len(imp1))
        for chunk in gen_batches(len(imp1), 500):
            dist[chunk] = np.sum(np.square(Lx[imp1[chunk]] - Lx[imp2[chunk]]),
                                 axis=1)
        return imp1, imp2, dist

    @staticmethod
    def _find_impostors_batch(x1, x2, t1, t2, batch_size=500):
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
