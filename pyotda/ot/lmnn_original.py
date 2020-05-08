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
from scipy import optimize
import numpy as np
from scipy import sparse


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


class LargeMarginNearestNeighbor():
    def __init__(self, k=3, mu=0.5, nFtsOut=None, maxCst=int(1e7),
                 randomState=None, maxiter=100, margin=1):
        self.k_ = k
        self.mu = mu
        self.nFtsOut_ = nFtsOut
        self.maxCst = maxCst
        self.randomState = randomState
        self.maxiter = maxiter
        self.margin = margin


    def fit(self, X, y):
        self.X_ = X
        self.y = y
        # Store the appearing classes and the class index for each sample
        self.labels_, self.y_ = np.unique(y, return_inverse=True)
        self.classes_ = np.arange(len(self.labels_))
        # Check that the number of neighbors is achievable for all classes
        min_class_size = np.bincount(self.y_).min()
        max_neighbors = min_class_size - 1
        self.k_ = min(self.k_, max_neighbors)
        # Initialize matrix L
        cov_ = np.cov(self.X_, rowvar=False)
        evals, evecs = np.linalg.eigh(cov_)
        self.L_ = np.fliplr(evecs).T  # Initialize to eigenvectors
        if self.nFtsOut_ is None:
            self.nFtsOut_ = self.L_.shape[0]
        nFtsIn = self.X_.shape[1]
        if self.nFtsOut_ > nFtsIn:
            self.nFtsOut_ = nFtsIn
        if self.L_.shape[0] > self.nFtsOut_:
            self.L_ = self.L_[:self.nFtsOut_]
        # Find target neighbors (fixed)
        self.targets_ = np.empty((self.X_.shape[0], self.k_), dtype=int)
        for class_ in self.classes_:
            class_ind, = np.where(np.equal(self.y_, class_))
            dist = euclidean_distances(self.X_[class_ind], squared=True)
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

        self.gradStatic = sum_outer_products(self.X_, targets_sparse)
        # Call optimizer
        L, loss, details = optimize.fmin_l_bfgs_b(
                                 func=self._loss_grad, x0=self.L_, maxiter=self.maxiter)
        # Reshape result from optimizer
        self.L_ = L.reshape(self.nFtsOut_, L.size // self.nFtsOut_)
        return self

    def transform(self, X=None):
        if X is None:
            X = self.X_
        else:
            X = check_array(X)
        return X.dot(self.L_.T)

    def _loss_grad(self, L):
        n, nFtsIn = self.X_.shape
        self.L_ = L.reshape(self.nFtsOut_, nFtsIn)
        Lx = self.transform()

        # Compute distances to target neighbors under L (plus margin 1)
        dist_tn = np.zeros((n, self.k_))
        for k in range(self.k_):
            dist_tn[:, k] = np.sum(np.square(Lx - Lx[self.targets_[:, k]]),
                                   axis=1) + self.margin
        margin_radii = dist_tn[:, -1]

        # Compute distances to impostors under L
        imp1, imp2, dist_imp = self._find_impostors(Lx, margin_radii)

        loss = 0
        A0 = sparse.csr_matrix((n, n))
        for k in reversed(range(self.k_)):
            loss1 = np.maximum(dist_tn[imp1, k] - dist_imp, 0)
            act, = np.where(loss1 != 0)
            A1 = sparse.csr_matrix((2*loss1[act], (imp1[act], imp2[act])),
                                   (n, n))
            loss2 = np.maximum(dist_tn[imp2, k] - dist_imp, 0)
            act, = np.where(loss2 != 0)
            A2 = sparse.csr_matrix((2*loss2[act], (imp1[act], imp2[act])),
                                   (n, n))
            vals = np.squeeze(np.asarray(A2.sum(0) + A1.sum(1).T))
            A0 = A0 - A1 - A2 + sparse.csr_matrix(
                               (vals, (range(n), self.targets_[:, k])), (n, n))
            loss = loss + np.sum(loss1 ** 2) + np.sum(loss2 ** 2)
        grad_new = sum_outer_products(self.X_, A0, remove_zero=True)
        df = self.L_.dot((1-self.mu)*self.gradStatic + self.mu*grad_new)
        df *= 2
        loss = ((1-self.mu)*(self.gradStatic*(self.L_.T.dot(self.L_))).sum() +
                self.mu*loss)
        return loss, df.flatten()

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
