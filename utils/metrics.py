"""Evaluation metric for the cost models"""
import numpy as np
from utils.util import INFINITE_ERROR, warn_once
import time

try:
    import torch
except:
    warn_once("Fail to import torch")

def norm(x):
    if isinstance(x, np.ndarray):
        return np.linalg.norm(x)
    elif isinstance(x, torch.Tensor):
        return torch.norm(x, p=2)
    else:
        raise ValueError()

def l2diff(x1, x2):
    """
    standard euclidean norm
    """
    # return (x1-x2).norm(p=2)
    return norm(x1-x2)

def max_curve(trial_scores):
    """Make a max curve f(n) = max([s[i] fo i < n]) """
    ret = np.empty(len(trial_scores))
    keep = -1e9
    for i, score in enumerate(trial_scores):
        keep = max(keep, score)
        ret[i] = keep
    return ret


def metric_r_squared(preds, labels):
    """Compute R^2 value"""
    s_tot = np.sum(np.square(labels - np.mean(labels)))
    s_res = np.sum(np.square(labels - preds))
    if s_tot < 1e-6:
        return 1
    return 1 - s_res / s_tot


def metric_rmse(preds, labels):
    """Compute RMSE (Rooted Mean Square Error)"""
    return np.sqrt(np.mean(np.square(preds - labels)))


def vec_to_pair_com(vec):
    return (vec.reshape((-1, 1)) - vec) > 0


def metric_pairwise_comp_accuracy(preds, labels):
    """Compute the accuracy of pairwise comparision"""
    n = len(preds)
    if n <= 1:
        return 0.5
    preds = vec_to_pair_com(preds)
    labels = vec_to_pair_com(labels)
    correct_ct = np.triu(np.logical_not(np.logical_xor(preds, labels)), k=1).sum()
    return correct_ct / (n * (n-1) / 2)


def metric_top_k_recall(preds, labels, top_k):
    """Compute recall of top-k@k = |(top-k according to prediction) intersect (top-k according to ground truth)| / k."""
    real_top_k = set(np.argsort(-labels)[:top_k])
    predicted_top_k = set(np.argsort(-preds)[:top_k])
    recalled = real_top_k.intersection(predicted_top_k)
    return 1.0 * len(recalled) / top_k


def metric_peak_score(preds, labels, top_k):
    """Compute average peak score"""
    trials = np.argsort(preds)[::-1][:top_k]
    trial_scores = labels[trials]
    curve = max_curve(trial_scores) / np.max(labels)
    return np.mean(curve)


def metric_elementwise_mape(preds, labels):
    preds = preds.flatten()
    labels = labels.flatten()
    assert preds.shape == labels.shape
    return np.abs((labels-preds)/labels)

def metric_mape(preds, labels):
    return float(np.mean(metric_elementwise_mape(preds, labels)))

def metric_error_accuracy(preds, labels, error_bounds):
    if not isinstance(error_bounds, list):
        error_bounds = [error_bounds]
    mape = metric_elementwise_mape(preds, labels)
    rst = []
    for error_bound in error_bounds:
        rst.append(float(1.0 * sum(mape <= error_bound) / len(preds)))
    return rst


def random_mix(values, randomness):
    random_values = np.random.uniform(np.min(values), np.max(values), len(values))
    return randomness * random_values + (1 - randomness) * values


from scipy.optimize import curve_fit
def inverse_proportional_func(x, a, b, c):
    return a / (x + b) + c
p0 = [1, 1, 0]
bounds = ((0, 0, 0), (np.inf, np.inf, np.inf))

def forecast_converge_value_impl(scalars, weight):
    ''' According to the input list of scalars, predict the convergence point
    Parameters
    ----------
    scalars: list
        A list of scalars sorted in chronological order
    weight: float between 0 and 1
        The smoothing rate
    '''
    assert weight < 1 and weight >= 0 and len(scalars) > 2
    if isinstance(scalars[0], tuple):
        xdata, scalars = zip(*scalars)
    else:
        xdata = np.arange(len(scalars))+1
    
    if weight > 0:
        ### Smooth, refer to https://stackoverflow.com/questions/42281844/what-is-the-mathematics-behind-the-smoothing-parameter-in-tensorboards-scalar
        last = scalars[0]  # First value in the plot (first timestep)
        smoothed = []
        for point in scalars:
            smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
            smoothed.append(smoothed_val)                        # Save it
            last = smoothed_val                                  # Anchor the last smoothed value
        scalars = smoothed_val
    
    popt, pcov = curve_fit(inverse_proportional_func, xdata, scalars, p0, bounds=bounds)
    return popt

def forecast_convergence_value(scalars, x=None, weight=0):
    ''' According to the input list of scalars, predict the convergence point
    Parameters
    ----------
    scalars: list
        A list of scalars sorted in chronological order
    x: numerical value
        Use x to get the corresponding forecased value. Set it to None means x=inf
    weight: float between 0 and 1
        The smoothing rate
    '''
    try:
        popt = forecast_converge_value_impl(scalars, weight)
    except RuntimeError:
        return INFINITE_ERROR
    if x is None:
        return popt[-1]
    else:
        return inverse_proportional_func(x, *popt)
    
def test_forecast_convergence_value(scalars=None, weight=0):
    if scalars is None:
        scalars = [100, 50, 20, 18]
    popt = forecast_converge_value_impl(scalars, weight)
    print(popt)
    def test_func(x):
        return inverse_proportional_func(x, *popt)
    for x in [1, 2, 3, 4, 100, 1000, 10000]:
        print(x, test_func(x))

def centroid_diff(x1, x2):
    '''
    Centroid difference between two distributions
    
    Parameters
    ----------
    X, X_test: np.ndarray of shape (N_sample, N_entry)
    '''
    xc1 = x1.mean(0)
    xc2 = x2.mean(0)
    return l2diff(xc1, xc2)

###############################################################
# Metric CMD
# Refer to "CENTRAL MOMENT DISCREPANCY (CMD) FOR DOMAIN-INVARIANT REPRESENTATION LEARNING",
# ICLR 2017
def moment_diff(sx1, sx2, k):
    """
    difference between moments
    """
    if isinstance(sx1, np.ndarray):
        ss1 = np.power(sx1, k)
        ss2 = np.power(sx2, k)
    elif isinstance(sx2, torch.Tensor):
        ss1 = torch.pow(sx1, k)
        ss2 = torch.pow(sx2, k)
    else:
        raise ValueError()

    ss1 = ss1.mean(0)
    ss2 = ss2.mean(0)

    return l2diff(ss1, ss2)

def metric_cmd(X, X_test, K=5, bound=1):
    """
    central moment discrepancy (cmd)

    Parameters
    ----------
    X, X_test: np.ndarray of shape (N_sample, N_entry)
    """
    assert len(X.shape) == 2 and len(X_test.shape) == 2
    x1 = X
    x2 = X_test
    mx1 = x1.mean(0)
    mx2 = x2.mean(0)

    sx1 = x1 - mx1
    sx2 = x2 - mx2

    dm = l2diff(mx1, mx2) / bound

    scms = [dm]
    for i in range(K-1):
        scms.append(moment_diff(
            sx1, sx2, i+2)/(bound**(i+2)))
    return sum(scms)

########### MMD ################

def pairwise_distances(x, y):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]

    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x**2).sum(1).view(-1, 1)
    y_t = torch.transpose(y, 0, 1)
    y_norm = (y**2).sum(1).view(1, -1)
    
    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    return torch.clamp(dist, 0.0, np.inf)

def metric_mmd(X1, X2):
    ''' Reference:
    Paper: Minimax Estimation of Maximum Mean Discrepancy with Radial Kernels, NeurIPS16, https://papers.nips.cc/paper/2016/hash/5055cbf43fac3f7e2336b27310f0b9ef-Abstract.html
    Implementation: Shift-Robust GNNs: Overcoming the Limitations of Localized Graph Training Data, NeurIPS21, https://arxiv.org/pdf/2108.01099.pdf
    '''
    if isinstance(X1, np.ndarray):
        X1 = torch.Tensor(X1)
    if isinstance(X2, np.ndarray):
        X2 = torch.Tensor(X2)
    X1X1 = pairwise_distances(X1, X1)
    X1X2 = pairwise_distances(X1, X2)
    X2X2 = pairwise_distances(X2, X2)

    H = torch.exp(- 1e0 * X1X1) + torch.exp(- 1e-1 * X1X1) + torch.exp(- 1e-3 * X1X1)
    f = torch.exp(- 1e0 * X1X2) + torch.exp(- 1e-1 * X1X2) + torch.exp(- 1e-3 * X1X2)
    z = torch.exp(- 1e0 * X2X2) + torch.exp(- 1e-1 * X2X2) + torch.exp(- 1e-3 * X2X2)

    MMD_dist = H.mean() - 2 * f.mean() + z.mean()
    return MMD_dist

############### KMM ################

def metric_kmm(X1, X2, _A=None, _sigma=1e1):
    ''' Reference:
    Implementation: Shift-Robust GNNs: Overcoming the Limitations of Localized Graph Training Data, NeurIPS21, https://arxiv.org/pdf/2108.01099.pdf
    '''

    if False:
        H = X1.matmul(X1.T)
        f = X1.matmul(X2.T)
        z = X2.matmul(X2.T)
    else:
        X1X1 = pairwise_distances(X1, X1)
        X1X2 = pairwise_distances(X1, X2)
        X2X2 = pairwise_distances(X2, X2)

        H = torch.exp(- 1e0 * X1X1) + torch.exp(- 1e-1 * X1X1) + torch.exp(- 1e-3 * X1X1)
        f = torch.exp(- 1e0 * X1X2) + torch.exp(- 1e-1 * X1X2) + torch.exp(- 1e-3 * X1X2)
        z = torch.exp(- 1e0 * X2X2) + torch.exp(- 1e-1 * X2X2) + torch.exp(- 1e-3 * X2X2)
        H /= 3
        f /= 3

    MMD_dist = H.mean() - 2 * f.mean() + z.mean()
    
    nsamples = X1.shape[0]
    f = - X1.shape[0] / X2.shape[0] * f.matmul(torch.ones((X2.shape[0],1)))
    G = - np.eye(nsamples)
    b = np.ones([_A.shape[0],1]) * 20
    h = - 0.2 * np.ones((nsamples,1))
    
    from cvxopt import matrix, solvers
    #return quadprog.solve_qp(H.numpy(), f.numpy(), qp_C, qp_b, meq)
    try:
        solvers.options['show_progress'] = False
        sol = solvers.qp(matrix(H.numpy().astype(np.double)), matrix(f.numpy().astype(np.double)), matrix(G), matrix(h), matrix(_A), matrix(b))
    except:
        import code
        code.interact(local=locals())

    return np.array(sol['x']), MMD_dist.item()
    
class KMeansDiff:
    def __init__(self, base_set, k=100):
        from sklearn.cluster import KMeans
        self.n_clusters = k

        st = time.time()
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=0).fit(base_set)
        cluster_rst = self.kmeans.labels_
        cluster_ceters = self.kmeans.cluster_centers_

        print(f"[Metric] take {time.time() - st:.3f} s to divide the base set to {self.n_clusters} clusters")

        self.cluster_sizes = [len(np.where(cluster_rst == cluster_id)[0]) for cluster_id in range(self.n_clusters)]
        self.cluster_sizes = self.cluster_sizes / np.linalg.norm(self.cluster_sizes)

    def _normed_size(self, target_set):
        _cluster_rst = self.kmeans.predict(target_set)
        _size = [len(np.where(_cluster_rst == cluster_id)[0]) for cluster_id in range(self.n_clusters)]
        return _size / np.linalg.norm(_size)

    def diff(self, target_set):
        _size_norm = self._normed_size(target_set)
        return np.linalg.norm(self.cluster_sizes - _size_norm)
    
    def diff2set(self, A, B):
        norm_size_A = self._normed_size(A)
        norm_size_B = self._normed_size(B)
        return np.linalg.norm(norm_size_A - norm_size_B)