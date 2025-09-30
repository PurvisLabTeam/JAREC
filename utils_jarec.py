import numpy as np 
import warnings
import pandas as pd
#import cupy as cp
from sklearn.decomposition import PCA
import random
import multiprocessing
from joblib import Parallel, delayed
from sklearn.metrics import mean_squared_error
from scipy.stats import vonmises
from scipy.spatial.transform import Rotation
import scipy as sc
import scipy.stats as stats 
import math
import numpy.matlib
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler


def pca_new(X, L=None):
    """
    Principal Component Analysis for input data set.

    Parameters:
    - X: a NxD matrix which contains one D dimension data in each row.
    - L: the dimension to which the dataset will be reduced (default is min(N,D)).

    Returns:
    - U: a DxL matrix which stores principle components in columns.
    - lambda: a vector containing min(N,D) eigenvalues.
    - xc: the centroid of input data set.
    """
    N, D = X.shape  # N data points with dimension D.

    if L is None:
        L = min(N, D)

    xc = np.mean(X, axis=0)  # Obtain the centroid of the original dataset.
    X = X - np.tile(xc, (N, 1))  # Zero-mean data set.
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        if D <= N:
            S = np.matmul(X.T, X)  # DxD Covariance matrix.
        else:
            S = np.matmul(X, X.T)  # NxN Gram matrix.


    # Assuming S is a NumPy array
    eigvalues, eigvectors = np.linalg.eig(S)

    # Extract the eigenvalues
    eigvalues, eigvectors = np.linalg.eig(S)

    # Sort the eigenvalues in descending order and get the corresponding indices
    sorted_indices = np.argsort(eigvalues)[::-1]
    lambda_ = eigvalues[sorted_indices]

    # Extract the corresponding eigenvectors
    U = eigvectors[:, sorted_indices[:L]]


# Note: lambda is a reserved keyword in Python, so I've used lambda_values instead

    if D > N:
        lv = lambda_[:L]
        z = np.where(lv > 0)
        lv[z] = 1.0 / np.sqrt(lv[z])
        U = np.matmul(X.T, U)
        U = np.matmul(U, np.diag(lv))

    lambda_ = lambda_ / N

    xc = np.array(xc)
    xc = xc.reshape(-1, 1)
    xc = xc.T

    return U, lambda_, xc

# Example usage:
# U, lambda_, xc = pca_new(X, L)

def Spherelets(X, d):
    # Find the best d-dimensional sphere to fit data X

    # Input: X = data matrix
    # Output: c = center of the spherelet
    #         r = radius of the spherelet
    #         V = the subspace where the sphere lies in the affine space c+V

    n, m = X.shape  # n = sample size


    if n > d + 1:  # If there are enough samples, fit the data by a sphere or a hyperplane

        # Do d+1 dimensional PCA first
        V, lambd, mu = pca_new(X, d+1)

        Y = np.ones((n, 1)) @ mu + (X - np.ones((n, 1)) @ mu) @ V @ V.T
        Y = np.array(Y)

        l = np.zeros((n,1))

        for i in range(n):
            l[i] = np.linalg.norm(Y[i, :]) ** 2

        lbar = np.mean(l)
        H = np.zeros((m, m))
        f = np.zeros((m,1))

        for i in range(n):
            H += np.matmul((mu - Y[i, :]).T, (mu - Y[i, :]))
            f += (l[i] - lbar) * (mu - Y[i, :]).T 

        H_inv = np.linalg.pinv(H)

        # Calculate the center c of the sphere
        c = mu.T + np.matmul(np.matmul(V, V.T), (-0.5 * np.matmul(H_inv, f) - mu.T))
        
        Riemd = np.zeros((n, 1))  # Initialize an array to store distances

        for i in range(n):

            distance =  np.sqrt(np.matmul((c.T - Y[i, :]),(c.T - Y[i, :]).T))
            Riemd[i] = distance  # Store the distance in the array

        # Calculate the mean of the distances
        r = np.mean(Riemd)
    else:
        c = None
        r = None
        V = None
    
    return c, r, V


# Example usage:
# c, r, V = Spherelets(X, d)


def Proj(X, r, c, V):
    ## X is n x p
    ## r is 1x1 
    ## c is p x 1
    ## V is p x (d+1)


    num = ((X - c.T) @ V)
    denom = np.linalg.norm(num, axis = 1) #n x 1
    denom = denom[:, np.newaxis]
    # Divide by rown in denom
    res = r * num / denom

    #return dist
    return res

def ProjHyp(X, r, c, V):
    ## X is n x p
    ## r is 1x1 
    ## c is p x 1
    ## V is p x (d+1)

    ## returns projection of X into new space, but now the projection is n x p dimensions
    unos = np.ones((len(X), 1))
    center_ones = unos @ c.T

    num = ((X - (unos @ c.T )) @ (V @ V.T)) 
    denom = np.linalg.norm(num, axis = 1)
    denom = denom[:, np.newaxis]
    projd = center_ones + (r * (num / denom))

    #return dist
    return np.array(projd)


def HypTestBoot(X_big_1, X_big_2, d, gamma=1):
    
    # Assuming Spherelets and Proj functions are defined elsewhere
    X_1 = X_big_1.sample(n=len(X_big_1), replace=True)
    X_2 = X_big_2.sample(n=len(X_big_2), replace=True)


    center_SPCA_1 , radius_SPCA_1, output_SPCA_intrinsic_1 = Spherelets(X_1, d)
    center_SPCA_2, radius_SPCA_2, output_SPCA_intrinsic_2 = Spherelets(X_2, d)

    projection_1_onto_2 = np.linalg.norm(np.array(X_1) - ProjHyp(X_1, radius_SPCA_2, center_SPCA_2, output_SPCA_intrinsic_2), ord=2) / len(X_1)
    projection_2_onto_1 = np.linalg.norm(np.array(X_2) - ProjHyp(X_2, radius_SPCA_1, center_SPCA_1, output_SPCA_intrinsic_1), ord=2) / len(X_2)

    d_test =  ((projection_1_onto_2**2) + (projection_2_onto_1**2))

    X_tot = pd.concat([X_1, X_2])

    center_big, radius_big, V_big = Spherelets(X_tot, d)
    projection_big_1 = np.linalg.norm(np.array(X_1) - ProjHyp(np.array(X_1), radius_big, center_big, V_big), ord=2) / len(X_1)
    projection_big_2 = np.linalg.norm(np.array(X_2) - ProjHyp(np.array(X_2), radius_big, center_big, V_big), ord=2) / len(X_2)
    d_null = (projection_big_1**2) + (projection_big_2**2)
    ss_ratio = ss_test(X_1, X_2, d=2)

    
    return d_null, d_test, ss_ratio

def HypTestBootAll(X_1, X_2, B, d, multi, gamma=1):
    def worker(b):
        return HypTestBoot(X_1, X_2, gamma=gamma, d=d)

    if multi==True:
        num_cores = multiprocessing.cpu_count()  # Get the number of CPU cores
        results = Parallel(n_jobs=num_cores-1)(delayed(worker)(b) for b in tqdm(range(B), desc='Bootstrap Distance', leave=False, disable=True))
        d_null = [result[0] for result in results]
        d_test = [result[1] for result in results]
        ss_ratio = [result[2] for result in results]
    else: 
        res = 0
        d_null = [None] * B
        d_test = [None] * B
        ss_ratio = [None] * B
        for b in range(B):
            d_null_single, d_test_single, ss_single = HypTestBoot(X_1, X_2, gamma=gamma, d=d)
            d_null[b] = d_null_single
            d_test[b] = d_test_single
            ss_ratio[b] = ss_single
    return np.array(d_null), np.array(d_test), np.array(ss_ratio)


def AddGaussianNoise(df, col, level):
    noise = np.random.normal(0, level * df[col].std(), len(df))
    df[col] = df[col] + noise
    return df


def FeatImp(X_1, X_2, gamma, B, diffs, multi):

    #Input: X_1, X_2 are datasets in their original space.
    # gamma is the scaling factor from training data for the test between X_1 and X_2
    # B is number of boostrap replicates
    # subsamp is subsample size to take from each bootstrap 
    # result is the existing hypothesis test result from X_1 and X_2
    features = X_1.columns
    feat_imp_list = [None] * len(features)
    f=0
    for feat in features:
        X_1_noise = X_1.copy()
        X_2_noise = X_2.copy()

        X_1_noise = AddGaussianNoise(X_1_noise.copy(), feat, 0.05)
        X_2_noise = AddGaussianNoise(X_2_noise.copy(), feat, 0.05)

        out = HypTestBootAll(X_1_noise, X_2_noise, gamma, B, multi=multi)
        newresult = out[0] 
        newdiffs = out[1:]

        feat_imp_list[f] = np.mean(newdiffs) - np.mean(diffs)
        f += 1
    feat_imp_list = feat_imp_list / max(feat_imp_list)
    feat_results_df = pd.DataFrame(
        {'Features': features,
         'varImp': feat_imp_list}
    )
    feat_results_df = feat_results_df.sort_values(by=['varImp'], ascending=True)

    return feat_results_df

def mse_to_sphere(X, c, r, V):
    # Input: X = n x d cells projected into the lower dimensional space
    #         c = center of the spherelet
    #         r = radius of the spherelet
    #         V = the subspace where the sphere lies in the affine space c+V
    # Output: MSE of points on a sphere
    if isinstance(c, np.ndarray) and len(c.shape) == 1:
        c = c.reshape(-1, 1)

    Xhat = ProjHyp(X, r, c, V)


    mse = np.sum((np.array(X) - Xhat)**2) / len(X)
    

    return mse

def ss_to_sphere(X, c, r, V):
    # Input: X = n x d cells projected into the lower dimensional space
    #         c = center of the spherelet
    #         r = radius of the spherelet
    #         V = the subspace where the sphere lies in the affine space c+V
    # Output: MSE of points on a sphere
    if isinstance(c, np.ndarray) and len(c.shape) == 1:
        c = c.reshape(-1, 1)

    Xhat = ProjHyp(X, r, c, V)

    #mse = np.linalg.norm((X - Xhat))**2 / len(X)
    #mse = mean_squared_error(X, Xhat)

    mse = sum(np.linalg.norm(X-Xhat, axis=1)**2)
    

    return mse

import numpy as np

def generate_data_on_sphere(V, c, r, D, d, n, seed=1453):
    """
    Generate random points on a hypersphere.
    
    Parameters:
    - V: Subspace where the sphere lies in the affine space c+V
    - c: Center of the spherelet
    - r: Radius of the spherelet
    - D: Ambient dimension
    - d: Dimension of the sphere
    - n: Number of points to generate
    - seed: Random seed for reproducibility
    
    Returns:
    - X: Generated points on the sphere
    """
    
    
    if isinstance(c, np.ndarray) and len(c.shape) == 1:
        c = c.reshape(-1, 1)
    
    # Generate points uniformly on the hypersphere
    Y = rand_uniform_hypersphere(n, d+1, seed=seed)
    
    # Generate Gaussian noise
    eps = np.random.normal(0, 0.01, size=(n, D))
    
    # Transform points to the affine space c + V
    X = c.T + (r * (np.matmul(V, Y.T)).T)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X = X + eps

    return X

def rand_uniform_hypersphere(n, d, seed=1453):
    """
    Generate n points uniformly on the surface of a d-dimensional unit sphere.
    
    Parameters:
    - n: Number of points
    - d: Dimension of the sphere
    - seed: Random seed for reproducibility
    
    Returns:
    - points: Generated points on the hypersphere
    """
    rng = np.random.default_rng(seed)
    # Generate random Gaussian vectors
    gaussian_vectors = rng.normal(size=(n, d))
    # Normalize each vector to lie on the surface of the sphere
    points = gaussian_vectors / np.linalg.norm(gaussian_vectors, axis=1)[:, np.newaxis]
    
    return points



def shift_params(V, c, r, V_degrees, c_shift, r_shift):
    r_new = r + r_shift
    c_new = c + c_shift
    

    # Create a rotation matrix for the specified angle
    # Define your rotation angle in degrees

    # Convert the rotation angle from degrees to radians
    rotation_angle_radians = math.radians(V_degrees)

    # Create the 2D rotation matrix
    rotation_matrix = np.array([[math.cos(rotation_angle_radians), -math.sin(rotation_angle_radians)],
                                [math.sin(rotation_angle_radians), math.cos(rotation_angle_radians)]])

    # Create the 10x10 identity matrix
    identity_matrix = np.eye(len(V))

    # Insert the 2x2 rotation matrix into the 10x10 identity matrix
    identity_matrix[:2, :2] = rotation_matrix

    # Multiply the extended rotation matrix by the column vector
    V_new = np.dot(identity_matrix, V)


    return V_new, c_new, r_new


def hausdorff_distance(V_1, c_1, r_1, V_2, c_2, r_2):
    d_h = np.linalg.norm(np.eye(len(V_1)) - (V_1 @ V_1.T)) + np.linalg.norm((V_1 @ V_1.T)) * np.linalg.norm(c_1 - c_2) + abs(r_1 - r_2) + r_1 * np.linalg.norm(V_1 - V_2)
    return d_h

def ss_test(X_1, X_2, d):

    c_1, r_1, V_1 = Spherelets(X_1, d)
    c_2, r_2, V_2 = Spherelets(X_2, d)

    X_1hat = ProjHyp(X_1, r_1, c_1, V_1)
    X_2hat = ProjHyp(X_2, r_2, c_2, V_2)

    SS_1 = np.sum((np.array(X_1) - X_1hat)**2)
    SS_2 = np.sum((np.array(X_2) - X_2hat)**2)

    X_union = pd.concat([X_1, X_2])
    c_union, r_union, V_union = Spherelets(X_union, d)
    X_unionhat = ProjHyp(X_union, r_union, c_union, V_union)
    SS_union = np.sum((np.array(X_union) - X_unionhat)**2)


    ss_stat = (SS_1 + SS_2) / SS_union
    

    return ss_stat

def get_tuned_pval(d_metric_0, d_metric_1, gamma):
    result = stats.ttest_ind(d_metric_0, d_metric_1, equal_var= False)
    cdf_value = stats.t.cdf(abs(result.statistic * gamma), result.df)

    adj_pval = 2 * (1 - cdf_value) 

    return adj_pval

def tuning_gamma(x_batch1, x_batch2, confidence_level):
    x_batch1 = pd.DataFrame(x_batch1)
    x_batch2 = pd.DataFrame(x_batch2)
    d_null_tune, d_test_tune, ss_ratio = HypTestBootAll(x_batch1, x_batch2, B=1000, gamma =1, d = 2, multi=True)
    ttest = stats.ttest_ind(d_test_tune, d_null_tune, equal_var= False)
    dof = ttest.df
    test_stat = ttest.statistic
    cdf_value = stats.t.cdf(abs(test_stat), dof)

    p_value_two_tailed = 2 * (1 - cdf_value)
    p_value_two_tailed

    lvl = stats.t.ppf(confidence_level/2, dof)
    gamma = lvl / test_stat

    return gamma



