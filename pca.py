import numpy as np


def pca(data, nr=None):
    """
input: datamatrix as loaded by numpy.loadtxt('dataset.txt')

optional param nr: the first N eigenvalues (and eigenvectors) to return. Else we return all

output:  

    1) the eigenvalues in a vector (numpy array) in descending order

    2) the unit eigenvectors in a matrix (numpy array) with each column being \
    an eigenvector (in the same order as its associated eigenvalue)

note: make sure the order of the eigenvalues (the projected variance) is \
decreasing, and the eigenvectors have the same order as their associated \
eigenvalues

the column eig_values_sorted_reverse[:,i] is the eigenvector corresponding to \
the eigenvalue eig_vectors_sorted_reverse[i].
    """
    cov_mat = np.cov(
        data.T)  
    eigenvalues, eigenvectors = np.linalg.eig(cov_mat)
    eig_pairs = [(np.abs(eigenvalues[i]), eigenvectors[:, i])
                 for i in range(len(eigenvalues))]
    eig_pairs_sorted = sorted(eig_pairs, key=lambda tup: tup[0])
    eig_values_sorted_reverse = []
    eig_vectors_sorted_reverse = []
    eig_pairs_sorted.reverse()
    for tup in eig_pairs_sorted:
        eig_values_sorted_reverse.append(tup[0])
        eig_vectors_sorted_reverse.append(tup[1])
    eig_values_sorted_reverse = np.array(eig_values_sorted_reverse)
    eig_vectors_sorted_reverse = np.array(eig_vectors_sorted_reverse)
    if nr is None:
        return eig_values_sorted_reverse, eig_vectors_sorted_reverse
    else:
        return eig_values_sorted_reverse[:nr], eig_vectors_sorted_reverse[:nr]
