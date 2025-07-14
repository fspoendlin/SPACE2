import numpy as np
import numba as nb
import pandas as pd
from joblib import Parallel, delayed
from SPACE2.util import (
    cluster_antibodies_by_CDR_length, rmsd, dtw, parse_antibodies, possible_combinations, check_param,
    reg_def, reg_def_CDR_all, reg_def_fw_all, same_length_all_cdrs
)


@nb.njit(cache=True, fastmath=True)
def compare_CDRs_for_cluster(cluster, d_funct, selection=reg_def_CDR_all, anchors=reg_def_fw_all):
    """Computes the CDR structural similarity between pairs of antibodies

    :param cluster: list of tuples, antibodies
    :param d_funct: function, distance function
    :param selection: np.arrays, indices of residues for distance calculation
    :param anchors: np.arrays, indices of residues for structural alignment
    :return distances: np.array, structural distance between all antibody pairs
    :return indices: tuple, indices of antibodies in each pair
    """
    size = len(cluster)

    idx_1, idx_2 = possible_combinations(size)
    lindices = len(idx_1)
    
    dist_calculations = np.empty(lindices)
    for i in range(lindices):
        dist_calculations[i] = d_funct(cluster[idx_1[i]], cluster[idx_2[i]], selection=selection, anchors=anchors)
    
    return (idx_1, idx_2), dist_calculations


def get_distance_matrix(cluster, ids, d_funct, selection=reg_def_CDR_all, anchors=reg_def_fw_all):
    """Get matrix of CDR structural similarity of length clustered antibodies.

    :param cluster: list of tuples, antibodies
    :param ids: list, antibody ids
    :param d_funct: function, distance function
    :param selection: np.arrays, indices of residues for distance calculation
    :param anchors: np.arrays, indices of residues for structural alignment
    :return ids: list, antibody ids
    :return dist_mat: np.array, matrix of structural distances
    """
    indices, distances = compare_CDRs_for_cluster(nb.typed.List(cluster), d_funct, selection=selection, anchors=anchors)

    dist_mat = np.zeros((len(cluster), len(cluster)))
    for i, index in enumerate(zip(*indices)):
        dist_mat[index] = distances[i]
    
    dist_mat = dist_mat + dist_mat.T
    return (ids, dist_mat)


def get_distance_matrices(files, selection=reg_def["CDR_all"], anchors=reg_def["fw_all"], d_metric='rmsd',
                          length_clustering='bins', length_tolerance=same_length_all_cdrs, n_jobs=-1):
    """ Calculate CDR distance matrices between antibody pdb files.
    Antibodies are first clustered by CDR length and then a distance matrix is calculated for each cluster.

    :param files: list, paths to pdb files of all antibodies
    :param selection: list of np.arrays, indices of residues used for structural similarity calculations.
    :param anchors: list of np.arrays, indices of residues used for structural alignment of antibodies.
    :param d_metric: str, metric for structural distance calculation. Options are 'rmsd' or 'dtw'.
    :param length_clustering: str, method for clustering antibodies by CDR length. Options are 'bins' or 'greedy'.
    :param length_tolerance: np.array, binwidth for length clustering per CDR.
    :param n_jobs: number of cpus to use when parallelising. (default is all)
    :return dist_matrices: dict, key: length cluster index, value: distance matrix
    """
    check_param(length_tolerance, d_metric) # is combination of length tolerance and d_metric valid?

    antibodies = parse_antibodies(files, n_jobs=n_jobs)
    cdr_clusters, cdr_cluster_ids = cluster_antibodies_by_CDR_length(
        antibodies, files, selection=selection, clustering=length_clustering, tolerance=length_tolerance
        )
    sorted_keys = sorted(cdr_cluster_ids, key=lambda k: len(cdr_cluster_ids[k]), reverse=True)
    
    # convert to single array for njit
    selection = np.concatenate(selection)
    anchors = np.concatenate(anchors)

    # select distance function
    if d_metric == 'rmsd':
        d_funct = rmsd
    elif d_metric == 'dtw':
        d_funct = dtw
    
    dist_matrices = Parallel(n_jobs=n_jobs)(
        delayed(get_distance_matrix)(
            cdr_clusters[key], cdr_cluster_ids[key], d_funct, selection=selection, anchors=anchors
            ) for key in sorted_keys)
    dist_matrices = {key: dist_matrices[i] for i, key in enumerate(sorted_keys)}

    return dist_matrices


def matrices_to_pandas_list(matrices_dict):
    """ Sort distance matrix dict into a pandas dataframe and list of distance matrices.

    :param matrices_dict: dict
    :return: pd.DataFrame, list
    """
    df = pd.DataFrame()
    cluster_by_length = []
    ids = []
    size = []
    matrix_index = []
    matrices = []

    for i, key in enumerate(matrices_dict):
        cluster_by_length.append(key)
        ids.append(matrices_dict[key][0])
        size.append(len(matrices_dict[key][0]))
        matrix_index.append(i)
        matrices.append(matrices_dict[key][1])
        
    df['cluster_by_length'] = cluster_by_length
    df['IDs'] = ids
    df['cluster_size'] = size
    df['matrix_index'] = matrix_index

    return df, matrices


def cluster_matrix(distance_matrix, clustering_method):
    """ Cluster a distance matrix using a clustering method

    :param distance_matrix: np.array
    :param clustering_method: class, clustering method
    :return: np.array, antibody cluster labels
    """
    if distance_matrix.shape == (1,1):
        return np.array([0])
    else:
        clustering_method.fit(distance_matrix)
        return clustering_method.labels_


def cluster_martices(distance_matrices, clustering_method, n_jobs=-1):
    """ Cluster distances matrix using a clustering method

    :param distance_matrices: np.array
    :param clustering_method: class, clustering method
    :param n_jobs: float, number of cpus for parallelisation
    :return: list of arrays, antibody cluster labels
    """
    cluster_labels = Parallel(n_jobs=n_jobs)(delayed(cluster_matrix)(matrix, clustering_method) for matrix in distance_matrices)
    return cluster_labels


def get_clustering(df, clustering):
    """ Sort output into a pandas dataframe

    :param df: pd.DataFrame, clustering metadata
    :param clustering: list of arrays, antibody cluster labels
    :return: pd.DataFrame
    """
    length = []
    cluster = []
    ids = []

    for i, length_cluster in df.iterrows():
        for n, id in enumerate(length_cluster.IDs):
            ids.append(id)
            cluster.append(np.array(length_cluster.IDs)[clustering[i] == clustering[i][n]][0])
            length.append(length_cluster.cluster_by_length)

    return pd.DataFrame({'ID': ids, 'cluster_by_length': length, 'cluster_by_rmsd': cluster})


def cluster_with_algorithm(method, files, selection=reg_def["CDR_all"], anchors=reg_def["fw_all"], 
                           d_metric='rmsd', length_clustering='bins', length_tolerance='same_length_only', n_jobs=-1):
    """ Sort a list of antibody pdb files into clusters.
    Antibodies are first clustered by CDR length and the by structural similarity

    :param method: Class, initialised clustering method, should be consistent with sklearn API
    :param files: list, paths to pdb files of all antibodies to be clusterd
    :param selection: list of np.arrays, indices of residues used for structural similarity calculations.
    :param anchors: list of np.arrays, indices of residues used for structural alignment of antibodies.
    :param d_metric: str, metric for structural distance calculation. Options are 'rmsd' or 'dtw'.
    :param length_clustering: str, method for clustering antibodies by CDR length. Options are 'bins' or 'greedy'.
    :param length_tolerance: str or np.array: binwidth for length clustering per CDR.
    :param n_jobs: int, number of cpus to use for parallelising.
    :return: pd.DataFrame, clustering output with columns ID, cluster_by_length, cluster_by_rmsd, matrix_index
    """
    if isinstance(length_tolerance, str):
        if length_tolerance == 'same_length_only':
            n = len(selection)
            length_tolerance = np.ones(n)

    matrices_dict = get_distance_matrices(
        files, selection=selection, anchors=anchors, d_metric=d_metric, length_clustering=length_clustering,
        length_tolerance=length_tolerance, n_jobs=n_jobs
        )
    meta_data, rmsd_matrices = matrices_to_pandas_list(matrices_dict)
    cluster_labels = cluster_martices(rmsd_matrices, method, n_jobs=n_jobs)

    return get_clustering(meta_data, cluster_labels)
