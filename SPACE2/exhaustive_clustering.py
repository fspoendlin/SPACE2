import numpy as np
import numba as nb
import pandas as pd
from joblib import Parallel, delayed
from SPACE2.util import cluster_antibodies_by_CDR_length, rmsd, parse_antibodies, possible_combinations, reg_def, reg_def_CDR_all, reg_def_fw_all


@nb.njit(cache=True, fastmath=True)
def compare_CDRs_for_cluster(cluster, selection=reg_def_CDR_all, anchors=reg_def_fw_all):
    """ Used for exhaustive clustering.
    Computes the CDR rmsd between every pair of antibodies
    :param cluster: list of antibody tuples
    :return:
    """
    size = len(cluster)

    idx_1, idx_2 = possible_combinations(size)
    lindices = len(idx_1)
    
    rmsd_calculations = np.empty(lindices)
    for i in range(lindices):
        rmsd_calculations[i] = rmsd(cluster[idx_1[i]], cluster[idx_2[i]], selection=selection, anchors=anchors)
    
    return (idx_1, idx_2), rmsd_calculations


def get_distance_matrix(cluster, ids, selection=reg_def_CDR_all, anchors=reg_def_fw_all):
    """Get matrix with rmsd distances between CDRs of length matched antibodies.
    :param cluster: list of antibody tuples
    :return:
    """
    indices, distances = compare_CDRs_for_cluster(nb.typed.List(cluster), selection=selection, anchors=anchors)

    dist_mat = np.zeros((len(cluster), len(cluster)))
    for i, index in enumerate(zip(*indices)):
        dist_mat[index] = distances[i]
    
    dist_mat = dist_mat + dist_mat.T
    return (ids, dist_mat)


def get_distance_matrices(files, selection=reg_def["CDR_all"], anchors=reg_def["fw_all"], n_jobs=-1):
    """ Calculate CDR distance matrices between antibody pdb files.
    Antibodies are first clustered by CDR length and then an rmsd matrix is calculated for each cluster.

    :param files: list of antibody pdb files. These will be used to identify each antibody
    :param n_jobs: number of cpus to use when parallelising. (default is all)
    :return:
    """
    antibodies = parse_antibodies(files, n_jobs=n_jobs)
    cdr_clusters, cdr_cluster_ids = cluster_antibodies_by_CDR_length(antibodies, files, selection=selection)
    sorted_keys = sorted(cdr_cluster_ids, key=lambda k: len(cdr_cluster_ids[k]), reverse=True)
    
    # convert to single array for njit
    selection = np.concatenate(selection)
    anchors = np.concatenate(anchors)
    
    rmsd_matrices = Parallel(n_jobs=n_jobs)(
        delayed(get_distance_matrix)(cdr_clusters[key], cdr_cluster_ids[key], selection=selection, anchors=anchors) for key in sorted_keys)
    rmsd_matrices = {key: rmsd_matrices[i] for i, key in enumerate(sorted_keys)}

    return rmsd_matrices


def matrices_to_pandas_list(matrices_dict):
    """ Sort rmsd matrix dict into a pandas dataframe and list of distance matrices.

    :param rmsd_matrices: dictionary of tupes outputted from distance matrix calculation
    :return: pandas dataframe, list
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


def cluster_matrix(rmsd_matrix, clustering_method):
    """ Cluster a distance matrix using a clustering method

    :param rmsd_matrix: matrix of rmsd matrices between antibody CDRs
    :param clustering_method: clustering method
    :return: array of antibody cluster labels
    """
    if rmsd_matrix.shape == (1,1):
        return np.array([0])
    else:
        clustering_method.fit(rmsd_matrix)
        return clustering_method.labels_


def cluster_martices(matrices, clustering_method, n_jobs=-1):
    """ Cluster distances matrix using a clustering method

    :param rmsd_matrix: matrix of rmsd matrices between antibody CDRs
    :param clustering_method: clustering method
    :param n_jobs: number of cpus to use when parallelising. (default is all)
    :return: list of arrays of antibody cluster labels
    """
    cluster_labels = Parallel(n_jobs=n_jobs)(delayed(cluster_matrix)(matrix, clustering_method) for matrix in matrices)
    return cluster_labels


def get_clustering(df, clustering):
    """ Sort output into a pandas dataframe

    :param df: clustering metadata
    :param clustering: list of arrays of antibody cluster labels
    :return: pandas dataframe
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


def cluster_with_algorithm(method, files, selection=reg_def["CDR_all"], anchors=reg_def["fw_all"], n_jobs=-1):
    """ Sort a list of antibody pdb files into clusters.
    Antibodies are first clustered by CDR length and the by structural similarity

    :param method: initialised clustering method, the method should be consistent with sklearn API
    :param files: list of antibody pdb files. These will be used to identify each antibody
    :return: pandas dataframe with columns ID, cluster_by_length, cluster_by_rmsd, matrix_index
    """
    matrices_dict = get_distance_matrices(files, selection=selection, anchors=anchors, n_jobs=n_jobs)
    meta_data, rmsd_matrices = matrices_to_pandas_list(matrices_dict)
    cluster_labels = cluster_martices(rmsd_matrices, method, n_jobs=n_jobs)

    return get_clustering(meta_data, cluster_labels)
