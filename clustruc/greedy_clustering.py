import numpy as np
from joblib import Parallel, delayed
from clustruc.util import get_CDR_lengths, possible_combinations, rmsd, parse_antibodies


def cluster_antibodies_by_CDR_length(antibodies, ids):
    """ Sort a list of antibody tuples into groups with the same CDR lengths

    :param antibodies: list of antibody tuples
    :param ids: list of ids for each antibody
    :return:
    """
    clusters = dict()
    names = dict()

    for i, antibody in enumerate(antibodies):
        cdr_l = "_".join(get_CDR_lengths(antibody))
        if cdr_l not in clusters:
            clusters[cdr_l] = [antibody]
            names[cdr_l] = [ids[i]]
        else:
            clusters[cdr_l].append(antibody)
            names[cdr_l].append(ids[i])
    return clusters, names


def compare_CDRs_for_cluster(cluster, n_jobs=-1):
    """ Used for exhaustive clustering.
    Computes the CDR rmsd between every pair of antibodies

    :param cluster: list of antibody tuples
    :param n_jobs: number of cpus to use for parallelising. (default is all)
    :return:
    """

    size = len(cluster)

    indices = possible_combinations(size)
    rmsd_calculations = Parallel(n_jobs=n_jobs)(delayed(rmsd)(cluster[i], cluster[j]) for i, j in zip(*indices))

    return indices, rmsd_calculations


def greedy_cluster(cluster, cutoff=1.0):
    """ Use greedy clustering to sort antibodies in to structurally similar groups.

    :param cluster: list of antibody tuples
    :param cutoff: cutoff rmsd from cluster center for antibody to be considered in the cluster
    :return: dictionary containing the index of antibodies belonging to each cluster
    """
    size = len(cluster)

    out_clusters = dict()
    indices = np.arange(size)
    rmsds = np.zeros(size)

    while len(indices) > 0:
        for i in range(len(rmsds)):
            rmsds[i] = rmsd(cluster[indices[0]], cluster[indices[i]])
        ungrouped = np.array(rmsds) > cutoff
        out_clusters[indices[0]] = indices[~ungrouped]
        indices = indices[ungrouped]
        rmsds = rmsds[ungrouped]

    return out_clusters


def greedy_cluster_ids(cluster, ids, cutoff=1.0):
    """ Use greedy clustering to sort antibodies in to structurally similar groups.

    :param cluster: list of antibody tuples
    :param ids: list of unique ids for each antibody (often just filename)
    :param cutoff: cutoff rmsd from cluster center for antibody to be considered in the cluster
    :return: dictionary containing the ids of each cluster
    """
    out_clusters = dict()

    clustered_indices = greedy_cluster(cluster, cutoff=cutoff)
    for key in clustered_indices:
        out_clusters[ids[key]] = [ids[x] for x in clustered_indices[key]]

    return out_clusters


def cluster_by_rmsd(files, cutoff=1.0, n_jobs=-1):
    """ Sort a list of antibody pdb files into clusters.
    Antibodies are first clustered by CDR length and the by structural similarity

    :param files: list of antibody pdb files. These will be used to identify each antibody
    :param cutoff: cutoff rmsd for structural clustering
    :param n_jobs: number of cpus to use when parallelising. (default is all)
    :return:
    """
    antibodies = parse_antibodies(files, n_jobs=n_jobs)
    cdr_clusters, cdr_cluster_ids = cluster_antibodies_by_CDR_length(antibodies, files)

    final_clustering = Parallel(n_jobs=n_jobs)(
        delayed(greedy_cluster_ids)(cdr_clusters[key], cdr_cluster_ids[key], cutoff) for key in cdr_cluster_ids.keys())
    final_clustering = {key: final_clustering[i] for i, key in enumerate(cdr_clusters)}

    return final_clustering
