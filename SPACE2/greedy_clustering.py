import numpy as np
from joblib import Parallel, delayed
from SPACE2.util import rmsd, parse_antibodies, cluster_antibodies_by_CDR_length, output_to_pandas, reg_def, reg_def_CDR_all, reg_def_fw_all


def greedy_cluster(cluster, selection=reg_def_CDR_all, anchors=reg_def_fw_all, cutoff=1.25):
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
            rmsds[i] = rmsd(cluster[indices[0]], cluster[indices[i]], selection=selection, anchors=anchors)
        ungrouped = np.array(rmsds) > cutoff
        out_clusters[indices[0]] = indices[~ungrouped]
        indices = indices[ungrouped]
        rmsds = rmsds[ungrouped]

    return out_clusters


def greedy_cluster_ids(cluster, ids, selection=reg_def_CDR_all, anchors=reg_def_fw_all, cutoff=1.25):
    """ Use greedy clustering to sort antibodies in to structurally similar groups.

    :param cluster: list of antibody tuples
    :param ids: list of unique ids for each antibody (often just filename)
    :param cutoff: cutoff rmsd from cluster center for antibody to be considered in the cluster
    :return: dictionary containing the ids of each cluster
    """
    out_clusters = dict()

    clustered_indices = greedy_cluster(cluster, selection=selection, anchors=anchors, cutoff=cutoff)
    for key in clustered_indices:
        out_clusters[ids[key]] = [ids[x] for x in clustered_indices[key]]

    return out_clusters


def greedy_clustering(files, selection=reg_def["CDR_all"], anchors=reg_def["fw_all"], cutoff=1.25, n_jobs=-1):
    """ Sort a list of antibody pdb files into clusters.
    Antibodies are first clustered by CDR length and the by structural similarity

    :param files: list of antibody pdb files. These will be used to identify each antibody
    :param cutoff: cutoff rmsd for structural clustering
    :param n_jobs: number of cpus to use when parallelising. (default is all)
    :return:
    """
    antibodies = parse_antibodies(files, n_jobs=n_jobs)
    cdr_clusters, cdr_cluster_ids = cluster_antibodies_by_CDR_length(antibodies, files, selection=selection)

    # convert to single array for njit
    selection = np.concatenate(selection)
    anchors = np.concatenate(anchors)

    final_clustering = Parallel(n_jobs=n_jobs)(
        delayed(greedy_cluster_ids)(cdr_clusters[key], cdr_cluster_ids[key], selection=selection, anchors=anchors, cutoff=cutoff) for key in cdr_cluster_ids.keys())
    final_clustering = {key: final_clustering[i] for i, key in enumerate(cdr_clusters)}

    return output_to_pandas(final_clustering)
