import numpy as np
from joblib import Parallel, delayed
from SPACE2.util import (
    rmsd, dtw, parse_antibodies, cluster_antibodies_by_CDR_length, output_to_pandas, check_param,
    reg_def, reg_def_CDR_all, reg_def_fw_all, same_length_only
)

def greedy_cluster(cluster, d_funct, selection=reg_def_CDR_all, anchors=reg_def_fw_all, cutoff=1.25):
    """Greedy algorithm to sort antibodies in to structural clusters.

    :param cluster: list of tuples, antibodies
    :param d_funct: function, distance function
    :param selection: np.arrays, indices of residues for distance calculation
    :param anchors: np.arrays, indices of residues for structural alignment
    :param cutoff: float, distance threshold for clustering
    :return: dict, containing the index of antibodies in each cluster
    """
    size = len(cluster)

    out_clusters = dict()
    indices = np.arange(size)
    distances = np.zeros(size)

    while len(indices) > 0:
        for i in range(len(distances)):
            distances[i] = d_funct(cluster[indices[0]], cluster[indices[i]], selection=selection, anchors=anchors)
        ungrouped = np.array(distances) > cutoff
        out_clusters[indices[0]] = indices[~ungrouped]
        indices = indices[ungrouped]
        distances = distances[ungrouped]

    return out_clusters


def greedy_cluster_ids(cluster, ids, d_funct, selection=reg_def_CDR_all, anchors=reg_def_fw_all, cutoff=1.25):
    """Greedy algorithm to sort antibodies in to structural clusters.

    :param cluster: list of tuples, antibodies
    :param ids: list, antibody ids
    :param d_funct: function, distance function
    :param selection: np.arrays, indices of residues for distance calculation
    :param anchors: np.arrays, indices of residues for structural alignment
    :param cutoff: float, distance threshold for clustering
    :return: dict, containing the index of antibodies in each cluster
    """
    out_clusters = dict()

    clustered_indices = greedy_cluster(
        cluster, d_funct, selection=selection, anchors=anchors, cutoff=cutoff
    )
    for key in clustered_indices:
        out_clusters[ids[key]] = [ids[x] for x in clustered_indices[key]]

    return out_clusters


def greedy_clustering(files, selection=reg_def["CDR_all"], anchors=reg_def["fw_all"], cutoff=1.25,  d_metric='rmsd',
                      length_clustering='bins', length_tolerance=same_length_only, n_jobs=-1):
    """ Sort a list of antibody pdb files into clusters using greedy algorithm.
    Antibodies are first clustered by CDR length and then by structural similarity

    :param files: list, paths to pdb files of all antibodies to be clusterd
    :param selection: list of np.arrays, indices of residues used for structural similarity calculations.
                      Generally these are the indices of selected CDR regions. The indices should be formatted into 
                      a list with one np.array for each CDR region. (default is all 6 CDRs regions)
    :param anchors: list of np.arrays, indices of residues used for structural alignment of antibodies.
                    Generally these are the indices of framework regions. The indices should be formatted into 
                    a list with one np.array for each framework region. (default is heavy and light chain frameworks)
    :param cutoff: float, distance threshold for structural clustering in Angstroms. (default is 1.25A)
    :param d_metric: str, metric for structural distance calculation. Options are 'rmsd' or 'dtw'. (default is 'rmsd')
                     rmsd: root mean square deviation, dtw: dynamic time warping
    :param length_clustering: str, method for clustering antibodies by CDR length. Options are 'bins' or 'greedy'. (default is 'bins')
                              bins: CDRs are grouped into precalculated and equally spaced length bins
                              greedy: stochastic selection of cluster centers
    :param length_tolerance: np.array, binwidth for length clustering per CDR. (default clustering into bins of identical length)
                             array is required to have the same length as selection, with each element corresponding to the length 
                             tolerance of an individual CDR region.
    :param n_jobs: int, number of cpus to use for parallelising. (default is all)

    :return final_clustering: pd.DataFrame, containing the cluster assignments
    """
    check_param(length_tolerance, d_metric)

    antibodies = parse_antibodies(files, n_jobs=n_jobs)
    cdr_clusters, cdr_cluster_ids = cluster_antibodies_by_CDR_length(
        antibodies, files, selection=selection, clustering=length_clustering, tolerance=length_tolerance
        )
    # convert to single array for njit
    selection = np.concatenate(selection)
    anchors = np.concatenate(anchors)

    # select distance function
    if d_metric == 'rmsd':
        d_funct = rmsd
    elif d_metric == 'dtw':
        d_funct = dtw

    final_clustering = Parallel(n_jobs=n_jobs)(
        delayed(greedy_cluster_ids)(
            cdr_clusters[key], cdr_cluster_ids[key], d_funct, selection=selection, anchors=anchors, cutoff=cutoff
            ) for key in cdr_cluster_ids.keys())
    final_clustering = {key: final_clustering[i] for i, key in enumerate(cdr_clusters)}

    return output_to_pandas(final_clustering)
