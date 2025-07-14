from sklearn.cluster import AgglomerativeClustering
from SPACE2.exhaustive_clustering import cluster_with_algorithm
from SPACE2.util import reg_def


def agglomerative_clustering(files, selection=reg_def["CDR_all"], anchors=reg_def["fw_all"], cutoff=1.25, 
                             d_metric='rmsd', length_clustering='bins', length_tolerance='same_length_only',
                             n_jobs=-1):
    """ Sort a list of antibody pdb files into clusters using agglomerative algorithm.
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
    :param length_tolerance: str or np.array, binwidth for length clustering per CDR. (default clustering into bins of identical length)
                             array is required to have the same length as selection, with each element corresponding to the length 
                             tolerance of an individual CDR region.
    :param n_jobs: int, number of cpus to use for parallelising. (default is all)

    :return final_clustering: pd.DataFrame, containing the cluster assignments
    """

    clustering_algorithm = AgglomerativeClustering(n_clusters=None, metric='precomputed', distance_threshold=cutoff, linkage='complete')
    final_clustering = cluster_with_algorithm(
        clustering_algorithm, files, selection=selection, anchors=anchors, d_metric=d_metric,
        length_clustering=length_clustering, length_tolerance=length_tolerance, n_jobs=n_jobs
        )
    
    return final_clustering
