from sklearn.cluster import AgglomerativeClustering
from SPACE2.exhaustive_clustering import cluster_with_algorithm


def agglomerative_clustering(files, cutoff=1.0, n_jobs=-1):
    """ Sort a list of antibody pdb files into clusters.
    Antibodies are first clustered by CDR length and the by structural similarity

    :param files: list of antibody pdb files. These will be used to identify each antibody
    :param cutoff: cutoff rmsd for structural clustering
    :param n_jobs: number of cpus to use when parallelising. (default is all)
    :return:
    """
    clustering_algorithm = AgglomerativeClustering(n_clusters=None, metric='precomputed', distance_threshold=cutoff, linkage='complete')
    final_clustering = cluster_with_algorithm(clustering_algorithm, files)
    
    return final_clustering