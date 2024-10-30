import numpy as np
from numpy.linalg import norm
import numba as nb
import pandas as pd
from joblib import Parallel, delayed

# All are north region definitions in imgt numbering
CDRs = ["CDRH1", "CDRH2", "CDRH3", "CDRL1", "CDRL2", "CDRL3"]
fw = ["fwH", "fwL"]
reg_def = dict()
reg_def["fwH"] = [6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 41, 42, 43, 44, 45, 46, 47, 48, 49,
                  50, 51, 52, 53, 54, 67, 68, 69, 70, 71, 72, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87,
                  88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 118, 119, 120, 121, 122, 123]
reg_def["fwL"] = [134, 135, 136, 137, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 169, 170, 171,
                  172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 198, 199, 200, 202, 203, 204, 205, 206, 207,
                  208, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229,
                  230, 231, 232, 246, 247, 248, 249]
reg_def["CDRH1"] = [24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]
reg_def["CDRH2"] = [55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66]
reg_def["CDRH3"] = [105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117]
reg_def["CDRL1"] = [152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168]
reg_def["CDRL2"] = [183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197]
reg_def["CDRL3"] = [233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245]

# numba does not like lists very much
reg_def = {x: np.array(reg_def[x]) for x in reg_def}

reg_def["CDR_H_all"] = [reg_def[CDR] for CDR in ["CDRH1", "CDRH2", "CDRH3"]]
reg_def["CDR_L_all"] = [reg_def[CDR] for CDR in ["CDRL1", "CDRL2", "CDRL3"]]
reg_def["CDR_all"] = reg_def["CDR_H_all"] + reg_def["CDR_L_all"]
reg_def["fw_all"] = [reg_def[fw] for fw in ["fwH", "fwL"]]

# Store these to use in jit
reg_def_CDR_all = np.concatenate(reg_def["CDR_all"])
reg_def_fw_all = np.concatenate(reg_def["fw_all"])

same_length_only = np.array([1, 1, 1, 1, 1, 1])


def random_rot():
    """ Just a random rotation

    :return: a random rotation
    """
    rot = np.random.rand(3)

    # Normalize quaternions
    norm = np.sqrt(1 + np.square(rot).sum(axis=-1))
    b, c, d = (rot / norm)
    a = 1 / norm

    # Make rotation matrix from quaternions
    R = (
        (a ** 2 + b ** 2 - c ** 2 - d ** 2), (2 * b * c - 2 * a * d), (2 * b * d + 2 * a * c),
        (2 * b * c + 2 * a * d), (a ** 2 - b ** 2 + c ** 2 - d ** 2), (2 * c * d - 2 * a * b),
        (2 * b * d - 2 * a * c), (2 * c * d + 2 * a * b), (a ** 2 - b ** 2 - c ** 2 + d ** 2)
    )

    return np.array(R).reshape((3, 3))


def get_antibody(text):
    lines = [x for x in text.split("\n") if x[13:15] == "CA"]
    size = len(lines)
    numbers = np.empty(size, dtype=int)
    coords = np.empty((size, 3))

    for i in range(size):
        line = lines[i]
        assert (line[21] == "H") or (line[21] == "L"), "Chains must be labelled H for heavy and L for light" 
        chain_term = 128 if line[21] == "L" else 0
        number = int(line[22:26])
        if number <= 128:
            numbers[i] = number + chain_term
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            coords[i] = (x, y, z)
        else:
            numbers[i] = -1

    return numbers[numbers!=-1], coords[numbers!=-1]


def parse_antibody(file):
    with open(file) as f:
        txt = f.read()
    return get_antibody(txt)


def parse_antibodies(files, n_jobs=20):
    return Parallel(n_jobs=n_jobs)(delayed(parse_antibody)(file) for file in files)


@nb.njit(cache=True, fastmath=True)
def get_residues(antibody, selection):
    numbers, coords = antibody

    ids = np.zeros(len(numbers), dtype=nb.int32)
    for i, n in enumerate(numbers):
        if n in selection:
            ids[i] = 1

    select = np.zeros((sum(ids), 3))
    count = 0
    for i, val in enumerate(ids):
        if val == 1:
            select[count] = coords[i]
            count = count + 1

    return select


@nb.njit(cache=True, fastmath=True)
def remove_insertions(ab):
    nums = ab[0]
    l_ab = len(nums)
    mask = np.ones(l_ab, np.int64)
    for i in range(1,l_ab):
        if nums[i] == nums[i-1]:
            mask[i] = 0
            
    indices = np.empty(sum(mask), np.int64)
    count = 0
    
    for i in range(l_ab):
        if mask[i] == 1:
            indices[count] = i
            count = count + 1
    return nums[indices], ab[1][indices]


def get_CDR_lengths(antibody, selection=reg_def_CDR_all):
    return [len(get_residues(antibody, CDR)) for CDR in selection]


@nb.njit(cache=True, fastmath=True)
def possible_combinations(size):
    out1 = np.zeros(size * (size - 1) // 2, dtype=nb.int32)
    out2 = np.zeros(size * (size - 1) // 2, dtype=nb.int32)
    count = 0
    for i in range(size):
        for j in range(size):
            if i > j:
                out1[count], out2[count] = i, j
                count = count + 1

    return out1, out2


@nb.njit(cache=True, fastmath=True)
def get_alignment_transform(fixed, moveable, anchors):
    fixed, moveable = remove_insertions(fixed), remove_insertions(moveable)
    anchors = np.intersect1d(np.intersect1d(anchors, fixed[0]), moveable[0])
  
    anch1 = get_residues(moveable, anchors)
    anch2 = get_residues(fixed, anchors)
    
    n_residues = anch1.shape[0]
    
    anch1_center = anch1.sum(0) / n_residues
    anch2_center = anch2.sum(0) / n_residues
    
    anch1 = anch1-anch1_center
    anch2 = anch2-anch2_center

    V, _, W = np.linalg.svd(anch1.T @ anch2)
    U = V @ W

    if np.linalg.det(U) < 0:
        U = (np.array([[1, 1, -1]]) * V) @ W

    return anch1_center, anch2_center, U


@nb.njit(cache=True, fastmath=True)
def align(fixed, moveable, anchors):
    x, y, U = get_alignment_transform(fixed, moveable, anchors)

    return moveable[0], ((moveable[1] - x) @ U + y)


@nb.njit(cache=True, fastmath=True)
def rmsd(ab1, ab2, selection=reg_def_CDR_all, anchors=reg_def_fw_all):
    ''' Root Mean Square Deviation between two antibodies'''
    residues1 = get_residues(ab1, selection)
    residues2 = get_residues(align(ab1, ab2, anchors), selection)
    l = len(residues1)

    total = 0
    for i in range(l):
        total = total + sum((residues1[i] - residues2[i]) ** 2)

    return np.sqrt(total / l)


@nb.njit(cache=True, fastmath=True)
def dtw(ab1, ab2, selection=reg_def_CDR_all, anchors=reg_def_fw_all):
    ''' Dynamic Time Warping distance between two antibodies'''
    residues1 = get_residues(ab1, selection)
    residues2 = get_residues(align(ab1, ab2, anchors), selection)
    l1, l2 = residues1.shape[0], residues2.shape[0]
    dtw_matrix = np.empty((l1,l2))

    dtw_matrix[0][0] = norm(residues1[0]-residues2[0])**2
    for i in range(1, l1):
        dtw_matrix[i][0] = dtw_matrix[i - 1][0] + norm(residues1[i] - residues2[0])**2
    for i in range(1, l2):
        dtw_matrix[0][i] = dtw_matrix[0][i - 1] + norm(residues1[0] - residues2[i])**2
    for i in range(1, l1):
        for j in range(1, l2):
            v = norm(residues1[i] - residues2[j])**2

            v1 = dtw_matrix[i - 1][j]
            v2 = dtw_matrix[i - 1][j - 1]
            v3 = dtw_matrix[i][j - 1]

            dtw_matrix[i][j] = min(v1, v2, v3) + v
    
    normalisation = 1 / max(l1, l2)
    return np.sqrt(normalisation * np.sqrt(dtw_matrix[-1][-1])**2)


def cluster_antibodies_by_CDR_length(antibodies, ids, selection=reg_def['CDR_all'], clustering='bins', tolerance=same_length_only):
    """ Sort a list of antibody tuples into groups with the same CDR lengths

    :param cluster: list of tuples, antibodies
    :param ids: list, antibody ids
    :param selection: list of np.arrays, indices of residues used for length clustering.
    :param clustering: str, method for clustering antibodies by CDR length. Options are 'bins' or 'greedy'. (default is 'bins')
                       bins: CDRs are grouped into precalculated and equally spaced length bins
                       greedy: stochastic selection of cluster centers
    :param tolerance: np.array, binwidth for length clustering per CDR. (default clustering into bins of identical length)
                      array is required to have the same length as selection, with each element corresponding to the length 
                      tolerance of an individual CDR region.
    :return clusters: dictionary with clusters of antibodies tuples
    :return names: dictionary with ids of the antibodies in each cluster
    """
    clusters = dict()
    names = dict()
    greedy_tolerance = (tolerance - 1) / 2 # length difference instead of binwidth
    
    for i, antibody in enumerate(antibodies):
        cdr_l = np.array(get_CDR_lengths(antibody, selection=selection))

        if clustering == 'bins': # use predefined bins to asign length clusters
            bin = (cdr_l // tolerance) * tolerance
            bin = "_".join(map(str, bin))
            if bin not in clusters:
                clusters[bin] = [antibody]
                names[bin] = [ids[i]]
            else:
                clusters[bin].append(antibody)
                names[bin].append(ids[i])

        elif clustering == 'greedy': # use greedy algorithm to asign length cluster
            in_cluster = False
            for key in clusters:
                key_lengths = np.array(list(map(int, key.split("_"))))

                if all(np.absolute(cdr_l - key_lengths) <= greedy_tolerance):
                    clusters[key].append(antibody)
                    names[key].append(ids[i])
                    in_cluster = True
                    break
            
            if not in_cluster:
                cdr_key = "_".join(map(str, cdr_l))
                clusters[cdr_key] = [antibody]
                names[cdr_key] = [ids[i]]

    return clusters, names


def output_to_pandas(output):
    """ Sort output into a pandas dataframe

    :param output: dict of dicta of lists, clustering output
    :return: pd.DataFrame
    """
    df = pd.DataFrame()

    cluster_by_length = []
    cluster_by_rmsd = []
    structure_id = []

    for key in output:
        for key2 in output[key]:
            for ID in output[key][key2]:
                cluster_by_length.append(key)
                cluster_by_rmsd.append(key2)
                structure_id.append(ID)

    df["ID"] = structure_id
    df["cluster_by_length"] = cluster_by_length
    df["cluster_by_rmsd"] = cluster_by_rmsd

    return df


def check_param(tolerance, d_metric):
    ''' Check if the input parameters are valid'''
    if any(tolerance < 1):
        raise ValueError("All entries of length tolerance must be >= 1")
    if d_metric == 'rmsd' and not all(tolerance == 1):
        raise ValueError("RMSD metric only supports length tolerance 1 for all CDRs")
