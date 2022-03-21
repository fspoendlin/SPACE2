import numpy as np
import numba as nb
import pandas as pd
from joblib import Parallel, delayed

# All are north region definitions in imgt numbering
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
reg_def["CDR_H_all"] = reg_def["CDRH1"] + reg_def["CDRH2"] + reg_def["CDRH3"]
reg_def["CDR_L_all"] = reg_def["CDRL1"] + reg_def["CDRL2"] + reg_def["CDRL3"]
reg_def["CDR_all"] = reg_def["CDR_H_all"] + reg_def["CDR_L_all"]
reg_def["fw_all"] = reg_def["fwH"] + reg_def["fwL"]

# numba does not like lists very much
reg_def = {x: np.array(reg_def[x]) for x in reg_def}

# Store these to use in jit
reg_def_CDR_all = reg_def["CDR_all"]
reg_def_fw_all = reg_def["fw_all"]


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
    numbers = np.zeros(size, dtype=int)
    coords = np.zeros((size, 3))

    for i in range(size):
        line = lines[i]
        chain_term = 128 if line[21] == "L" else 0
        numbers[i] = int(line[22:26]) + chain_term
        x = float(line[30:38])
        y = float(line[38:46])
        z = float(line[46:54])
        coords[i] = (x, y, z)

    return numbers, coords


def parse_antibody(file):
    with open(file) as f:
        txt = f.read()
    return get_antibody(txt)


def parse_antibodies(files, n_jobs=20):
    return Parallel(n_jobs=n_jobs)(delayed(parse_antibody)(file) for file in files)


@nb.njit
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


def get_CDR_lengths(antibody):
    len_h1 = str(len(get_residues(antibody, reg_def["CDRH1"])))
    len_h2 = str(len(get_residues(antibody, reg_def["CDRH2"])))
    len_h3 = str(len(get_residues(antibody, reg_def["CDRH3"])))
    len_l1 = str(len(get_residues(antibody, reg_def["CDRL1"])))
    len_l2 = str(len(get_residues(antibody, reg_def["CDRL2"])))
    len_l3 = str(len(get_residues(antibody, reg_def["CDRL3"])))
    return len_h1, len_h2, len_h3, len_l1, len_l2, len_l3


@nb.njit
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


@nb.njit
def mean_numba(x):
    res = []
    for i in range(x.shape[-1]):
        res.append(x[:, i].mean())

    return np.array(res)


@nb.njit
def get_alignment_transform(fixed, moveable, anchors):
    anch1 = get_residues(moveable, anchors)
    anch2 = get_residues(fixed, anchors)

    V, _, W = np.linalg.svd(anch1.T @ anch2)
    U = V @ W

    if np.linalg.det(U) < 0:
        U = (np.array([[1, 1, -1]]) * V) @ W

    return mean_numba(anch1), mean_numba(anch2), U


@nb.njit
def align(fixed, moveable, anchors):
    x, y, U = get_alignment_transform(fixed, moveable, anchors)

    return moveable[0], ((moveable[1] - x) @ U + y)


@nb.njit
def rmsd(ab1, ab2, selection=reg_def_fw_all, anchors=reg_def_CDR_all):
    residues1 = get_residues(ab1, selection)
    residues2 = get_residues(align(ab1, ab2, anchors), selection)
    l = len(residues1)

    total = 0
    for i in range(l):
        total = total + sum((residues1[i] - residues2[i]) ** 2)

    return np.sqrt(total / l)


def output_to_pandas(output):
    """ Sort output into a pandas dataframe

    :param output: dictionary of dictionaries of lists outputted by clustering
    :return: pandas dataframe
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
