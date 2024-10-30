# Structural Profiling of Antibodies to Cluster by Epitope 2 (SPACE2)

The SPACE2 algorithm is a method that rapidly clusters antibodies by the similarity of structural models and accurately groups antibodies that bind the same epitope.

SPACE2 requires structural models of antibodies as an input, these can be generated with <a href="https://github.com/brennanaba/ImmuneBuilder">ImmuneBuilder</a>. Antibodies are then clustered in three main steps. Initially, the models are split into groups of identical CDR lengths. Models in each group are then structurally aligned on the Cα of residues in framework regions and a pairwise distance matrix is computed of the Cα RMSDs of CDR loop residues. The antibodies are then clustered based on these distances.

## Install

To download and install:

```bash
$ git clone https://github.com/fspoendlin/SPACE2.git
$ pip install SPACE2/
```

## Usage

To run the clustering you will need antibody models which are IMGT numbered and with chain identifier 'H' for the heavy chain and 'L' for the light chain. Models with IMGT numbering and correct chain identifiers can be obtained from a default run of <a href="https://github.com/brennanaba/ImmuneBuilder">ImmuneBuilder</a>. Once you have a directory with antibody models you can cluster them using SPACE2. 

An example of how to cluster antibodies with SPACE2 using agglomerative clustering and default parameters is shown below. This is the recommended way of clustering antibodies.

```python
import glob
import SPACE2

antibody_models = glob.glob("path/to/antibody/models/*.pdb")

clustered_dataframe = SPACE2.agglomerative_clustering(antibody_models, cutoff=1.25, n_jobs=-1)
```

The above code will divide antibodies into groups of same length CDRs. For each group antibody modells are structurally superimposed on the heavy and light chain framework regions. Then C-alpha RMSD is calculated across all six CDRs (North CDR definitions are used by default) and a agglomerative clustering algorithm with a distance threshold of 1.25 Å is used to group the antibodies. The output is a pandas dataframe containing the assigned structural cluster for each antibody.

The SPACE2 package supports a range of options to customise clustering e.g:
* Select CDRs for structural comparison and framework regions for structural alignment
* Dynamic time warping distance calculation (drops requirement for identical CDR lengths)
* Custom clustering algorithms 
See notebooks for example usage.

## Output

SPACE2 outputs a pandas dataframe containing the assigned structural cluster for each antibody. The output is formatted as below with columns indicating the antibody name (ID), the length of all CDRs considered during clustering in order H1-3 and L1-3 (cluster_by_length) and a representative of the assigned structural cluster (cluster_by_rmsd). 

<div align="center">

| | ID | cluster_by_length | cluster_by_rmsd |
| --- | --- | --- | --- |
| 0 | BD56-1450.pdb | 15_9_12_11_8_8 | BD56-1450.pdb |
| 1 | BD55-6240.pdb | 15_9_12_11_8_8 | BD56-1450.pdb |
| 2 | BD55-1117.pdb | 13_10_13_13_8_11 | BD55-1117.pdb |
| ... | ... | ... | ... |
 
</div>

## Stats

SPACE2 clusters 10,000 antibodies in approximately 2 min when parallelised over 12 CPUs. The algorithm scales roughly
at O(n<sup>1.5</sup>) with the number of antibodies (n). 

## Citation

```
@article{Spoendlin2023,
	title = {Improved computational epitope profiling using structural models identifies a broader diversity of antibodies that bind the same epitope},
	author = {Fabian C. Spoendlin, Brennan Abanades, Matthew I. J. Raybould, Wing Ki Wong, Guy Georges, and Charlotte M. Deane},
	journal = {Frontiers in Molecular Biosciences},
	doi = {10.3389/fmolb.2023.1237621},
	volume = {10},
	year = {2023},
}
```

