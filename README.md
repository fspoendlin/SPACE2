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

To run the clustering you will need antibody models which are IMGT numbered and with chain identifier 'H' for the heavy chain and 'L' for the light chain. Models with IMGT numbering and correct chain identifiers can be obtained from a default run of <a href="https://github.com/brennanaba/ImmuneBuilder">ImmuneBuilder</a>.

Once you have a directory with antibody models you can cluster them using SPACE2. We recommend using the agglomerative clustering method with an RMSD threshold of 1.25 Å and default `selection` and `anchors` parameters. This will use the length of all six CDRs to group the antibodies and RMSDs will be calculated across all six CDRs after alignment on both heavy and light chain framework regions. North CDR definitions are used by default.

```python
import glob
import SPACE2

antibody_models = glob.glob("path/to/antibody/models/*.pdb")

clustered_dataframe = SPACE2.agglomerative_clustering(antibody_models, cutoff=1.25)
```

Alternatively a greedy clustering algorithm is implemented:

```python
clustered_dataframe = SPACE2.greedy_clustering(antibody_models, cutoff=1.25)
```

or a custom clustering aglorithm can be used. The clustering algorithm should be a class and follow the syntax of scikit-learn. The class must have a `self.fit(X)` method that takes a distance matrix as an input and store cluster labels in a `self.labels_` attribute.

```python
algorithm = CustomClusteringAlgorithm(*args, **kwargs)
clustered_dataframe = SPACE2.cluster_with_algorithm(algorithm, antibody_models)
```

CDRs considered during the initial step of grouping by length and used for the RMSD calculation can be modified with `selection` arguments. Frameworks to be used for structural alignment can be modified using the `anchor` arguments. Here, is an example of how to restict SPACE2 to only consider the length and RMSDs of heavy chain CDRs and align on the heavy chain framework.

```python
from SPACE2 import reg_def

cdr_selection = [reg_def['CDRH1'], reg_def['CDRH2'], reg_def['CDRH3']]
fw_selection = [reg_def['fwH']]

clustered_dataframe = SPACE2.agglomerative_clustering(antibody_models, selection=cdr_selection, anchors=fw_selection, cutoff=1.25)
```

Lists of residues can be passed for selection or anchors to customise the regions considered during clustering. The lists should contain the the IMGT number of each residue to include for the heavy chain and the IMGT number + 128 for the light chain. The code below provides an example of how to use CDR3 residues by IMGT region definitions.

```python
from SPACE2 import reg_def

CDRH3 = list(range(105, 118))
CDRL3 = [x+128 for x in CDRH3]

cdr_selection = [reg_def['CDRH1'], reg_def['CDRH2'], CDRH3, reg_def['CDRL1'], reg_def['CDRL2'], CDRL3]
fw_selection = [reg_def['fwH']]

clustered_dataframe = SPACE2.agglomerative_clustering(antibody_models, selection=cdr_selection, anchors=fw_selection, cutoff=1.25)
```

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

