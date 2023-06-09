# Structural Profiling of Antibodies to Cluster by Epitope 2 (SPACE2)

Code to rapidly cluster antibody models by the structure of their CDRs.

## Install

To download and install:

```bash
$ git clone https://github.com/fspoendlin/SPACE2.git
$ pip install SPACE2/
```

## Usage

To run the clustering you will need antibody models which are IMGT numbered, these can be obtained from <a href="https://github.com/brennanaba/ImmuneBuilder">ImmuneBuilder</a>.

Once you have a directory with antibody models you can cluster them using agglomerative clustering:

```python
import glob
import SPACE2

antibody_models = glob.glob("path/to/antibody/models/*.pdb")

clustered_dataframe = SPACE2.agglomerative_clustering(antibody_models, cutoff=1.25)
```

greedy clustering:

```python
clustered_dataframe = SPACE2.greedy_clustering(antibody_models, cutoff=1.25)
```

or a custom clustering aglorithm. The clustering algorithm should be a class and follow the syntax of scikit-learn. The class must have a `self.fit(X)` method that takes a distance matrix as an input and store cluster labels in a `self.labels_` attribute.

```python
algorithm = CustomClusteringAlgorithm(*args, **kwargs)
clustered_dataframe = SPACE2.cluster_with_algorithm(algorithm, antibody_models)
```

Antibodies are first grouped by CDR length and then clustered by structural similarity with the selected algorithm. The output is a pandas dataframe with columns for filenames, CDR length clusters (containing the number of residues of all CDRs considered during the clustering) and structural clusters (indicating the filename of a cluster representative).

By default the length of all CDRs will be used to group the antibodies and RMSDs will be calculated across all of them after alignment on both heavy and light chain framework regions. CDRs and frameworks to used can be adapted with `selection` and `anchor` arguments. Here is an example how to restict clustering to heavy chain CDRs and framework.

```python
from SPACE2 import reg_def

cdr_selection = [reg_def['CDRH1'], reg_def['CDRH2'], reg_def['CDRH3']]
fw_selection = [reg_def['fwH']]

clustered_dataframe = SPACE2.agglomerative_clustering(antibody_models, selection=cdr_selection, anchors=fw_selection, cutoff=1.25)
```
