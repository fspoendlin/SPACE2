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

clustered_dataframe = SPACE2.agglomerative_clustering(antibody_models, cutoff=1.0)
```

greedy clustering:

```python
clustered_dataframe = SPACE2.greedy_clustering(antibody_models, cutoff=1.0)
```

or a custom clustering aglorithm. The clustering algorithm should be a class and follow the syntax of scikit-learn. The class must have a `self.fit(X)` method that takes a distance matrix as an input and store cluster labels in a `self.labels_` attribute.

```python
algorithm = CustomClusteringAlgorithm(*args, **kwargs)
clustered_dataframe = SPACE2.cluster_with_algorithm(algorithm, antibody_models)
```

Antibodies are first grouped by CDR length and then clustered by structural similarity with the selected algorithm. The output will be a pandas dataframe with columns for filenames, CDR length clusters and structural clusters.
