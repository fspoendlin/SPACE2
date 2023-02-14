# CDR Structural Clustering

Code to rapidly cluster antibody models by the structure of their CDRs.

## Install

To download and install:

```bash
$ git clone https://github.com/brennanaba/cdr-structural-clustering.git
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

or greedy clustering algorithms:

```python

clustered_dataframe = SPACE2.greedy_clustering(antibody_models, cutoff=1.0)
```

Antibodies are first grouped by CDR length and then clustered by structural similarity with the selected algorithm. The output will be a pandas dataframe with columns for filenames, CDR length clusters and structural clusters.
