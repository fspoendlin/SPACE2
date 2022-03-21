# CDR Structural Clustering

Code to rapidly cluster antibody models by the structure of their CDRs.

## Install

To download and install:

```bash
$ git clone https://github.com/brennanaba/cdr-structural-clustering.git
$ pip install cdr-structural-clustering/
```

## Usage

To run the clustering you will need antibody models, these can be obtained from <a href="http://opig.stats.ox.ac.uk/webapps/newsabdab/sabpred/abodybuilder/">ABodyBuilder</a>.

Once you have a directory with antibody models you can cluster them (first by CDR length and then by structural similarity) using:


```python
import glob
import clustruc

antibody_models = glob.glob("path/to/antibody/models/*.pdb")

output_dict = clustruc.cluster_by_rmsd(OAS_files)
clustered_dataframe = clustruc.util.output_to_pandas(out)
```
