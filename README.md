# MANTIS

Spatial Metabolomics (SM) technology is transforming the fine-mapping of metabolic states associated with tissue function, and its power is greatly enhanced if augmented by a matching Spatial Transcriptomics (ST) profile. Several recent tools enable analysis of matched SM and ST data, including alignment and integration of the two modalities. However, there is only rudimentary support for probing biological relationships between the two omics views. We present a computational tool called MANTIS, which analyzes paired and aligned SM+ST profiles of a sample, optionally along with spatial domain or cell type information, to discover metabolite spatial distribution patterns and gene-metabolite relationships. It employs a novel strategy to assess statistical significance of such findings, based on specialized permutation tests that control for spatial autocorrelation of metabolite distribution. It disentangles different sources of spatial patterns and correlations, viz., those arising from regional preferences, cell type associations of a molecule, or other unknown factors. It also introduces the use of spatial cross-correlation statistics for quantifying gene-metabolite associations. We present the functionalities of MANTIS through application to three data sets spanning different spatial technologies for measuring metabolites and gene expression, different tissues and species. We compare our findings to suitable baselines representing existing tools and argue that MANTIS achieves superior specificity in detecting patterns and associations, through its rigorous statistical procedures.  

![MANTIS_overview](https://github.com/yuhaotuo/MANTIS/main/files/figure_1.png)

---

## How to install MANTIS

The package can be installed using pip.

```
pip install sc-mantis
```

---

## How to use MANTIS

The detailed tutorial is available [here](https://github.com/yuhaotuo/MANTIS/main/files/tutorial.ipynb).

MANTIS expects data (both single cell transcriptomic and metabolomic) to be in csv format and can be loaded using 

### Loading Data

```
mdata = mt.io.load_data(coords= "spot_coordinates.csv", m_file="msi.csv", g_file = "rna.csv", region="regions.csv")
```

 - `coords`:  Coordinates of the spots/cells.
 - `m_file`: Metabolite expression file. (spots x n_metabolites).
 - `g_file`: Gene expression file. (spots x n_genes).
 - `region`: Region annotation file (spots x 1).
 - `cell_type`: Cell type annotation file (spots x 1).


### Sampling metabolite null

```
dmin = math.sqrt(2.0)
l = 4 * dmin
mdata, G = mt.tl.sample(mdata, l = l)
```
 - `l`:  Distance parameter.

### Regional Metabolites

```
mdata = mt.tl.compute_regional_metabolite(mdata, alpha = 0.1)
```
 - `alpha`: Significance threshold.
 
### Cell Type Metabolites

```
mdata = mt.tl.celltype_met(mdata)
```

### Spatially Variable Metabolites

```
mdata = mt.tl.spatvar_metabolite(mdata)
```

### Gene-Metabolite SCI

```
mdata = mt.tl.compute_genemet_sci(mdata)
```

### SPC CT

```
mdata = mt.tl.compute_spc_ct(mdata)
```

### SPC SD

```
mdata = mt.tl.compute_spc_ct(mdata)
```