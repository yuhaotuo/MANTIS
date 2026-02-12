import pandas as pd
import numpy as np
import anndata as ad
from mudata import MuData
from mudata import set_options
import mudata as mu
import os
import types

def _summarize(self):
    """Structured summary of MuData with grouped .uns fields."""
    print(f"MuData object with n_obs × n_vars = {self.n_obs} × {self.n_vars}")
    print("obs:", ", ".join(self.obs_keys()))
    print("obsm:", ", ".join(self.obsm_keys()))
    print("uns:")

    # --- Predefined grouping map ---
    group_map = {
        'cell_type': ['celltype_metabolite', 'celltype_metabolite_per_ct'],
        'region': ['regional_metabolite', 'regional_metabolite_fit'],
        'spatvar': ['spatvar_metabolite', 'spatvar_metabolite_celltype', 
                    'spatvar_metabolite_celltype_coef', 'spatvar_metabolite_celltype_summary', 
                    'spatvar_metabolite_combined', 'spatvar_metabolite_region'],
        'genemet': ['genemet_sci', 'genemet_sci_summary'],
        'misc': [],
    }

    # flatten group map for lookup
    assigned = set(k for keys in group_map.values() for k in keys)
    uns_keys = set(self.uns.keys())
    ungrouped = sorted(uns_keys - assigned)

    # print present keys per group
    for group, keys in group_map.items():
        present = [k for k in keys if k in self.uns]
        if present:
            print(f"  {group:<22}: {', '.join(present)}")

    if ungrouped:
        print(f"  {'other':<22}: {', '.join(ungrouped)}")


def _attach_summarize(mdata):
    """Attach summarize() method to a single MuData object in memory."""
    mdata.summarize = types.MethodType(_summarize, mdata)
    return mdata

def load_data(
    g_file=None,          # CSV of gene expression (cells × genes)
    m_file=None,          # CSV of metabolite abundances (cells × metabolites)
    coords=None,          # CSV of spatial coordinates
    cell_type=None,       # CSV of cell type labels
    region=None,          # CSV of region labels
    mdata=None            # existing .h5mu file path
):
    set_options(pull_on_update=False)
    """
    Load or build a MuData object containing multi-omic and spatial data.

    If `mdata` is provided, loads it directly and ignores all other inputs.
    Otherwise, builds a MuData object from the provided CSV files.
    """

    if mdata is not None:
        if not os.path.exists(mdata):
            raise FileNotFoundError(f"File not found: {mdata}")
        mdata_obj = mu.read_h5mu(mdata)
        mdata_obj = _attach_summarize(mdata_obj)
        # mu.MuData.summarize = _summarize
        return mdata_obj

    if coords is None:
        raise ValueError("Spatial coordinate file (coords) must be provided.")
    coords_df = pd.read_csv(coords, index_col=0)

    gene_df = pd.read_csv(g_file, index_col=0) if g_file else None
    met_df = pd.read_csv(m_file, index_col=0) if m_file else None

    common_idx = coords_df.index
    if gene_df is not None:
        common_idx = common_idx.intersection(gene_df.index)
    if met_df is not None:
        common_idx = common_idx.intersection(met_df.index)

    if len(common_idx) == 0:
        raise ValueError("No common cell indices found among provided datasets.")

    coords_df = coords_df.loc[common_idx]
    if gene_df is not None:
        gene_df = gene_df.loc[common_idx]
    if met_df is not None:
        met_df = met_df.loc[common_idx]

    adatas = {}
    if gene_df is not None:
        adatas["gene"] = ad.AnnData(
            X=gene_df.to_numpy(dtype=float),
            obs=pd.DataFrame(index=common_idx),
            var=pd.DataFrame(index=gene_df.columns)
        )
    if met_df is not None:
        adatas["metabolite"] = ad.AnnData(
            X=met_df.to_numpy(dtype=float),
            obs=pd.DataFrame(index=common_idx),
            var=pd.DataFrame(index=met_df.columns)
        )

    if not adatas:
        raise ValueError("No omic data provided. Must provide at least one of g_file or m_file.")

    mdata_obj = MuData(adatas)

    # obs_meta = pd.DataFrame(index=common_idx)
    if cell_type is not None:
        celltype_df = pd.read_csv(cell_type, index_col=0)
        mdata_obj.obsm["cell_type"] = celltype_df.loc[common_idx]
    if region is not None:
        region_df = pd.read_csv(region, index_col=0)
        mdata_obj.obsm["region"] = region_df.loc[common_idx]
        # print(obs_meta)
    # mdata_obj.obsm = obs_meta

    mdata_obj.obsm["spatial"] = coords_df
    mdata_obj = _attach_summarize(mdata_obj)
    mu.MuData.summarize = _summarize
    return mdata_obj
