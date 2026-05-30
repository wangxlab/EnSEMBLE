# ENSEMBLE

ENSEMBLE provides enhancer-centric differential analysis and enrichment workflows.

## Enhancer Set Enrichment Analysis (ESEA)

- To reproduce the bundled workflow, run `Rscript run_example.R [optional_output_dir]`. When no directory is supplied, results are written to `outputs/example_run` under the current working directory.
- The script automates loading the package, filters enhancer sets via `retain_specific_enhancers()`, and saves the aligned metadata plus ESEA and GSEA tables inside the output directory.
- Ensure the file paths defined near the top of `run_example.R` point to your local copies of the bundled synthetic count matrices, metadata, and MSigDB collections before launching the script.

### Synthetic example data

A lightweight synthetic dataset ships with the package under
`inst/extdata`. Use `ensemble_example_data()` to locate the files
inside an installed copy of ENSEMBLE or call `use_example_data()` to copy the
assets to a writable directory:

```r
dest <- use_example_data()
list.files(dest)
#> example_background.txt, example_metadata.csv, example_enhancer_counts.tsv, ...
```

The folder contains synthetic enhancer and gene count matrices, metadata,
helper/ESEA/GSEA tables, a synthetic GeneHancer-format annotations example,
GTF annotations, enhancer-set GMTs, and a filled background form. Point the
file paths in `run_example.R` (or your own workflow) to these files to
execute the entire pipeline end-to-end without downloading external resources.

> Note: `inst/extdata/example_genehancer_annotations.tsv` is a **synthetic
> format example** (e.g. IDs `GH001_EMT`, toy coordinates `chr1:1000-1500`)
> and is **not** a subset of real GeneHancer data. For real analyses, see
> the GeneHancer installation note below.

## Enhancer Overlap Enrichment Analysis (eORA)

The eORA utilities map SNPs or loci to GeneHancer IDs and test enrichment
against enhancer sets. The GeneHancer BED is assumed to be BED0 (0-based start,
end-exclusive) and is converted internally to 1-based coordinates for
`GenomicRanges`.

Example: map rsIDs to GHIDs from the bundled demo inputs.

```r
source("R/eORA.R")
snps_in <- readLines("inst/extdata/example_eORA_SNPs.txt")
enhancer_sets <- read_concepts("normal_CellType2Enhancer_v2.gmt")
res <- run_eORA(
    snps = snps_in,
    gh_bed = "inst/extdata/GeneHancer_v5.24.bed",
    enhancer_sets = enhancer_sets,
    ghid_col = 4,
    B = 10000,
    seed = 1,
    alpha = 0.05,
    p_adjust_method = "BH",
    input_type = "rsid"
)
```

Notes:
- `run_eORA` accepts rsIDs, locus strings (`chr:pos` or `chr:start-end`), or
  data frames with `chr+pos` or `chr+start+end`.
- rsID mapping uses `biomaRt` and requires network access; use locus inputs if
  you need fully offline runs. If biomaRt returns 0-based positions, the code
  shifts to 1-based and emits a message when `quiet = FALSE`.

## Installation

ENSEMBLE depends on several Bioconductor packages (`DESeq2`, `edgeR`,
`fgsea`, `GenomicRanges`, `IRanges`, `S4Vectors`, `rtracklayer`) plus CRAN
packages such as `data.table`, `Matrix`, and `jsonlite`. Install the required
Bioconductor components first, then pull ENSEMBLE from GitHub:

```r
install.packages("BiocManager")
BiocManager::install(c("DESeq2", "edgeR", "fgsea", "GenomicRanges",
                       "IRanges", "S4Vectors", "rtracklayer"))

install.packages("devtools")
devtools::install_github("cloudmacchiato/ENSEMBLE")
```

After installation, attach the package and locate the bundled demo files:

```r
library(ENSEMBLE)
ensemble_example_data()
```

### GeneHancer (required for enhancer-to-gene mapping)

EnSEMBLE uses **GeneHancer v5.24**, which is distributed under license by
GeneCards / LifeMap Sciences and **cannot be redistributed**. It is therefore
**not** bundled with this repository. Obtain GeneHancer v5.24 directly from
GeneCards (https://www.genecards.org/, GeneHancer licensing), place the bed
file at `inst/extdata/GeneHancer_v5.24.bed`, and run
`Rscript inst/scripts/setup_genehancer.R` to verify.

Expected format: tab-separated, no header, columns `chr`, `start`, `end`,
`GHid` (e.g. `chr20  237139  238398  GH20J000237`).

Please cite Fishilevich et al. (2017), *Database*,
doi:[10.1093/database/bax028](https://doi.org/10.1093/database/bax028).

### Swapping in your own datasets

`run_example.R` now reads the bundled example paths by default and lets you
override any input via `example_config.json` in your working directory. Update
the JSON with lab-specific files (enhancer/gene counts, metadata, annotations,
MSigDB GMTs, and contrast definition) and rerun the script:

```json
{
  "counts_file_enh": "/path/to/GeneHancer_counts.tsv",
  "counts_file_gene": "/path/to/HGNC_counts.tsv",
  "genehancer_file": "/path/to/GeneHancer.saf",
  "gencode_file": "/path/to/gencode.gtf",
  "enhancer_set_file": "/path/to/enhancer_sets.gmt",
  "metadata_file": "/path/to/metadata.csv",
  "msigdb_files": ["/path/msigdb/h.all.v2024.1.Hs.symbols.gmt"],
  "contrast": ["group", "Treatment", "Control"]
}
```

Leave keys untouched to keep using the packaged synthetic example data.

---

# Python Evidence-Classifier Agent (`local_agent/`)

`local_agent/` is a Python module that consumes the R-side outputs
(GSEA results, ESEA helpers, background context) and produces structured
verdicts (SUPPORTED / PARTIAL / GENE_LEVEL_ONLY) plus a 2–3 page PDF
report per dataset. v2.0 uses **Anthropic Claude (Sonnet/Opus 4.5)** as
the backing LLM.

If you were using the v1.x Gemini-backed agent, see [MIGRATION.md](MIGRATION.md)
for the flag-mapping table. The v1.x CLI invocation continues to work via
the back-compat shim, but Google Gemini is no longer supported.

## 1. Prerequisites

- Python 3.10 or newer
- An Anthropic API key (https://console.anthropic.com)
- WeasyPrint system libraries: `libpango-1.0-0`, `libcairo2`,
  `libgdk-pixbuf-2.0-0`, `libffi-dev` (Debian/Ubuntu) or
  `brew install pango cairo gdk-pixbuf` (macOS)
- R-side outputs already in place: `GSEA_results.csv`, `ESEA_helpers.csv`,
  and a filled `background.txt` (use `background_form_template.txt` as a
  scaffold)

## 2. Install Python dependencies

```bash
conda create -n ensemble python=3.10
conda activate ensemble
pip install -r requirements.txt
```

`requirements.txt` pins: `anthropic`, `pydantic`, `pandas`, `matplotlib`,
`networkx`, `scipy`, `scikit-learn`, `dynamicTreeCut`, `Markdown`,
`WeasyPrint`.

## 3. Configure the API key

```bash
export ANTHROPIC_API_KEY=sk-ant-...     # in current shell
# or save to a sourceable file:
echo 'export ANTHROPIC_API_KEY=sk-ant-...' > ~/.anthropic_env
chmod 600 ~/.anthropic_env
source ~/.anthropic_env
```

The CLI never accepts the key as a flag (keys on the command line leak
into `ps` output and shell history).

## 4. Run the agent

There are two equivalent invocation styles. **Either works** — pick
whichever fits your existing workflow.

### 4.1 File-path style (v1.x-compatible)

Useful when files live in arbitrary locations on disk:

```bash
python -m local_agent.cli \
  --gsea-csv GSEA_results.csv \
  --esea-csv ESEA_helpers.csv \
  --background-txt background.txt \
  --output-dir outputs/my_run
```

### 4.2 Dataset style (v2.0 canonical)

Easier when you organise inputs by dataset name. Expects:

```
inputs/<dataset>/GSEA_results.csv
inputs/<dataset>/ESEA_helpers.csv
inputs/<dataset>/backgrounds.txt
```

Then:

```bash
python -m local_agent.cli --dataset bt20
python -m local_agent.cli --dataset all   # runs bt20, ipsc, panc1, snai1
```

### 4.3 Model choice

```bash
python -m local_agent.cli --dataset bt20 --model claude-sonnet-4-5    # default, cheaper
python -m local_agent.cli --dataset bt20 --model claude-opus-4-5      # for graded PARTIAL outputs
```

Opus is recommended for production reports — Sonnet sometimes collapses
to binary SUPPORTED vs GENE_LEVEL_ONLY and skips the PARTIAL tier.

### 4.4 Tuning knobs

The agent runs deterministically (temperature=0) by default. Useful flags:

| Flag | Default | Effect |
|---|---|---|
| `--temperature FLOAT` | 0.0 | Increase for prose variety; keep 0 for reproducibility |
| `--max-tokens INT` | 8192 | Larger output budget if iPSC-scale datasets hit a cap |
| `--q-threshold FLOAT` | 0.05 | GSEA pathway q-value cutoff before clustering |
| `--merge-jaccard FLOAT` | 0.5 | Post-clustering merger threshold on full leading-edge Jaccard; 0 disables |
| `--theme-cap-total INT` | 40 | Maximum themes per dataset sent to the LLM |
| `--no-api` | off | Skip the API call; emit all-GLO fallback (smoke test) |

See `python -m local_agent.cli --help` for the full list. v1.x flags are
accepted with a deprecation notice — run
`python -m local_agent.cli --migration-guide` for the mapping table.

### 4.5 Background contrast checklist (unchanged from v1.x)

- Fill the `Contrast:` field in `background_form_template.txt`
  (e.g. `Treatment vs Control`).
- State which cohort is the numerator so positive/negative NES is
  interpreted correctly.

## 5. Mini-thesis (narrative report)

After classification succeeds, generate a 400–600 word structured
mini-thesis:

```bash
python -m local_agent.report.build_thesis --dataset bt20
```

This calls Claude Sonnet (cheap; `~$0.05`) and writes:

- `outputs/<dataset>/mini_thesis.md` — structured 4-section markdown
- `outputs/<dataset>/thesis_validation.json` — length + section + hallucination warnings

## 6. PDF report (figures + verdicts + thesis)

```bash
python -m local_agent.report.build_report --dataset bt20
```

Writes `outputs/<dataset>/report_<dataset>.pdf` containing:

1. Compression figure (significant pathways → themes → verdicts)
2. Bipartite evidence network (SUPPORTED/PARTIAL above, GLO context below)
3. ESEA helper overview (linked vs unlinked)
4. Mini-thesis prose
5. Full verdict appendix table

## 7. Reproducibility (optional)

For a published claim table, run the dataset 3× and consolidate:

```bash
python -m local_agent.reproducibility --runs 3 --model claude-opus-4-5
python -m local_agent.consensus --output outputs/v2_2_lock
python -m local_agent.report.build_thesis --dataset all --output-dir outputs/v2_2_lock
python -m local_agent.report.build_report --dataset all
```

Consensus rules: 3/3 unanimous → ship; 2/3 majority on verdict + union/intersection on helpers (helpers in ≥2 runs survive); 3-way split → GENE_LEVEL_ONLY (conservative). See `outputs/v2_2_lock/lockfile.md` and `outputs/v2_2_lock/consensus_report.md` after running.

## 8. Outputs

Per dataset, you'll find:

```
outputs/<dataset>/
├── agent_input.json           # what was sent to Claude
├── verdicts.json              # final verdicts (with theme_weight + linked_helpers list)
├── validation.json            # deterministic validator outcome
├── api_log.json               # full request/response
├── clustering/
│   ├── cluster_themes_{up,down}.json
│   ├── merge_log.{json,md}
│   ├── cluster_themes_{up,down}_network.{pdf,png}
│   └── theme_summaries.json
├── mini_thesis.md             # (if --report or build_thesis was run)
├── thesis_validation.json
├── figures/
│   ├── fig_compression.{pdf,png}
│   ├── fig_network.{pdf,png}
│   └── fig_esea_overview.{pdf,png}
├── verdict_table.md
└── report_<dataset>.pdf       # final unified PDF deliverable
```

## 9. Troubleshooting

- **`ANTHROPIC_API_KEY not set`**: export the variable in the current shell
  before running.
- **`max_tokens` truncated output**: bump `--max-tokens 12288` (Claude 4.5
  supports up to 64k).
- **API credit / org disabled**: rotate the key at console.anthropic.com.
  Do **not** paste keys in chat transcripts — Anthropic auto-revokes leaked
  keys.
- **All-GENE_LEVEL_ONLY output**: check `validation.json`. If `used_fallback`
  is true, the LLM output violated a validation rule (capacity, direction,
  rationale length, etc.) and the runner emitted the conservative fallback.
- **Different dataset names than `bt20/ipsc/panc1/snai1`**: any dataset name
  works as long as `inputs/<name>/` has the three required files. The
  default datasets are just for `--dataset all` convenience.

## 10. Python tests

```bash
pytest local_agent/tests/                  # 39 tests: validator, merger, weight, thesis validator
# or individually:
python local_agent/tests/test_validator.py
```

These are deterministic — no API calls required.
