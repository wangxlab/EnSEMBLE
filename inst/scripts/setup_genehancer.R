#!/usr/bin/env Rscript
# GeneHancer v5.24 is distributed under license by GeneCards/LifeMap Sciences
# and CANNOT be redistributed. This script does not download it.
# Obtain GeneHancer v5.24 directly from https://www.genecards.org/ (GeneHancer
# licensing), then place the bed file at inst/extdata/GeneHancer_v5.24.bed.
#
# Expected format (tab-separated, no header), columns:
#   [1] chr   [2] start   [3] end   [4] GHid   (e.g. chr20  237139  238398  GH20J000237)

dest <- file.path("inst", "extdata", "GeneHancer_v5.24.bed")
if (file.exists(dest)) {
  message("GeneHancer v5.24 found at ", dest, " - ready.")
} else {
  stop(
    "GeneHancer v5.24 not found at ", dest, ".\n",
    "It is licensed and not bundled with EnSEMBLE. Obtain it from GeneCards ",
    "(https://www.genecards.org/, GeneHancer licensing) and place it there.\n",
    "Expected columns: chr, start, end, GHid (tab-separated)."
  )
}
