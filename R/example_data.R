#' Locate bundled example data
#'
#' Returns the absolute path to the directory containing the packaged
#' synthetic example files (all prefixed `example_`). Use this helper when
#' programmatically constructing file paths for the example workflow or
#' copying the assets to a writable directory prior to running a local
#' analysis.
#'
#' @return Absolute path to the directory holding the example files.
#' @export
ensemble_example_data <- function() {
  system.file("extdata", package = "ENSEMBLE")
}

#' Copy the bundled synthetic example files to a destination directory
#'
#' This helper copies the synthetic example files (counts, metadata,
#' helper tables, and background form; all prefixed `example_`) into a
#' writable directory so that users can run the preprocessing and agent
#' workflows without touching the original files inside the installed
#' package. The user-supplied GeneHancer bed (a licensed reference,
#' not bundled) is intentionally NOT copied even if it has been placed
#' alongside the example files.
#'
#' @param dest_dir Path to the directory where the files should be copied.
#'   Created when it does not already exist. Defaults to a temporary directory.
#'
#' @return Character vector with the copied file paths (invisibly).
#' @export
use_example_data <- function(dest_dir = tempfile("ensemble_example_")) {
  src <- ensemble_example_data()
  if (!nzchar(src)) {
    stop("Example data directory not found inside the package.")
  }
  if (!dir.exists(dest_dir)) {
    dir.create(dest_dir, recursive = TRUE, showWarnings = FALSE)
  }
  example_files <- list.files(src, pattern = "^example_", full.names = TRUE)
  copied <- file.path(dest_dir, basename(example_files))
  file.copy(example_files, copied, overwrite = TRUE)
  invisible(copied)
}
