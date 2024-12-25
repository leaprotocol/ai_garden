(
  echo '<?xml version="1.0" encoding="UTF-8"?>';
  echo '<multipleDirSummary>';

  # tex/clona2 - main tex files and parts
  node dir_summary.js ~/Projects/clona/tex/clona2 --max-depth 2 --include-content --max-items-per-folder 10 \
    --exclude-content-types .aux,.log,.out,.toc,.bbl,.blg,.synctex.gz,.pdf,.png,.jpg,.lua,.cls \
    --ignore-path ~/Projects/clona/venv,~/Projects/clona/tex/clona2/.git,~/Projects/clona/tex/clona2/Images,~/Projects/clona/tex/clona2/unused,~/Projects/clona/tex/clona2/fithesis \
    --content-length 2000;

  # repo/clona - source code and configs
  node dir_summary.js ~/Projects/clona/repo/clona --max-depth 3 --include-content --max-items-per-folder 20 \
    --exclude-content-types .pyc,.pyo,.git,.coverage \
    --ignore-path ~/Projects/clona/venv,~/Projects/clona/repo/clona/__pycache__,~/Projects/clona/repo/clona/.pytest_cache,~/Projects/clona/repo/clona/.git \
    --content-length 4000;

  # datasets - metadata and analysis results
  node dir_summary.js ~/Projects/clona/repo/clona/datasets --max-depth 3 --include-content --max-items-per-folder 15 \
    --exclude-content-types .pkl,.h5,.npy,.npz,.zip,.tar,.gz,.raw,.cr2,.nef,.arw,.jpg,.png \
    --ignore-path ~/Projects/clona/venv,~/Projects/clona/repo/clona/datasets/temp \
    --content-length 3000;

  echo '</multipleDirSummary>'
) > combined_output.xml