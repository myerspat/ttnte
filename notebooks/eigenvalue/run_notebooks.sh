# Script for running eigenvalue notebooks

# Anaconda kernel
kernel="tt_nte"

# List of notebooks to run
notebooks=(
  "./square/square.ipynb"
  "./circle/circle.ipynb"
  "./quarter_circle/quarter_circle.ipynb"
  "./pincell/pincell.ipynb"
  "./lightbridge/lightbridge_ba.ipynb"
  "./lightbridge/lightbridge_gas.ipynb"
)

for notebook in ${notebooks[*]}
do
  echo "Running papermill on $notebook"

  # Get directory name and base name
  dir=$(dirname "$notebook")
  base=$(basename "$notebook")

  # Run papermill on the notebook in that directory
  (
    cd $dir || exit
    papermill $base $base -k $kernel
  )

  if [ $? -eq 0 ]; then
      echo "✓ Successfully executed: $notebook"
  else
      echo "✗ Failed to execute: $notebook"
  fi

done
