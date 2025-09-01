#!/bin/bash

MAN= ~/nnlab/data/manifests/aircraft_eval.csv

(
  echo "filepath,label,split"
  find ~/nnlab/datasets/raw/opt_aircraft -type f -iname '*.png' \
  | while IFS= read -r p; do
      abs=$(readlink -f "$p")
      # take the first folder after ".../opt_aircraft/"
      top=$(printf "%s\n" "$abs" | sed -E 's|.*/opt_aircraft/([^/]+)/.*|\1|')
      printf "%s,%s,val\n" "$abs" "$top"
    done
) > "$MAN"
