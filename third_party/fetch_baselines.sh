#!/bin/bash
# Clone the published mixers used by explore3d/zoo2d_ext.py (and the Pathfinder generator) at the pinned commits.
#   bash third_party/fetch_baselines.sh
set -e
cd "$(dirname "$0")"
mkdir -p baselines
grep -v '^#' baselines.txt | while read -r name url commit; do
  [ -z "$name" ] && continue
  if [ ! -d "baselines/$name" ]; then git clone -q "$url" "baselines/$name"; fi
  git -C "baselines/$name" checkout -q "$commit"
  echo "$name @ $commit"
done
