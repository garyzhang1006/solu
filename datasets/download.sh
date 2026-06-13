#!/usr/bin/env bash
# Re-download all datasets from their original sources.
set -e
cd "$(dirname "$0")"
curl -L -o esol.csv          "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv"
curl -L -o freesolv.csv      "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/SAMPL.csv"
curl -L -o lipophilicity.csv "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/Lipophilicity.csv"
curl -L -o aqsoldb.csv       "https://dataverse.harvard.edu/api/access/datafile/3407241"  # tab-separated
echo "done"
