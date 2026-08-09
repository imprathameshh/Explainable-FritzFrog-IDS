#!/usr/bin/env bash
# Phase 3 — download the CSE-CIC-IDS2018 processed CSVs into data/raw/.
#
# Precondition for the cleaning pipeline. Requires the AWS CLI. No AWS account
# is needed: the bucket allows unsigned requests (--no-sign-request).
#
# Tip: for a first run, do NOT sync everything (~6.5 GB). Copy one smaller CSV
# to test the pipeline end-to-end, then sync the full set once cleaning works,
# e.g.:
#   aws s3 cp --no-sign-request --region ca-central-1 \
#     "s3://cse-cic-ids2018/Processed Traffic Data for ML Algorithms/Friday-02-03-2018_TrafficForML_CICFlowMeter.csv" \
#     data/raw/
#
# Run from the project root: bash src/00_download_data.sh
set -euo pipefail

DEST="data/raw"
mkdir -p "$DEST"

if ! command -v aws >/dev/null 2>&1; then
  echo "AWS CLI not found. Install it first: https://aws.amazon.com/cli/" >&2
  exit 1
fi

aws s3 sync --no-sign-request --region ca-central-1 \
  "s3://cse-cic-ids2018/Processed Traffic Data for ML Algorithms/" \
  "$DEST/"

echo "Download complete. Files in $DEST:"
ls -1 "$DEST"
