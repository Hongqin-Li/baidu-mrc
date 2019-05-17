#!/bin/bash

# Since the raw json file is too large, we need to split it into chunks first

# Usage `bash split_raw.sh xxx.json`

DIRECTORY="./data_chunks/"
PREFIX="chunk_" # The output chunks is data_chunks/
LINE_SIZE=1000 # Each chunk file should have at most 1000 lines

split $1 "${DIRECTORY}${PREFIX}" -d -a3 -l$LINE_SIZE --additional-suffix=.json

if (( $? )); then
    echo "Usage: bash split_raw.sh xxx.json"
    exit 1
else 
    echo "Formating raw..."
    python3 format_raw.py --directory=$DIRECTORY
fi
exit 0

