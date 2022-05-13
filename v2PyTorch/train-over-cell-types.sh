#!/bin/bash

# /path/data
# no trailing / !
DATA_DIR=$1
# /path/save_root
# no trailing / !
SAVE_ROOT=$2
# loop through all subdirectories in $DATA_DIR
for d in $DATA_DIR/*/ ; do
    # full path of dir including trailing /
    CELL_TYPE=$(basename $d)
    MODEL_NAME=${CELL_TYPE}_attchrome
    CONTAINING_RESULT_DIR=$SAVE_ROOT/$CELL_TYPE/$MODEL_NAME
    echo "On cell type: $CELL_TYPE"
    echo $d
    # directory where checkpoints will be saved
    echo $CONTAINING_RESULT_DIR
    mkdir -p $CONTAINING_RESULT_DIR
    python3 train.py --save_root $SAVE_ROOT \
        --data_root $DATA_DIR \
        --epochs 30 \
        --cell_type $CELL_TYPE \
        > $CONTAINING_RESULT_DIR/${MODEL_NAME}.log
    echo "Finished training cell type: $CELL_TYPE"
    echo
done