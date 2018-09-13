#!/usr/bin/env bash

test_single_file(){
    test_file=$1
    echo "Testing $test_file ..."
    cp ./unittests/$test_file .
    python $test_file
    rm $test_file
    echo "Done."
}


set -e

test_single_file test_dl4mt.py
test_single_file test_transformer.py
test_single_file test_bpe.py
test_single_file test_reload_from_checkpoints.py
