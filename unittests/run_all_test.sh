#!/usr/bin/env bash

test_single_file(){
    test_file=$1
    echo "Testing $test_file ..."
    cp $test_file .
    file_name=$(basename $test_file)
    python $file_name
    rm $file_name
    echo "Done."
}


set -e

test_single_file ./unittests/test_dl4mt.py
test_single_file ./unittests/test_transformer.py
test_single_file ./unittests/test_bpe.py

