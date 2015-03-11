#!/bin/sh 

echo "generating data..."
./Gen data 500000 count

./PageViewCount data

rm data
