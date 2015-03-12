#!/bin/sh 

echo "generating data..."
./Gen data 250000 count

./PageViewCount data

rm data
