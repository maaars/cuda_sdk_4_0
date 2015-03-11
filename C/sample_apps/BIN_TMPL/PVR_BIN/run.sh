#!/bin/sh 

echo "generating data..."
./Gen data 500000 rank

./PageViewRank data

rm data
