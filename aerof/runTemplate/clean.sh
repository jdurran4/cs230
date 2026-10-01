#!/bin/bash
# Delete all outputs of this stage (including restart files in references/). In a case
# directory this does not touch steady/: run steady/clean.sh too to redo the whole case.
cd "$(dirname "$0")"

rm -f references/*
rm -f postpro/*
rm -f results/*

rm -f DONE
rm -f *.out
rm -f *~
rm -f *.exo
