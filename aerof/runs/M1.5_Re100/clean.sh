#!/bin/bash
# Delete all outputs of this case (including restart files in references/)
cd "$(dirname "$0")"

rm -f references/*
rm -f postpro/*
rm -f results/*

rm -f *.out
rm -f *~
rm -f *.exo
