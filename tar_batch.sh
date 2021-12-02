#!/bin/bash

for i in `ls *.tar`
do
    mkdir ./${i%.tar}
    tar xvf $i -C ./${i%.tar}
    #echo ${i%.tar}
done
