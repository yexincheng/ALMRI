#!/bin/bash

source activate medsam

cust_func(){
    python eval_bbox_regression.py --num $num --sseed 2024
}

for num in {11..20}
do
    cust_func $num & # Put a function in the background
    sleep 2
done
## Put all cust_func in the background and bash 
## would wait until those are completed 
## before displaying all done message
wait 
echo "All done"
 