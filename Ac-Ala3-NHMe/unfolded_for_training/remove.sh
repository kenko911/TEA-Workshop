#!/bin/bash

# Loop through all subdirectories in the current directory
for dir in */; do
    cd ${dir}/
         rm -rf *.bin *.npz train_* test_*
	 rm -rf energies.json forces.json stresses.json extract_model.py state_attr.pt 
    cd ../	
done
