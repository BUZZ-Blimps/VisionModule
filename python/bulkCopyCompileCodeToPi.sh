#!/bin/bash

# Define temporary file to store counter
temp_file="/tmp/tmp_bulk_copy_compile_code";

# Define command to decrement counter
command_dec_counter="counter=\$(< $temp_file) && echo \$((counter-1)) > $temp_file";

# Initialize temporary counter to 0
bash -c "echo 0 > $temp_file";

# Assume we are using all blimps 1,...,6
blimp_ids=($(seq 1 6));

# If arguments are passed, assume they are specific blimps to use
if (($# > 0)); then
    blimp_ids=("$@");
fi

# Get number of blimp ids
num_blimps=${#blimp_ids[@]}

# Print blimp ids to use
printf "Flashing blimps: ";
for i in $(seq 0 $((num_blimps-1)));
do
    blimp_id=${blimp_ids[$i]}
    printf "%d " $blimp_id
done
printf "\n"

# Copy compile vision to blimps
for i in $(seq 0 $((num_blimps-1)));
do
    # Increment temporary counter
    bash -c "counter=\$(< $temp_file) && echo \$((counter+1)) > $temp_file";

    # Copy and compile code, then decrement counter
    blimp_id=${blimp_ids[$i]}
    bash -c "./copyCompileCodeToPi.sh 192.168.0.10$blimp_id && $command_dec_counter" &
done

# Wait for counter to reach 0 before exiting
waiting=true;
while [ $waiting == true ]; do
    eval "counter=\$(< $temp_file)"
    if (($counter == 0)); then
        waiting=false;
    else
        sleep 0.1
    fi
done