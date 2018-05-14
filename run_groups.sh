#!/bin/bash

num_intervals=20
if command -v bsub >/dev/null 2>&1; then
    # bsub is present
    prefix=(bsub -W 120:00 -n $num_intervals -R "rusage[mem=3000]" )
    prefix2=(bsub -W 24:00 -n $num_intervals -R "rusage[mem=3000]" )
else
  prefix=()
  prefix2=()
fi

compr3=10
## Keep group size constant (2) but vary compression levels
dir=./results_nrel/groups2
mkdir -p $dir
dir3=./results_nrel/groups3
mkdir -p $dir3
for compr in $(seq 1 1 10); do
  for compr2 in $(seq 1 1 10); do
    command=( "${prefix[@]}" # combine arrays
              python ./analysis.py --data_dir $dir --compression_levels $compr $compr2 --num_intervals $num_intervals --num_rep 20)
    "${command[@]}"
    # repeat with group size 3
    command=( "${prefix[@]}" # combine arrays
              python ./analysis.py --data_dir $dir3 --compression_levels $compr $compr2 $compr3 --num_intervals $num_intervals --num_rep 20)
    "${command[@]}"
    done
done

# Compress always with the same level but increase group size
dir=./results_nrel/groups
mkdir -p $dir
mkdir -p "$dir"_uniform
mkdir -p "$dir"_powerlaw
mkdir -p "$dir"_step
compr=10
param=()
sizes=$(seq 1 1 15)
for s in ${sizes[@]}; do
    param=("${param[@]}" $compr)
    name=()
    printf -v name[0] '_%s' "${param[@]}"
    command=( "${prefix2[@]}" # combine arrays
              python ./analysis.py --data_dir $dir --compression_levels ${param[@]} --num_intervals $num_intervals --num_rep 5)
    "${command[@]}"
    command=( "${prefix2[@]}" # combine arrays
              python ./analysis.py --data_dir "$dir"_uniform --compression_levels ${param[@]} --num_intervals $num_intervals --num_rep 5 --grp_size_distr randint)
    "${command[@]}"
    command=( "${prefix2[@]}" # combine arrays
              python ./analysis.py --data_dir "$dir"_powerlaw --compression_levels ${param[@]} --num_intervals $num_intervals --num_rep 5 --grp_size_distr randint_power_law)
    "${command[@]}"
    command=( "${prefix2[@]}" # combine arrays
              python ./analysis.py --data_dir "$dir"_step --compression_levels ${param[@]} --num_intervals $num_intervals --num_rep 5 --grp_size_distr randint_step)
    "${command[@]}"
done

# vary fraction of grouped agents
mkdir -p "$dir"_fractions
compr=20
fractions=$(seq 0 0.1 1)
for f in ${fractions[@]}; do
  command=( "${prefix2[@]}" # combine arrays
            python ./analysis.py --data_dir "$dir"_fractions --compression_levels $compr $compr --num_intervals $num_intervals --num_rep 5 --group_fraction $f)
  "${command[@]}"
done
