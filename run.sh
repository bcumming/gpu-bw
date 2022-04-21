#!/bin/bash

echo "running on -- $(hostname)"
export NUMA_NODE=3
numactl --cpunodebind=$NUMA_NODE --membind=$NUMA_NODE ./bandwidth
