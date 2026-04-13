#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status
set -e

# Compile simulacion.cpp into the executable 'simulacion'
g++ -std=c++17 -fopenmp simulacion.cpp $(pkg-config --cflags --libs opencv4) -o simulacion

echo "Compilation successful: ./simulacion"
