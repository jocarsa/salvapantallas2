#!/bin/bash

# Define the source file and output executable name
SOURCE_FILE="main.cpp"
OUTPUT_NAME="stars"

# Compile the code using g++ with OpenCV libraries
g++ $SOURCE_FILE -o $OUTPUT_NAME `pkg-config --cflags --libs opencv4`

# Check if compilation was successful
if [ $? -eq 0 ]; then
    echo "Compilation successful. Executable created: $OUTPUT_NAME"
else
    echo "Compilation failed."
fi

