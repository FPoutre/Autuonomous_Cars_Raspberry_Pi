#!/bin/bash

echo "Base Model"
python benchmarks.py -m ../LaneFollowingModel/model.tflite -n 5000
echo ""
echo "Pruned Model"
python benchmarks.py -m ../LaneFollowingModel/pruned_model.tflite -n 5000
echo ""
echo "Quantized and Pruned Model"
python benchmarks.py -m ../LaneFollowingModel/quantized_and_pruned_model.tflite -n 5000