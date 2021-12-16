#!/bin/bash

echo "Base Model"
python -m ../LaneFollowingModel/model.tflite -n 5000
echo ""
echo "Pruned Model"
python -m ../LaneFollowingModel/pruned_model.tflite -n 5000
echo ""
echo "Quantized and Pruned Model"
python -m ../LaneFollowingModel/quantized_and_pruned_model.tflite -n 5000