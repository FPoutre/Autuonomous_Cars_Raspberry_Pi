#!/bin/bash

echo "Base Model"
python3 benchmarks.py -m ../LaneFollowingModel/base/model.tflite -n 5000
echo ""
echo "Pruned Model"
python3 benchmarks.py -m ../LaneFollowingModel/pruned/model.tflite -n 5000
echo ""
echo "Float16 Quantized Model"
python3 benchmarks.py -m ../LaneFollowingModel/f16/model.tflite -n 5000
echo ""
echo "Dynamic Range Quantized Model"
python3 benchmarks.py -m ../LaneFollowingModel/dq/model.tflite -n 5000
echo ""
echo "Integer (with float fallback) Quantized Model"
python3 benchmarks.py -m ../LaneFollowingModel/intq/model.tflite -n 5000
