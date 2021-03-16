#!/bin/bash
pushd model
    if [ ! -f "yolov5s.cambricon" ]; then
      wget -O yolov5s.cambricon http://video.cambricon.com/models/CambriconECO/Pytorch_Yolov5_Inference/yolov5s.cambricon
    else
      echo "yolov5s.cambricon exists."
    fi
popd

