#!/bin/bash
./src/yolov5_offline_simple_demo model/yolov5s.cambricon subnet0 0 0 ./data/image.jpg 2 data/label_map_coco.txt
#./src/yolov5_offline_simple_demo model/yolov5s.cambricon subnet0 0 0 ./data/000000000127.jpg  2 data/label_map_coco.txt
