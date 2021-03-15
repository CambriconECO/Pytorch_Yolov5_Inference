/*
All modification made by Cambricon Corporation: © 2020 Cambricon Corporation
All rights reserved.
All other contributions:
Copyright (c) 2014--2020, the respective contributors
All rights reserved.
For the list of contributors go to https://github.com/BVLC/caffe/blob/master/CONTRIBUTORS.md
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of Intel Corporation nor the names of its contributors
      may be used to endorse or promote products derived from this software
      without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "processor.hpp"

void writeVisualizeBBox(
    const vector<cv::Mat>& images,
    const vector<vector<vector<float>>> detections,
    const vector<string>& labelToDisplayName, 
    const vector<string>& imageNames,
    int input_dim) {
  // Retrieve detections.
  for (int i = 0; i < images.size(); ++i) {
    if (imageNames[i] == "null") continue;
    cv::Mat image;
    image = images[i];
    vector<vector<float>> result = detections[i];
    std::string name = imageNames[i];
    int positionMap = imageNames[i].rfind("/");
    if (positionMap > 0 && positionMap < imageNames[i].size()) {
      name = name.substr(positionMap + 1);
    }
    positionMap = name.find(".");
    if (positionMap > 0 && positionMap < name.size()) {
      name = name.substr(0, positionMap);
    }
    string filename = "result/"+name + ".txt";
    std::ofstream fileMap(filename);
    float scaling_factors = std::min(
        static_cast<float>(input_dim) / static_cast<float>(images[i].cols),
        static_cast<float>(input_dim) / static_cast<float>(images[i].rows));
    for (int j = 0; j < result.size(); j++) {
      result[j][0] =
          result[j][0]  -
          static_cast<float>(input_dim - scaling_factors * image.cols) / 2.0;
      result[j][2] =
          result[j][2]  -
          static_cast<float>(input_dim - scaling_factors * image.cols) / 2.0;
      result[j][1] =
          result[j][1]  -
          static_cast<float>(input_dim - scaling_factors * image.rows) / 2.0;
      result[j][3] =
          result[j][3]  -
          static_cast<float>(input_dim - scaling_factors * image.rows) / 2.0;

      for (int k = 0; k < 4; k++) {
        result[j][k] = result[j][k] / scaling_factors;
       // cout << result[j][k] << " ";
      }
     // cout << endl;
    }

    for (int j = 0; j < result.size(); j++) {
      result[j][0] = result[j][0] < 0 ? 0 : result[j][0];
      result[j][2] = result[j][2] < 0 ? 0 : result[j][2];
      result[j][1] = result[j][1] < 0 ? 0 : result[j][1];
      result[j][3] = result[j][3] < 0 ? 0 : result[j][3];
      result[j][0] = result[j][0] > image.cols ? image.cols : result[j][0];
      result[j][2] = result[j][2] > image.cols ? image.cols : result[j][2];
      result[j][1] = result[j][1] > image.rows ? image.rows : result[j][1];
      result[j][3] = result[j][3] > image.rows ? image.rows : result[j][3];
    }
    for (int j = 0; j < result.size(); j++) {
      int x0 = static_cast<int>(result[j][0]);
      int y0 = static_cast<int>(result[j][1]);
      int x1 = static_cast<int>(result[j][2]);
      int y1 = static_cast<int>(result[j][3]);
      //cout << "(" <<x0 << "," << y0 << ")" << "(" << y0 << "," << y1 << ")" << endl;
      cv::Point p1(x0, y0);
      cv::Point p2(x1, y1);
      cv::rectangle(image, p1, p2, cv::Scalar(0, 0, 255), 2);
      stringstream ss;
      ss << round(result[j][4] * 1000) / 1000.0;
      std::string str = labelToDisplayName[static_cast<int>(result[j][5])] + ":" + ss.str();
      //std::string str = ss.str();
      cv::Point p5(x0, y0 - 2);
     // cv::rectanle(image, p1, p2, )
      cv::putText(image, str, p5, cv::FONT_HERSHEY_SIMPLEX, 0.6,
                  cv::Scalar(255, 0, 0), 2);
      //std::cout << "it is "  << result[j][5] << ":" << result[j][4] << std::endl;
      fileMap << labelToDisplayName[static_cast<int>(result[j][5])] << " "
              << ss.str() << " "
              << static_cast<float>(result[j][0])  << " "
              << static_cast<float>(result[j][1])  << " "
              << static_cast<float>(result[j][2])  << " "
              << static_cast<float>(result[j][3])  << " "
              << image.cols << " " << image.rows << std::endl;
    }
    fileMap.close();
    stringstream ss;
    string outFile;
    ss <<  "result/yolov5_" << name << ".jpg";
    ss >> outFile;
    cv::imwrite(outFile.c_str(), image);
  }
}

void readLabels(string filename, vector<string>& labels) {
  std::ifstream file(filename);
  if (file.fail())
    std::cerr << "failed to open labels file!";

  std::string line;
  while (getline(file, line)) {
    labels.push_back(line);
  }
  file.close();
}

using std::vector;
using std::string;
/*vector<vector<vector<float>>> getResults(float* outputData,
                                         int dimNumm,
                                         int *dimValues) {
  vector<vector<vector<float>>> detections;

  // BangOp implementation
  float max_limit = 1;
  float min_limit = 0;
  float input_size = 640;
  int batchSize = dimValues[0];
  int count = dimValues[3];
  for (int i = 0; i < batchSize; i++) {
    int num_boxes = static_cast<int>(outputData[i * count]);
    vector<vector<float>> batch_box;
    for (int k = 0; k < num_boxes; k++) {
      int index = i * count + 64 + k * 7;
      vector<float> single_box;
      float bl = std::max(
          min_limit, std::min(max_limit, outputData[index + 3]/input_size));  // x1
      float br = std::max(
          min_limit, std::min(max_limit, outputData[index + 5]/input_size));  // x2
      float bt = std::max(
          min_limit, std::min(max_limit, outputData[index + 4]/input_size));  // y1
      float bb = std::max(
          min_limit, std::min(max_limit, outputData[index + 6]/input_size));  // y2
      single_box.push_back(bl);
      single_box.push_back(bt);
      single_box.push_back(br);
      single_box.push_back(bb);
      single_box.push_back(outputData[index + 2]);
      single_box.push_back(outputData[index + 1]);
      for(auto s:single_box)
        cout << s << " ";
      cout << endl;
      if ((br - bl) > 0 && (bb - bt) > 0) {
        batch_box.push_back(single_box);
      }
    }
    detections.push_back(batch_box);
  }
  return detections;
}*/

vector<vector<vector<float>>> getResults(float* outputData,
                                         int dimNumm,
                                         int *dimValues) {
  vector<vector<vector<float>>> detections;

  int batchSize = dimValues[0];
  int count = dimValues[3];
  for (int i = 0; i < batchSize; i++) {
    int num_boxes = static_cast<int>(outputData[i * count]);
    vector<vector<float>> batch_box;
    for (int k = 0; k < num_boxes; k++) {
      int index = i * count + 64 + k * 7;
      vector<float> single_box;
      float bl = outputData[index + 3];
      float br = outputData[index + 5];
      float bt = outputData[index + 4];
      float bb = outputData[index + 6];
      single_box.push_back(bl);
      single_box.push_back(bt);
      single_box.push_back(br);
      single_box.push_back(bb);
      single_box.push_back(outputData[index + 2]);
      single_box.push_back(outputData[index + 1]);
    //  for(auto s:single_box)
    //    cout << s << " ";
    //  cout << endl;
      if ((br - bl) > 0 && (bb - bt) > 0) {
        batch_box.push_back(single_box);
      }
    }
    detections.push_back(batch_box);
  }
  return detections;
}

