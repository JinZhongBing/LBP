#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

#include<bitset>//等价模式中计算跳变次数用

void olbp_(InputArray _src, OutputArray _dst);
void getCircularLBPFeature(InputArray _src, OutputArray _dst, int radius, int neighbors);
void getRotationInvariantLBPFeature(InputArray _src, OutputArray _dst, int radius, int neighbors);
void getUniformPatternLBPFeature(InputArray _src, OutputArray _dst, int radius, int neighbors);
int getHopTimes(int n);



Mat histc_(const Mat& src, int minVal = 0, int maxVal = 255, bool normed = false);
Mat histc(InputArray _src, int minVal, int maxVal, bool normed);
Mat spatial_histogram(InputArray _src, int numPatterns, int grid_x, int grid_y, bool normed);



double chiSquare(InputArray _H1, InputArray _H2, int method = CV_COMP_CHISQR);
