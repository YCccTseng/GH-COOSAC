#pragma once
#ifndef METHOD_H
#define METHOD_H
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

vector<int> method_interval(int f, vector<double>& first, vector<double>& second, bool isLength);

vector<int> GH_filter(Mat original, Mat target, vector<Point2f> original_match_pt, vector<Point2f> target_match_pt);

vector<double> GH_COOSAC(Mat original, Mat target, vector<Point2f> original_match_pt, vector<Point2f> target_match_pt, vector<int> ground_truth, string path, int times);

#endif // !METHOD_H
