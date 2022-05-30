#pragma once
#ifndef CALCULATE_H
#define CALCULATE_H
#include <vector>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

static int len_pitch = 20;
static int angle_pitch = 5;
static int area_thres = 1000;

//vector<double> cal_pitch(vector<double>input, int pitch);
void cal_pitch(vector<double>& input, int pitch, vector<double>& pitch_list);

//vector<int> sort_index(vector<int> frequency);
void sort_index(vector<int>& frequency);

void sort_weight(vector<pair<int, double>>& save_weight);

vector<int> cal_frequency_descending(vector<double>&input, vector<double>& pitch_list);

//vector<int> save_frequency(int f, vector<double> input, vector<double> pitch_list, vector<int> frequency);
void save_frequency(int f, vector<double>& input, vector<double>& pitch_list, vector<int>& frequency, vector<int>& save);

//vector<int> save_interval(int f, vector<double> input, vector<double> pitch_list, vector<int> frequency);
void save_interval(int f, vector<double>& input, vector<double>& pitch_list, vector<int>& frequency, vector<int>& save, vector<double>& weight);

//vector<int> save_frequency_interval(int fre, int inter, vector<double> input, vector<double> pitch_list, vector<int> frequency);
void save_frequency_interval(int fre, int inter, vector<double>& input, vector<double>& pitch_list, vector<int>& frequency, vector<int>& save);

//struct pointPair {
//	vector<Point2f> ori_pt, tar_pt;
//} typedef pointPair;
//pointPair cal_method_pt(vector<int> method, vector<Point2f> ori_match_pt, vector<Point2f> tar_match_pt);
void cal_method_pt(vector<int>& method, vector<Point2f>& ori_match_pt, vector<Point2f>& tar_match_pt, vector<Point2f>& ori, vector<Point2f>& tar);

//double cal_area(Point2f ori_pt0, Point2f ori_pt1, Point2f tar_pt0, Point2f tar_pt1);
void cal_area(Point2f& ori_pt0, Point2f& ori_pt1, Point2f& tar_pt0, Point2f& tar_pt1, double& area);

vector<int> indexto01(vector<int>& input, int allpoint);

struct classificationPair {
	double TP, TN, FP, FN;
} typedef classificationPair;

classificationPair classification_model(vector<int> ground_truth, vector<int> method);

#endif // !CALCULATE_H
