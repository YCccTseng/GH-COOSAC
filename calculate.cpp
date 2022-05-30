#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath> 
#include <numeric>
#include <opencv2/opencv.hpp>

#include "calculate.h"

using namespace std;
using namespace cv;


void cal_pitch(vector<double>& input, int pitch, vector<double>& pitch_list) {

	double min_input = round(*min_element(input.begin(), input.end()) * 10) / 10.0;
	double max_input = round(*max_element(input.begin(), input.end()) * 10) / 10.0;

	pitch_list.push_back(min_input - pitch);
	while (min_input < (max_input + pitch)) {
		pitch_list.push_back(min_input);
		min_input += pitch;
	}
}


void sort_index(vector<int>& frequency) {
	vector<pair<int, int>> vp;
	for (int i = 0; i < frequency.size(); i++)
		vp.push_back(make_pair(frequency[i], i));

	sort(vp.rbegin(), vp.rend());

	for (int i = 0; i < vp.size(); i++)
		frequency[i] = vp[i].second;
}


void sort_weight(vector<pair<int, double>>& save_weight) {
	vector<int> index(save_weight.size());
	vector<pair<double, int>> weight;
	for (int i = 0; i < save_weight.size(); i++) {
		index[i] = save_weight[i].first;
		weight.push_back(make_pair(save_weight[i].second, i));
	}

	sort(weight.rbegin(), weight.rend());

	for (int i = 0; i < weight.size(); i++) {
		save_weight[i].first = index[weight[i].second];
		save_weight[i].second = weight[i].first;
	}
}


// calculate frequency
// pitch_list = store bins
// frequency = return frequency sort(less->more) index
vector<int> cal_frequency_descending(vector<double>& input, vector<double>& pitch_list) {

	// as np.histogram in python 
	int pitch_size = pitch_list.size();
	vector<int> frequency(pitch_size - 1, 0);
	for (auto index : input) {
		for (int i = 0; i < pitch_size - 1; i++) {
			if (pitch_list[i] < index && index <= pitch_list[i + 1]) {
				frequency[i]++;
			}
		}
	}

	// as np.argsort in python
	sort_index(frequency);
	return frequency;
}


void save_frequency(int f, vector<double>& input, vector<double>& pitch_list, vector<int>& frequency, vector<int>& save) {
	for (int i = 0; i < input.size(); i++) {
		for (int j = 0; j < f; j++) {
			if (pitch_list[frequency[j]] < input[i] && input[i] <= pitch_list[frequency[j] + 1])
				save.push_back(i);
		}
	}
}


void save_interval(int f, vector<double>& input, vector<double>& pitch_list, vector<int>& frequency, vector<int>& save, vector<double>& weight) {

	int tmp = frequency[0];
	vector<int> save_left, save_center, save_right;

	for (int i = 0; i < input.size(); i++) {
		if (tmp - f < 0) {
			if (pitch_list[0] < input[i] && input[i] <= pitch_list[tmp + f])
				save_center.push_back(i);
			if (pitch_list[tmp + f] < input[i] && input[i] <= pitch_list[tmp + (f + 1)])
				save_right.push_back(i);
		}
		else if (tmp + (f + 1) >= pitch_list.size()) {
			if (pitch_list[tmp - f] < input[i] && input[i] <= pitch_list[tmp])
				save_left.push_back(i);
			if (pitch_list[tmp] < input[i] && input[i] <= pitch_list[pitch_list.size() - 1])
				save_center.push_back(i);
		}
		else {
			if (pitch_list[tmp - f] < input[i] && input[i] <= pitch_list[tmp])
				save_left.push_back(i);
			if (pitch_list[tmp] < input[i] && input[i] <= pitch_list[tmp + f])
				save_center.push_back(i);
			if (pitch_list[tmp + f] < input[i] && input[i] <= pitch_list[tmp + (f + 1)])
				save_right.push_back(i);
		}
	}

	save.insert(save.begin(), save_left.begin(), save_left.end());
	save.insert(save.begin(), save_center.begin(), save_center.end());
	save.insert(save.begin(), save_right.begin(), save_right.end());
	
	weight.resize(save.size());
	for (int i = 0; i < save.size(); i++) {
		if (i < save_left.size())
			weight[i] = (double)save_left.size() / (double)save.size();
		else if (i < save_left.size() + save_center.size())
			weight[i] = (double)save_center.size() / (double)save.size();
		else
			weight[i] = (double)save_right.size() / (double)save.size();
	}
}


void save_frequency_interval(int fre, int inter, vector<double>& input, vector<double>& pitch_list, vector<int>& frequency, vector<int>& save) {
	for (int i = 0; i < fre; i++) {
		int tmp = frequency[i];
		for (int j = 0; j < input.size(); j++) {
			if (tmp - inter < 0) {
				if (pitch_list[0] < input[j] && input[j] <= pitch_list[tmp + (inter + 1)])
					save.push_back(j);
			}
			else if (tmp + (inter + 1) >= pitch_list.size()) {
				if (pitch_list[tmp - inter] < input[j] && input[j] <= pitch_list[pitch_list.size() - 1])
					save.push_back(j);
			}
			else {
				if (pitch_list[tmp - inter] < input[j] && input[j] <= pitch_list[tmp + (inter + 1)])
					save.push_back(j);
			}
		}
	}

	sort(save.begin(), save.end());
	save.erase(unique(save.begin(), save.end()), save.end());
}


void cal_method_pt(vector<int>& method, vector<Point2f>& ori_match_pt, vector<Point2f>& tar_match_pt, vector<Point2f>& ori, vector<Point2f>& tar) {

	for (int i = 0; i < method.size(); i++) {
		ori[i] = ori_match_pt[method[i]];
		tar[i] = tar_match_pt[method[i]];
	}
}


void cal_area(Point2f& ori_pt0, Point2f& ori_pt1, Point2f& tar_pt0, Point2f& tar_pt1, double& area) {
	double x1 = ori_pt0.x;
	double y1 = ori_pt0.y;
	double x2 = ori_pt1.x;
	double y2 = ori_pt1.y;
	double X1 = tar_pt0.x;
	double Y1 = tar_pt0.y;
	double X2 = tar_pt1.x;
	double Y2 = tar_pt1.y;

	area = abs((x1 * y2 + x2 * Y2 + X2 * Y1 + X1 * y1 - y1 * x2 - y2 * X2 - Y2 * X1 - Y1 * x1) / 2);
}


vector<int> indexto01(vector<int>& input, int allpoint) {

	vector<int> output(allpoint, 0);

	for (auto index : input)
		output[index] = 1;

	return output;
}


classificationPair classification_model(vector<int> ground_truth, vector<int> method) {
	classificationPair classification;
	double tp = 0;
	double tn = 0;
	double fp = 0;
	double fn = 0;

	for (int i = 0; i < ground_truth.size(); i++) {
		if (ground_truth[i] == 1) {
			if (method[i] == 1)
				tp += 1;
			else
				fn += 1;
		}
		else {
			if (method[i] == 1)
				fp += 1;
			else
				tn += 1;
		}
	}
	classification.TP = tp;
	classification.TN = tn;
	classification.FP = fp;
	classification.FN = fn;

	//cout << "TP is " << tp << " and TN is " << tn << " and FP is " << fp << " and FN is " << fn << endl;
	return classification;
}