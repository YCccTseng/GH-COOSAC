#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <iostream>
#include <vector>
#include <ctime>
#include <typeinfo>
#include <fstream>
#include <numeric> 
#include <windows.h>

#include "method.h"
#include "draw.h"
#include "calculate.h"
#include "ransac.h"

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;


int default_ransacReprojThreshold = 5;
int default_maxIters = 200000;
double default_confidence = 0.995;


vector<int> method_interval(int f, vector<double>& first, vector<double>& second, bool isLength) {
	vector<double> pitch_first;
	vector<int> fre_first;
	if (isLength) {
		cal_pitch(first, len_pitch, pitch_first);
		fre_first = cal_frequency_descending(first, pitch_first);
	}
	else {
		cal_pitch(first, angle_pitch, pitch_first);
		fre_first = cal_frequency_descending(first, pitch_first);
	}

	vector<int> save_first;
	vector<double> weight_first;
	save_interval(f, first, pitch_first, fre_first, save_first, weight_first);

	vector<double> filter_second(save_first.size());
	for (int i = 0; i < save_first.size(); i++)
		filter_second[i] = second[save_first[i]];

	vector<double> pitch_second;
	vector<int> fre_second;
	if (isLength) {
		cal_pitch(filter_second, angle_pitch, pitch_second);
		fre_second = cal_frequency_descending(filter_second, pitch_second);
	}
	else {
		cal_pitch(filter_second, len_pitch, pitch_second);
		fre_second = cal_frequency_descending(filter_second, pitch_second);
	}

	vector<int> save_second;
	vector<double> weight_second;
	save_interval(f, filter_second, pitch_second, fre_second, save_second, weight_second);

	vector<pair<int, double>> save_weight(save_second.size());
	for (int i = 0; i < save_second.size(); i++) {
		save_weight[i].first = save_first[save_second[i]];
		save_weight[i].second = weight_first[save_second[i]] + weight_second[i];
	}

	sort_weight(save_weight);
	/*for (auto i : save_weight) 
		cout << i.first << " " << i.second << endl;*/

	vector<int> save_index(save_weight.size());
	for (int i = 0; i < save_weight.size(); i++) 
		save_index[i] = save_weight[i].first;

	return save_index;
}


vector<int> GH_filter(Mat original, Mat target, vector<Point2f> original_match_pt, vector<Point2f> target_match_pt) {
	// Vi
	vector<Point2f> vec(original_match_pt.size());
	for (int i = 0; i < original_match_pt.size(); i++) {
		vec[i] = Point2f((target_match_pt[i].x + 1080 - original_match_pt[i].x), (target_match_pt[i].y - original_match_pt[i].y));
	}

	// length of Vi
	vector<double> vector_len(vec.size());
	for (int i = 0; i < vec.size(); i++) {
		double length = sqrt(pow(vec[i].x, 2) + pow(vec[i].y, 2));
		vector_len[i] = length;
	}

	// angle of Vi
	vector<double> vector_ang(vec.size());
	for (int i = 0; i < vec.size(); i++) {
		double theta = atan2(vec[i].y, vec[i].x);
		theta = abs(theta * 180.0 / CV_PI);
		vector_ang[i] = theta;
	}


	// ---------- GH Start ----------

	vector<int> save_ang_method_2 = method_interval(1, vector_ang, vector_len, false);

	return save_ang_method_2;
	// ---------- GH End ----------
}


// only one correspondence in tiny can activate init
vector<double> GH_COOSAC(Mat original, Mat target, vector<Point2f> original_match_pt, vector<Point2f> target_match_pt, vector<int> ground_truth, string path, int times) {

	/*ofstream ofs;
	ofs.open(path + ".csv");*/
	
	//vector<double> constant_list = { 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 };
	vector<double> constant_list = { 0.2 };

	double reduce_all = 0, time_all = 0, inlier_count_all = 0, inlier_rate_all = 0;
	double recall_all = 0, precision_all = 0, f1_score_all = 0, specificity_all = 0;
	double final_iteration_all = 0, init_iteration_all = 0, tiny_iteration_all = 0, draw_tiny_all = 0, area_fail_all = 0;

	for (int a = 0; a < constant_list.size(); a++) {

		double constant = constant_list[a];
		/*ofs << "Reduced" << "," << "Constant" << ","
			<< "Final iteration" << "," << "Init iteration" << "," << "Tiny iteration" << "," << "Draw tiny" << "," << "Random 4" << "," << "Tiny Random" << ","
			<< "Area fail" << "," << "Area" << "," << "Tiny  HAV" << "," << "Init HAV" << "," << "Total time" << ","
			<< "Inlier count" << "," << "Inlier rate" << "," << "Recall" << "," << "Precision" << "," << "F1-score" << "," << "Specificity" << endl;*/

		for (int i = 0; i < times; i++) {

			// ---------- SGH_COOSAC Start ---------- 
			//cout << "---------- SGH_COOSAC ----------" << endl;
			clock_t start, stop;
			start = clock();
			vector<int> output = GH_filter(original, target, original_match_pt, target_match_pt);
			vector<Point2f> ori(output.size()), tar(output.size());
			cal_method_pt(output, original_match_pt, target_match_pt, ori, tar);
			//draw_img(original, target, ori, tar);
			auto homo_mask = COOSAC(target_match_pt, original_match_pt, tar, ori, default_confidence, default_maxIters, default_ransacReprojThreshold, constant);
			stop = clock();

			double total_time = ((double)stop - (double)start) / 1000;
			//double total_time = SGH_time + HAV_time;

			Mat H = homo_mask.H;
			vector<int> mask = homo_mask.mask;
			double final_iteration = homo_mask._final_iteration;
			double init_iteration = homo_mask._init_iteration;
			double tiny_iteration = homo_mask._tiny_iteration;
			double tiny_iteration_sum = homo_mask._tiny_iteration_sum;
			double draw_tiny = homo_mask._draw_tiny;
			double area_fail = homo_mask._area_fail;

			double tiny_HAV_time = homo_mask._tiny_HAV_time;
			double init_HAV_time = homo_mask._init_HAV_time;
			double tiny_random_4_time = homo_mask._tiny_random_4_time;
			double tiny_area_time = homo_mask._tiny_area_time;
			double reduced_random_tiny_time = homo_mask._reduced_random_tiny_time;

			//cout << "Matches of second method is " << output.size() << ", Constant is " << constant << endl;
			//cout << "Final iteration is " << final_iteration << endl;
			//cout << "Tiny HAV time is " << tiny_HAV_time << " and init HAV time is " << init_HAV_time << endl;
			//cout << "Fail num is " << area_fail << " and area time is " << tiny_area_time << endl;
			//cout << "Total time is " << total_time << endl;
			/*ofs << method.size() << "," << constant << ","
				<< final_iteration << "," << init_iteration << "," << tiny_iteration_sum << "," << draw_tiny << "," 
				<< tiny_random_4_time << "," << reduced_random_tiny_time << ","
				<< area_fail << "," << tiny_area_time << "," << tiny_HAV_time << "," << init_HAV_time << ","  << total_time << ",";*/
			// ---------- SGH_COOSAC End ---------- 

			// ---------- Classification Start ----------
			//cout << "---------- Classification ----------" << endl;
			double inlier_count = accumulate(mask.begin(), mask.end(), 0);
			//cout << "Inlier count is " << inlier_count << " and  Inlier rate is " << inlier_count / (double)original_match_pt.size() << endl;
			//ofs << inlier_count << "," << inlier_count / (double)original_match_pt.size() << ",";

			auto classification = classification_model(ground_truth, mask);
			double TP = classification.TP; double TN = classification.TN;
			double FP = classification.FP; double FN = classification.FN;
			double recall = TP / (TP + FN);
			double precision = TP / (TP + FP);
			double specificity = TN / (TN + FP);
			double f1_score = 2 * precision * recall / (precision + recall);
			//cout << "Recall is " << recall << " and Precision is " << precision << endl;
			//cout << "F1-score is " << f1_score << " and Specificity is " << specificity << endl;
			//ofs << recall << "," << precision << "," << f1_score << "," << specificity << endl;
			// ---------- Classification End ----------
			//cout << "==========================================================================" << endl;

			reduce_all += output.size();
			time_all += total_time;
			inlier_count_all += inlier_count;
			inlier_rate_all += inlier_count / (double)original_match_pt.size();
			recall_all += recall;
			precision_all += precision;
			f1_score_all += f1_score;
			specificity_all += specificity;

			final_iteration_all += final_iteration;
			init_iteration_all += init_iteration;
			tiny_iteration_all += tiny_iteration_sum;
			draw_tiny_all += draw_tiny;
			area_fail_all += area_fail;
		}

		//ofs << "\n\n" << endl;

		vector<double> accuracy;
		accuracy.push_back(reduce_all / times);
		accuracy.push_back(time_all / times);
		accuracy.push_back(inlier_rate_all / times);
		accuracy.push_back(recall_all / times);
		accuracy.push_back(precision_all / times);
		accuracy.push_back(f1_score_all / times);
		return accuracy;
	}
}
