#include <opencv2/core.hpp>
#include <opencv2/core/types_c.h>
#include <opencv2/core/core_c.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <iostream>
#include <vector>
#include <numeric> 
#include <typeinfo>
#include <algorithm>
#include <cmath>
#include <execution>

#include "calculate.h"
#include "ransac.h"

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;


vector<vector<int>> combin{ {0,1},{0,2},{0,3},{1,2},{1,3},{2,3} };
Mat tiny_H, final_H;
double best_final_inlier_ratio, best_tiny_inlier_ratio;
double tiny_confidence, final_confidence;
double final_tiny_confidence = 0.995; //original = 0.7 or 0.995
double final_iteration, init_iteration, tiny_iteration, tiny_iteration_sum, draw_tiny;
double area_fail;
bool isActivate;
RNG rng((unsigned)time(NULL));
//static int area_thres = 1000;

// TIME
double tiny_HAV_time, init_HAV_time;
double tiny_random_4_time, tiny_area_time;
double reduced_random_tiny_time;

vector<vector<double>> testtt;
vector<double> temp;


void runKernel(float _m1[4][2], float _m2[4][2], Mat& H) {

	Mat m1 = Mat(4, 2, CV_32FC1, _m1);
	Mat m2 = Mat(4, 2, CV_32FC1, _m2);

	const Point2f* M = m1.ptr<Point2f>();
	const Point2f* m = m2.ptr<Point2f>();

	double LtL[9][9], W[9][1], V[9][9];
	Mat _LtL(9, 9, CV_64F, &LtL[0][0]);
	Mat matW(9, 1, CV_64F, W);
	Mat matV(9, 9, CV_64F, V);
	Mat _H0(3, 3, CV_64F, V[8]);
	Mat _Htemp(3, 3, CV_64F, V[7]);
	Point2d cM(0, 0), cm(0, 0), sM(0, 0), sm(0, 0);

	for (int i = 0; i < 4; i++) {
		cm.x += m[i].x; cm.y += m[i].y;
		cM.x += M[i].x; cM.y += M[i].y;
	}

	cm.x /= 4; cm.y /= 4;
	cM.x /= 4; cM.y /= 4;

	for (int i = 0; i < 4; i++) {
		sm.x += fabs(m[i].x - cm.x);
		sm.y += fabs(m[i].y - cm.y);
		sM.x += fabs(M[i].x - cM.x);
		sM.y += fabs(M[i].y - cM.y);
	}

	sm.x = 4 / sm.x; sm.y = 4 / sm.y;
	sM.x = 4 / sM.x; sM.y = 4 / sM.y;

	double invHnorm[9] = { 1. / sm.x, 0, cm.x, 0, 1. / sm.y, cm.y, 0, 0, 1 };
	double Hnorm2[9] = { sM.x, 0, -cM.x * sM.x, 0, sM.y, -cM.y * sM.y, 0, 0, 1 };
	Mat _invHnorm(3, 3, CV_64FC1, invHnorm);
	Mat _Hnorm2(3, 3, CV_64FC1, Hnorm2);

	_LtL.setTo(Scalar::all(0));
	for (int i = 0; i < 4; i++)
	{
		double x = (m[i].x - cm.x) * sm.x, y = (m[i].y - cm.y) * sm.y;
		double X = (M[i].x - cM.x) * sM.x, Y = (M[i].y - cM.y) * sM.y;
		double Lx[] = { X, Y, 1, 0, 0, 0, -x * X, -x * Y, -x };
		double Ly[] = { 0, 0, 0, X, Y, 1, -y * X, -y * Y, -y };
		int j, k;
		for (j = 0; j < 9; j++)
			for (k = j; k < 9; k++)
				LtL[j][k] += Lx[j] * Lx[k] + Ly[j] * Ly[k];
	}
	completeSymm(_LtL);

	eigen(_LtL, matW, matV);
	_Htemp = _invHnorm * _H0;
	_H0 = _Htemp * _Hnorm2;
	_H0.convertTo(H, _H0.type(), 1. / _H0.at<double>(2, 2));
}


void computeError(vector<Point2f>& src_pts, vector<Point2f>& dst_pts, Mat& H, vector<double>& proj_error_lst) {

	double count = src_pts.size();
	Mat homo_coordi_src_pts_lst(3, count, CV_64FC1);
	double Hf[] = { H.at<double>(0, 0), H.at<double>(0, 1), H.at<double>(0, 2), H.at<double>(1, 0), H.at<double>(1, 1), H.at<double>(1, 2), H.at<double>(2, 0), H.at<double>(2, 1) };

	for (int i = 0; i < count; i++) {
		double ww = (double)1 / (Hf[6] * src_pts[i].x + Hf[7] * src_pts[i].y + (double)1);
		double dx = (Hf[0] * src_pts[i].x + Hf[1] * src_pts[i].y + Hf[2]) * ww - dst_pts[i].x;
		double dy = (Hf[3] * src_pts[i].x + Hf[4] * src_pts[i].y + Hf[5]) * ww - dst_pts[i].y;

		proj_error_lst[i] = dx * dx + dy * dy;
	}
}


void findInliers(vector<double>& err, double projErr, double& inlier_count, vector<int>& inlier_mask) {

	double threshold = projErr * projErr;

	for (int i = 0; i < err.size(); i++) {
		double f = err[i] <= threshold;
		inlier_mask[i] = f;
		inlier_count += f;
	}
}


void RANSACUpdateNumIters(double confidence, double outlier_rate, double& maxIters) {
	double num = (double)1 - confidence;
	double denom = (double)1 - pow((double)1 - outlier_rate, 4);

	num = log(num);
	denom = log(denom);

	if (denom >= 0 || (-num) >= maxIters * (-denom))
		maxIters = maxIters;
	else
		maxIters = cvRound(num / denom);
}


void pick_4_correspondence(vector<Point2f>& tiny_src_pts, vector<Point2f>& tiny_dst_pts, vector<int>& picked_idx) {

	int tiny_size = tiny_src_pts.size();

	clock_t time1_1, time1_2, time2_1, time2_2;

	bool stop = false;
	while (stop == false) {

		// get hypothesis
		time1_1 = clock();

		for (int i = 0; i < 4; i++) {
			int idx_i;
			for (idx_i = rng.uniform(0, (int)tiny_size); find(picked_idx.begin(), picked_idx.end(), idx_i) != picked_idx.end(); idx_i = rng.uniform(0, (int)tiny_size)) {}
			picked_idx[i] = idx_i;
		}
		
		time1_2 = clock();
		tiny_random_4_time += ((double)time1_2 - (double)time1_1) * 0.001;

		final_iteration++;
		tiny_iteration++;

		// check area size
		time2_1 = clock();
		stop = true;
		for (auto i : combin) {
			int index0 = picked_idx[i[0]];
			int index1 = picked_idx[i[1]];
			double area;
			cal_area(tiny_src_pts[index0], tiny_src_pts[index1], tiny_dst_pts[index0], tiny_dst_pts[index1], area);
			if (area < area_thres) {
				area_fail++;
				stop = false;
				break;
			}
		}

		time2_2 = clock();
		tiny_area_time += ((double)time2_2 - (double)time2_1) * 0.001;
	}
}


void pick_tiny_corresponence(vector<Point2f>& reduced_src_pts, vector<Point2f>& reduced_dst_pts, vector<int>& tiny_correspondence, vector<Point2f>& tiny_src_pts, vector<Point2f>& tiny_dst_pts) {

	double tiny_size = tiny_correspondence.size();
	double reduced_count = reduced_src_pts.size();

	draw_tiny++;

	for (int i = 0; i < tiny_size; i++) {
		int idx_i;
		for (idx_i = rng.uniform(0, (int)reduced_count); find(tiny_correspondence.begin(), tiny_correspondence.end(), idx_i) != tiny_correspondence.end(); idx_i = rng.uniform(0, (int)reduced_count)) {}
		tiny_correspondence[i] = idx_i;
	}
	sort(tiny_correspondence.begin(), tiny_correspondence.end());

	cal_method_pt(tiny_correspondence, reduced_src_pts, reduced_dst_pts, tiny_src_pts, tiny_dst_pts);
}


void check_tiny_confidence(vector<Point2f>& reduced_src_pts, vector<Point2f>& reduced_dst_pts, vector<int>& tiny_correspondence, vector<Point2f>& tiny_src_pts, vector<Point2f>& tiny_dst_pts) {

	tiny_confidence = 1 - pow((1 - pow(best_tiny_inlier_ratio, 4)), tiny_iteration);
	//cout << "Current iteration in tiny is " << tiny_iteration << ", current tiny inlier ratio is " << tiny_confidence << ", current confidence in tiny is " << tiny_confidence << endl;

	if (tiny_confidence > final_tiny_confidence) {
		isActivate = true;
		tiny_iteration = 0;
		best_tiny_inlier_ratio = 0;
		tiny_confidence = 0;
		pick_tiny_corresponence(reduced_src_pts, reduced_dst_pts, tiny_correspondence, tiny_src_pts, tiny_dst_pts);
	}
}


void tiny_COOSAC(vector<Point2f>& tiny_src_pts, vector<Point2f>& tiny_dst_pts, double max_projErr, vector<int>& picked_idx) {

	int tiny_size = tiny_src_pts.size();
	double inlier_ratio = 0;

	tiny_iteration_sum++;

	// get H
	float _m1[4][2], _m2[4][2];
	for (int i = 0; i < 4; i++) {
		_m1[i][0] = tiny_src_pts[picked_idx[i]].x;
		_m1[i][1] = tiny_src_pts[picked_idx[i]].y;
		_m2[i][0] = tiny_dst_pts[picked_idx[i]].x;
		_m2[i][1] = tiny_dst_pts[picked_idx[i]].y;
	}

	runKernel(_m1, _m2, tiny_H);

	vector<double> projection_error(tiny_src_pts.size());
	computeError(tiny_src_pts, tiny_dst_pts, tiny_H, projection_error);

	double inlier_count = 0;
	vector<int> mask(projection_error.size());
	findInliers(projection_error, max_projErr, inlier_count, mask);

	inlier_ratio = inlier_count / tiny_size;
	if (inlier_ratio > best_tiny_inlier_ratio) {
		best_tiny_inlier_ratio = inlier_ratio;
	}
}


void init_COOSAC(vector<Point2f>& init_src_pts, vector<Point2f>& init_dst_pts, double max_projErr, double max_confidence, double& max_iteration, vector<int>& best_mask) {

	double init_count = init_src_pts.size();

	init_iteration++;

	vector<double> proj_error_lst(init_src_pts.size());
	computeError(init_src_pts, init_dst_pts, tiny_H, proj_error_lst);

	double inlier_count = 0;
	vector<int> mask(proj_error_lst.size());
	findInliers(proj_error_lst, max_projErr, inlier_count, mask);

	//cout << "Current best inlier ratio is " << best_final_inlier_ratio << ",  current inlier ratio is " << inlier_count / init_count << endl;
	if ((inlier_count / init_count) > best_final_inlier_ratio) {
		final_H = tiny_H;
		best_final_inlier_ratio = inlier_count / init_count;
		best_mask = mask;
	}

	RANSACUpdateNumIters(max_confidence, (init_count - inlier_count) / init_count, max_iteration);
}


homoPair COOSAC(vector<Point2f>& init_src_pts, vector<Point2f>& init_dst_pts, vector<Point2f>& reduced_src_pts, vector<Point2f>& reduced_dst_pts, double max_confidence, double max_iteration, double max_projErr, double tiny_size) {

	// ===== Global =====
	final_iteration = 0, init_iteration = 0, tiny_iteration = 0, tiny_iteration_sum = 0, draw_tiny = 0;
	area_fail = 0;
	best_final_inlier_ratio = 0, best_tiny_inlier_ratio = 0;

	tiny_random_4_time = 0, tiny_area_time = 0;
	reduced_random_tiny_time = 0;
	tiny_HAV_time = 0, init_HAV_time = 0;
	// ===== Global =====

	homoPair homo_mask;
	double init_count = init_src_pts.size();
	vector<int> best_mask(init_count, 0);

	// TIME
	clock_t time1_1, time1_2, time5_1, time5_2, time6_1, time6_2;

	vector<int> tiny_correspondence(round(reduced_src_pts.size() * tiny_size));
	vector<Point2f> tiny_src(tiny_correspondence.size()), tiny_dst(tiny_correspondence.size());
	pick_tiny_corresponence(reduced_src_pts, reduced_dst_pts, tiny_correspondence, tiny_src, tiny_dst);

	while (final_iteration < max_iteration) {

		isActivate = false;
		vector<int> picked_idx(4);

		pick_4_correspondence(tiny_src, tiny_dst, picked_idx);

		time1_1 = clock();
		tiny_COOSAC(tiny_src, tiny_dst, max_projErr, picked_idx);
		time1_2 = clock();
		tiny_HAV_time += ((double)time1_2 - (double)time1_1) * 0.001;

		time6_1 = clock();
		check_tiny_confidence(reduced_src_pts, reduced_dst_pts, tiny_correspondence, tiny_src, tiny_dst);
		time6_2 = clock();
		reduced_random_tiny_time += ((double)time6_2 - (double)time6_1) * 0.001;


		// test hypothesis
		time5_1 = clock();
		if (isActivate == true) 
			init_COOSAC(init_src_pts, init_dst_pts, max_projErr, max_confidence, max_iteration, best_mask);
		time5_2 = clock();
		init_HAV_time += ((double)time5_2 - (double)time5_1) * 0.001;

		final_confidence = 1 - pow((1 - pow(best_final_inlier_ratio, 4)), final_iteration);
		//cout << "Current iteration is " << final_iteration << ", tiny iteration is " << tiny_iteration << ", best tiny inlier ratio is " << best_tiny_inlier_ratio << ", tiny confidence is " << tiny_confidence << endl;

	}

	homo_mask.H = final_H;
	homo_mask.mask = best_mask;
	homo_mask._final_iteration = final_iteration;
	homo_mask._init_iteration = init_iteration;
	homo_mask._tiny_iteration = tiny_iteration;
	homo_mask._tiny_iteration_sum = tiny_iteration_sum;
	homo_mask._area_fail = area_fail;

	homo_mask._tiny_HAV_time = tiny_HAV_time;
	homo_mask._init_HAV_time = init_HAV_time;
	homo_mask._tiny_random_4_time = tiny_random_4_time;
	homo_mask._tiny_area_time = tiny_area_time;
	homo_mask._reduced_random_tiny_time = reduced_random_tiny_time;
	homo_mask._draw_tiny = draw_tiny;

	return homo_mask;
}

