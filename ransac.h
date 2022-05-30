#pragma once
#ifndef RANSAC_H
#define RANSAC_H
#include <vector>

using namespace std;

struct homoPair {
	Mat H;
	vector<int> mask;
	double _final_iteration;
	double _init_iteration;
	double _tiny_iteration;
	double _tiny_iteration_sum;
	double _area_fail;

	double _tiny_HAV_time;
	double _init_HAV_time;
	double _tiny_random_4_time;
	double _tiny_area_time;
	double _reduced_random_tiny_time;
	double _draw_tiny;

	vector<vector<double>> _testtt;
} typedef homoPair;


void runKernel(float _m1[4][2], float _m2[4][2], Mat& H);

void computeError(vector<Point2f>& src_pts, vector<Point2f>& dst_pts, Mat& H, vector<double>& proj_error_lst);

void findInliers(vector<double>& err, double projErr, double& inlier_count, vector<int>& inlier_mask);

void RANSACUpdateNumIters(double confidence, double outlier_rate, double& maxIters);

void pick_4_correspondence(vector<Point2f>& tiny_src_pts, vector<Point2f>& tiny_dst_pts, vector<int>& picked_idx);

void pick_tiny_corresponence(vector<Point2f>& reduced_src_pts, vector<Point2f>& reduced_dst_pts, vector<int>& tiny_correspondence, vector<Point2f>& tiny_src_pts, vector<Point2f>& tiny_dst_pts);

void check_tiny_confidence(vector<Point2f>& reduced_src_pts, vector<Point2f>& reduced_dst_pts, vector<int>& tiny_correspondence, vector<Point2f>& tiny_src_pts, vector<Point2f>& tiny_dst_pts);

void tiny_COOSAC(vector<Point2f>& tiny_src_pts, vector<Point2f>& tiny_dst_pts, double max_projErr, vector<int>& picked_idx);

void init_COOSAC(vector<Point2f>& init_src_pts, vector<Point2f>& init_dst_pts, double max_projErr, double max_confidence, double& max_iteration, vector<int>& best_mask);

homoPair COOSAC(vector<Point2f>& init_src_pts, vector<Point2f>& init_dst_pts, vector<Point2f>& reduced_src_pts, vector<Point2f>& reduced_dst_pts, double max_confidence, double max_iteration, double max_projErr, double tiny_size);

#endif // !RANSAC_H
