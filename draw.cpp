#include<opencv2\core.hpp>
#include<opencv2\highgui.hpp>
#include<opencv2\imgcodecs.hpp>
#include<opencv2\imgproc.hpp>
#include <iostream>
#include <vector>

#include "draw.h"
#include "calculate.h"

using namespace cv;
using namespace std;


void draw(Mat ori_img, Mat tar_img) {

	Mat concate_result;
	hconcat(ori_img, tar_img, concate_result);
	imshow("concate_result", concate_result);
	waitKey(0);
}


void draw_img(Mat _ori_img, Mat _tar_img, vector<Point2f>& tiny_src_pts, vector<Point2f>& tiny_dst_pts) {

	Mat ori_img, tar_img;
	_ori_img.copyTo(ori_img);
	_tar_img.copyTo(tar_img);

	for (int i = 0; i < tiny_src_pts.size(); i++) {
		Point ori_circle(tiny_src_pts[i].x, tiny_src_pts[i].y);
		Point tar_circle(tiny_dst_pts[i].x, tiny_dst_pts[i].y);
		Scalar color(0, 0, 255);
		circle(ori_img, ori_circle, 1, color, 3);
		circle(tar_img, tar_circle, 1, color, 3);

	}

	Mat concate_result;
	hconcat(ori_img, tar_img, concate_result);
	int width = ori_img.size().width;
	for (int i = 0; i < tiny_src_pts.size(); i++) {
		Point ori_circle(tiny_src_pts[i].x, tiny_src_pts[i].y);
		Point tar_circle(tiny_dst_pts[i].x + width, tiny_dst_pts[i].y);
		Scalar color(rand() % 256, rand() % 256, rand() % 256);
		line(concate_result, ori_circle, tar_circle, color, 1);
	}

	imshow("concate_result", concate_result);
	waitKey(0);
}
