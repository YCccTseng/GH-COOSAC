#pragma once
#ifndef DRAW_H
#define DRAW_H
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;


void draw(Mat ori_img, Mat tar_img);

void draw_img(Mat ori_img, Mat tar_img, vector<Point2f>& tiny_src_pts, vector<Point2f>& tiny_dst_pts);

#endif // !DRAW_H
