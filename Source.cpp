#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <iostream>
#include <typeinfo>
#include <fstream>
#include <numeric> 

#include "method.h"
#include "ransac.h"
#include "draw.h"

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;


int main() {

	clock_t start, stop;
	vector<string> dataset_1, dataset_2;
	vector<float> adjust_inlier_rate = { 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 };
	double times = 10;

	// "Airport" "Small_Village" "University_Campus"
	string dataset = "Small_Village";

	if (dataset == "Airport") {
		dataset_1 = { "IMG_0061.JPG","IMG_0116.JPG","IMG_0177.JPG","IMG_0282.JPG","IMG_3479.JPG" };
		dataset_2 = { "IMG_0062.JPG","IMG_0117.JPG","IMG_0178.JPG","IMG_0283.JPG","IMG_3480.JPG" };
	}
	else if (dataset == "Small_Village") {
		dataset_1 = { "IMG_0924.JPG","IMG_0970.JPG","IMG_1011.JPG","IMG_1113.JPG","IMG_1204.JPG" };
		dataset_2 = { "IMG_0925.JPG","IMG_0971.JPG","IMG_1012.JPG","IMG_1114.JPG","IMG_1205.JPG" };
	}
	else if (dataset == "University_Campus") {
		dataset_1 = { "IMG_0060.JPG","IMG_0098.JPG","IMG_0172.JPG","IMG_0333.JPG","IMG_0403.JPG" };
		dataset_2 = { "IMG_0061.JPG","IMG_0099.JPG","IMG_0173.JPG","IMG_0334.JPG","IMG_0404.JPG" };
	}
	else {
		cout << "You need to choose a dataset." << endl;
	}


	// Airport
	//vector<string> dataset_1 = { "IMG_0061.JPG","IMG_0116.JPG","IMG_0177.JPG","IMG_0282.JPG","IMG_3479.JPG" };
	//vector<string> dataset_2 = { "IMG_0062.JPG","IMG_0117.JPG","IMG_0178.JPG","IMG_0283.JPG","IMG_3480.JPG" };

	// Small_Village
	//vector<string> dataset_1 = { "IMG_0924.JPG","IMG_0970.JPG","IMG_1011.JPG","IMG_1113.JPG","IMG_1204.JPG" };
	//vector<string> dataset_2 = { "IMG_0925.JPG","IMG_0971.JPG","IMG_1012.JPG","IMG_1114.JPG","IMG_1205.JPG" };

	// University_Campus
	//vector<string> dataset_1 = { "IMG_0060.JPG","IMG_0098.JPG","IMG_0172.JPG","IMG_0333.JPG","IMG_0403.JPG" };
	//vector<string> dataset_2 = { "IMG_0061.JPG","IMG_0099.JPG","IMG_0173.JPG","IMG_0334.JPG","IMG_0404.JPG" };


	ofstream ofs;
	ofs.open(".\\COOSAC\\" + dataset + "\\total.csv");

	ofs << "GT inlier" << "," << "Correspondence" << "," << "Adjust" << "," << "Reduce" << ","
		<< "Total time" << ", " << "Inlier rate" << ","
		<< "Recall" << "," << "Precision" << "," << "F1-score" << endl;


	for (int adjustIdx = 0; adjustIdx < adjust_inlier_rate.size(); adjustIdx++) {

		double GT_inlier_all = 0, correspondence_all = 0, adjust_all = 0, reduce_all = 0,
			time_all = 0, inlier_rate_all = 0, recall_all = 0, precision_all = 0, f1_score_all = 0;

		for (int datasetIdx = 0; datasetIdx < dataset_1.size(); datasetIdx++) {

			Mat original = imread(".\\" + dataset + "\\" + dataset_1[datasetIdx]);
			Mat target = imread(".\\" + dataset + "\\" + dataset_2[datasetIdx]);
			resize(original, original, Size(1080, 720), INTER_LINEAR);
			resize(target, target, Size(1080, 720), INTER_LINEAR);
			//draw(original, target);

			start = clock();
			vector<Point2f> all_original_match_pt, all_target_match_pt;
			vector<int> all_ground_truth;
			vector<int> remove_list;
			fstream file;
			string line;

			file.open(".\\" + dataset + "\\" + dataset_1[datasetIdx].substr(4, 4) + "_pts.csv");
			while (getline(file, line, '\n'))
			{
				istringstream templine(line);
				string data;
				vector<string> tmp;
				while (getline(templine, data, ','))
					tmp.push_back(data);
				all_original_match_pt.push_back(Point2f((stof(tmp[0])), (stof(tmp[1]))));
			}
			file.close();

			file.open(".\\" + dataset + "\\" + dataset_2[datasetIdx].substr(4, 4) + "_pts.csv");
			while (getline(file, line, '\n'))
			{
				istringstream templine(line);
				string data;
				vector<string> tmp;
				while (getline(templine, data, ','))
					tmp.push_back(data);
				all_target_match_pt.push_back(Point2f((stof(tmp[0])), (stof(tmp[1]))));
			}
			file.close();

			file.open(".\\" + dataset + "\\" + dataset_1[datasetIdx].substr(4, 4) + "_" + dataset_2[datasetIdx].substr(4, 4) + ".csv");
			while (getline(file, line, '\n'))
			{
				istringstream templine(line);
				string data;
				while (getline(templine, data, ',')) {
					if (data == "False")
						all_ground_truth.push_back(0);
					else
						all_ground_truth.push_back(1);
				}
			}
			file.close();
			//cout << "all_ground_truth = " << all_ground_truth.size() << endl;

			stringstream stream;
			stream.precision(1);
			stream << fixed;
			stream << adjust_inlier_rate[adjustIdx];
			string str_adjust = stream.str();
			file.open(".\\" + dataset + "\\adjust\\" + dataset_1[datasetIdx].substr(4, 4) + "_" + dataset_2[datasetIdx].substr(4, 4) + "_" + str_adjust + ".csv");
			while (getline(file, line, '\n'))
			{
				istringstream templine(line);
				string data;
				vector<string> tmp;
				while (getline(templine, data, ','))
					tmp.push_back(data);
				remove_list.push_back(stoi(tmp[0]));
			}
			file.close();

			vector<Point2f> original_match_pt(all_original_match_pt), target_match_pt(all_target_match_pt);
			vector<int> ground_truth(all_ground_truth);

			for (int remove_index = remove_list.size() - 1; remove_index >= 0; remove_index--) {
				original_match_pt.erase(original_match_pt.begin() + remove_list[remove_index]);
				target_match_pt.erase(target_match_pt.begin() + remove_list[remove_index]);
				ground_truth.erase(ground_truth.begin() + remove_list[remove_index]);
			}
			double gt_inlier = accumulate(ground_truth.begin(), ground_truth.end(), 0);
			stop = clock();
			//cout << "Time is " << ((double)stop - (double)start) * 0.001 << endl;

			/*cout << dataset_1[datasetIdx].substr(4, 4) << "_" << dataset_2[datasetIdx].substr(4, 4) <<
				": Inlier rate in ground truth = " << adjust_inlier_rate[adjustIdx] << ", size = " << ground_truth.size() << endl;*/

			string path = ".\\COOSAC\\" + dataset + "\\" + dataset_1[datasetIdx].substr(4, 4) + "_" + dataset_2[datasetIdx].substr(4, 4);

			//draw_img(original, target, original_match_pt, target_match_pt);

			vector<double> accuracy = GH_COOSAC(original, target, original_match_pt, target_match_pt, ground_truth, path, times);

			GT_inlier_all += gt_inlier;
			correspondence_all += ground_truth.size();
			adjust_all += adjust_inlier_rate[adjustIdx];
			reduce_all += accuracy[0];
			time_all += accuracy[1];
			inlier_rate_all += accuracy[2];
			recall_all += accuracy[3];
			precision_all += accuracy[4];
			f1_score_all += accuracy[5];
		}

		cout << "GT inlier = " << GT_inlier_all / dataset_1.size() << ", Correspondence = " << correspondence_all / dataset_1.size()
			<< ", Adjust = " << adjust_all / dataset_1.size() << ", Reduce = " << reduce_all / dataset_1.size() << endl
			<< ", Total time = " << time_all / dataset_1.size() << ", Inlier rate = " << inlier_rate_all / dataset_1.size()
			<< ", Recall = " << recall_all / dataset_1.size() << ", Precision = " << precision_all / dataset_1.size() 
			<< ", F1-score = " << f1_score_all / dataset_1.size() << endl;

		ofs << GT_inlier_all / dataset_1.size() << "," << correspondence_all / dataset_1.size() << "," << adjust_all / dataset_1.size() << ","
			<< reduce_all / dataset_1.size() << "," << time_all / dataset_1.size() << "," << inlier_rate_all / dataset_1.size() << ","
			<< recall_all / dataset_1.size() << "," << precision_all / dataset_1.size() << "," << f1_score_all / dataset_1.size() << endl;

		cout << "======================================================" << endl;
	}

	
	return 0;
}
