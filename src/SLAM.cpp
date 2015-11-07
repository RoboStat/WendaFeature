//============================================================================
// Name        : SLAM.cpp
// Author      : 
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
#include <vector>
#include <map>
#include <set>
#include <algorithm>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
using namespace std;
using namespace cv;

#define DEBUG 0

string fixedNum(int value, int digits = 3) {
	std::string result;
	while (digits-- > 0) {
		result += ('0' + value % 10);
		value /= 10;
	}
	std::reverse(result.begin(), result.end());
	return result;
}

Scalar RED(0, 0, 255);
Scalar BLUE(255, 0, 0);
Scalar GREEN(0, 255, 0);

//Ptr<ORB> ptr_orb = ORB::create(200, 1.2f, 8, 0, 0, 2, ORB::HARRIS_SCORE, 31);
Ptr<ORB> ptr_orb = ORB::create(300);
Ptr<DescriptorMatcher> ptr_matcher = DescriptorMatcher::create("BruteForce-Hamming");

unsigned int numOfKptsThre = 80;
int landMarkCount = 0;

// Track success metrics
float searchRad = 8;
float singleThre = 50;
float doubleRatio = 0.7;

// all the key points location and description
map<int, vector<KeyPoint>> kptsTrace;
map<int, Mat> kptsDescp;

void evenlyDetect(const Mat& frame, vector<KeyPoint>& kpts) {
	int size = 120;
	int numCol = frame.cols / size;
	int numRow = frame.rows / size;

	for (int c = 0; c < numCol; c++) {
		for (int r = 0; r < numRow; r++) {
			Mat sub = frame(Rect(c * size, r * size, size, size));
			vector<KeyPoint> subKpts;
			ptr_orb->detect(sub, subKpts);

			for (auto it = subKpts.begin(); it != subKpts.end(); it++) {
				//bias the keypoint position
				(it->pt.x) += c * size;
				(it->pt.y) += r * size;
				kpts.push_back(*it);
			}
		}
	}
}

void initKeyFrame(Mat& frame) {
	vector<KeyPoint> initKpts;
	Mat initDescp;
	//evenlyDetect(frame,initKpts);
	//ptr_orb->compute(frame,initKpts,initDescp);
	ptr_orb->detectAndCompute(frame, noArray(), initKpts, initDescp);

	for (auto it = initKpts.begin(); it != initKpts.end(); it++) {
		// init key point trace
		vector<KeyPoint> trace;
		trace.push_back(*it);
		kptsTrace[landMarkCount] = trace;
		// init key point descriptor
		kptsDescp[landMarkCount] = initDescp.row(landMarkCount).clone();
		landMarkCount++;
	}
}

//////////////////////main///////////////////////////
int main() {
	cv::namedWindow("SLAM", WINDOW_NORMAL);

	const std::string video_filename = "resource/video_simple";
	std::string video_ext = video_filename + ".mp4";
	cv::VideoCapture capture(video_ext);
	if (!capture.isOpened()) {
		std::cout << "Video file could not be opened" << std::endl;
		return -1;
	}

	int startFrame = 0;
	for (int i = startFrame; i < 634; i++) {
		//read image
		Mat left_frame = cv::imread("resource/left" + fixedNum(i) + ".jpg");
		Mat right_frame = cv::imread("resource/right" + fixedNum(i) + ".jpg");
		Mat display_frame = left_frame.clone();

		int t1=getTickCount();
		//track points
		if (i == startFrame) {
			initKeyFrame(left_frame);
		} else {
			// detect key points
			vector<KeyPoint> kpts;
			set<int> matchPts;
			//evenlyDetect(left_frame,kpts);
			ptr_orb->detect(left_frame, kpts);

#if DEBUG
			// draw all points
			for (auto d_it = kpts.begin(); d_it != kpts.end(); d_it++)
			circle(display_frame, d_it->pt, 1, RED);
#endif
			// build index map
			multimap<float, int> xind;
			multimap<float, int> yind;
			int kptsCount = 0;
			for (auto it = kpts.begin(); it != kpts.end(); it++) {
				xind.insert(make_pair(it->pt.x, kptsCount));
				yind.insert(make_pair(it->pt.y, kptsCount));
				kptsCount++;
			}

			// query for every point
			auto it = kptsTrace.begin();
			while (it != kptsTrace.end()) {

				Point2f pt = it->second.back().pt;
				map<float, int>::iterator itl, ith;
				set<int> xset, yset;
				// filter in x range
				itl = xind.lower_bound(pt.x - searchRad);
				ith = xind.upper_bound(pt.x + searchRad);
				for (; itl != ith; itl++) {
					xset.insert(itl->second);
				}
				// filter in y range
				itl = yind.lower_bound(pt.y - searchRad);
				ith = yind.upper_bound(pt.y + searchRad);
				for (; itl != ith; itl++) {
					yset.insert(itl->second);
				}
				// intersect both set
				vector<int> boundPtsInd(xset.size());
				vector<int>::iterator ptsEnd;
				ptsEnd = set_intersection(xset.begin(), xset.end(), yset.begin(), yset.end(), boundPtsInd.begin());

				// compute descriptors for sub points
				vector<KeyPoint> subKpts;
				for (auto subit = boundPtsInd.begin(); subit != ptsEnd; subit++) {
					subKpts.push_back(kpts[*subit]);
#if DEBUG
					circle(display_frame,kpts[*subit].pt,1,BLUE);
#endif
				}
#if DEBUG
				//display the result
				circle(display_frame, pt, 1, GREEN);
				imshow("SLAM", display_frame);
				waitKey(0);
#endif

				Mat subDescps;
				Mat descp = kptsDescp[it->first];
				ptr_orb->compute(left_frame, subKpts, subDescps);
				// matching the points
				vector<vector<DMatch>> matches;
				if (subDescps.rows > 0)
					ptr_matcher->knnMatch(descp, subDescps, matches, 2);

				// determine whether a good match
				if (!matches.empty() &&
						((matches[0].size() == 1 && matches[0][0].distance < singleThre)
								|| (matches[0].size() == 2 && (matches[0][0].distance / matches[0][1].distance) < doubleRatio))) {

					//record matched point
					matchPts.insert(boundPtsInd[matches[0][0].trainIdx]);
					// successful to track
					it->second.push_back(subKpts[matches[0][0].trainIdx]);
					kptsDescp[it->first] = subDescps.row(matches[0][0].trainIdx).clone();
					it++;
#if DEBUG
					cout << "success!" <<"size:" << matches[0].size()
					<< "dist:" << matches[0][0].distance;
					if(matches[0].size()==2) {
						cout << "ratio:" << matches[0][0].distance / matches[0][1].distance <<endl;
					}
#endif
				} else {
					// fail to track
					kptsDescp.erase(it->first);
					kptsTrace.erase(it++);
				}
			} // end of each last key point

			// check number of tracked points
			if (kptsTrace.size() < numOfKptsThre) {
				//insert yet unmatched points
				vector<KeyPoint> unMatchPts;
				for (unsigned int npt = 0; npt < kpts.size(); npt++) {
					if (matchPts.find(npt) == matchPts.end())
						unMatchPts.push_back(kpts[npt]);
				}

				Mat unMatchDescp;
				ptr_orb->compute(left_frame, unMatchPts, unMatchDescp);
				int rowCount=0;
				for (auto it = unMatchPts.begin(); it != unMatchPts.end(); it++) {
					// init key point trace
					vector<KeyPoint> trace;
					trace.push_back(*it);
					kptsTrace[landMarkCount] = trace;
					// init key point descriptor
					kptsDescp[landMarkCount] = unMatchDescp.row(rowCount++).clone();
					landMarkCount++;
				}
			}

		} // end tracking loop
		int t2=getTickCount();

		//state estimation

		//draw the result
		for (auto trace_it = kptsTrace.begin(); trace_it != kptsTrace.end(); trace_it++) {
			//draw each trace
			KeyPoint lastPt = trace_it->second.at(0);
			for (auto point_it = trace_it->second.begin(); point_it != trace_it->second.end(); point_it++) {
				line(display_frame, lastPt.pt, point_it->pt, RED);
				lastPt = *point_it;
			}
		}

		cout << i << ":" << "landmarks:" << kptsTrace.size() << endl;
		cout <<"time:" << float(t2-t1)/getTickFrequency() << endl;

		//display the result
		imshow("SLAM", display_frame);
		waitKey(0);

		//save to video
		//imwrite("output/s3/"+fixedNum(i)+".jpg",display_frame);
	}

	return 0;
}
