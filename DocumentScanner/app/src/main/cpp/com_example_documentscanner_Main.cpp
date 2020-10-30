//https://pullrequest.opencv.org/buildbot/export/opencv_releases/master-contrib_pack-contrib-android/20200821-041002--11257/
//#include <com_asav_processimage_MainActivity.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ximgproc.hpp>
#include<opencv2/features2d/features2d.hpp>

#include <string>
#include <vector>
#include <jni.h>
#include <android/log.h>

#define LOG_TAG "DocumentScanner"
#define LOGD(...) ((void)__android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__))

using namespace std;
using namespace cv;

extern "C" JNIEXPORT void JNICALL Java_com_example_documentscanner_Filter_rotate
        (JNIEnv *, jclass, jlong addrMatIn, jlong addrMatOut, jboolean clockwise) {
    LOGD("Java_com_example_documentscanner_Filter_rotate -- BEGIN");
    cv::Mat &mIn = *(cv::Mat *) addrMatIn;
    cv::Mat &mOut = *(cv::Mat *) addrMatOut;
    cv::transpose(mIn, mOut);
    cv::flip(mOut, mOut, clockwise ? 1 : 0);
    LOGD("Java_com_example_documentscanner_Filter_rotate -- END");
}

extern "C" JNIEXPORT void JNICALL Java_com_example_documentscanner_Filter_cornerDetect
        (JNIEnv *, jclass, jlong addrMatIn, jlong addrMatOut) {
    LOGD("Java_com_example_documentscanner_Filter_cornerDetect -- BEGIN");
    cv::Mat &mIn = *(cv::Mat *) addrMatIn;
    cv::Mat mInGray, bGray;
    if(mIn.type() != 0){
        cv::cvtColor(mIn, mInGray, COLOR_BGR2GRAY);
    }
    else{
        mInGray = mIn.clone();
    }
    LOGD("Java_com_example_documentscanner_Filter_cornerDetect -- cvtcolor");
    bGray = cv::Mat(mInGray.rows, mInGray.cols, mInGray.type());
    LOGD("tyt");
    cv::bilateralFilter(mInGray, bGray, 5, 75, 75);
    LOGD("tyt1");
    cv::Mat &mOut = *(cv::Mat *) addrMatOut;
    mOut = bGray.clone();

    LOGD("Java_com_example_documentscanner_Filter_rotate -- END");
    int thresh = 200;
    Mat dst = Mat::zeros(mIn.size(), CV_32FC1);
    cornerHarris(bGray, dst, 2, 3, 0.04);
    Mat out = mIn.clone();
    float max = 0;
    for (int i = 0; i < dst.rows; i++) {
        for (int j = 0; j < dst.cols; j++) {
            if(dst.at<float>(i, j) > max)
                max = dst.at<float>(i, j);
        }
    }
    std::vector<cv::Point2f> corners;
    for (int i = 0; i < dst.rows; i++) {
        for (int j = 0; j < dst.cols; j++) {
            if(dst.at<float>(i, j) > max * 0.01)
                corners.emplace_back(Point2f(i, j));
        }
    }
    std::vector<cv::Point2f> anchors_corners = {
            cv::Point2f(0,0),
            cv::Point2f(0, dst.cols - 1),
            cv::Point2f(dst.rows - 1, 0),
            cv::Point2f(dst.rows - 1, dst.cols - 1),
    };
    std::vector<cv::Point2f> result_corners;
    for(int i=0;i<anchors_corners.size();i++){
        auto min_dist = cv::norm(anchors_corners[i] - corners[0]);
        auto min_point = corners[0];
        auto min_index = 0;
        for(int j=1;j<corners.size();j++){
            auto dist = cv::norm(anchors_corners[i] - corners[j]);
            if(dist < min_dist){
                min_dist = dist;
                min_point = corners[j];
                min_index = j;
            }
        }
        corners.erase(corners.begin() + min_index);
        result_corners.push_back(min_point);
    }
    if(result_corners.size() == 4) {

        auto t = cv::getPerspectiveTransform(result_corners, anchors_corners);
        cv::warpPerspective(mIn, mOut, t, mOut.size());
    }
}

