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
    LOGD("Java_com_example_documentscanner_Filter_rotate -- BEGIN");
    cv::Mat &mIn = *(cv::Mat *) addrMatIn;
    cv::Mat mInGray;
    cv::cvtColor(mIn, mInGray, COLOR_BGR2GRAY);
    cv::Mat &mOut = *(cv::Mat *) addrMatOut;
    mOut = mIn.clone();

    LOGD("Java_com_example_documentscanner_Filter_rotate -- END");
    int blockSize = 2;
    int apertureSize = 3;
    double k = 0.04;
    int thresh = 200;
    Mat dst = Mat::zeros(mIn.size(), CV_32FC1);
    cornerHarris(mInGray, dst, blockSize, apertureSize, k);
    Mat dst_norm, dst_norm_scaled;
    normalize(dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
    convertScaleAbs(dst_norm, dst_norm_scaled);
    for (int i = 0; i < dst_norm.rows; i++) {
        for (int j = 0; j < dst_norm.cols; j++) {
            if ((int) dst_norm.at<float>(i, j) > thresh) {
                circle(dst_norm_scaled, Point(j, i), 5, Scalar(0), 2, 8, 0);
            }
        }
    }
    mOut = dst_norm_scaled.clone();
}



