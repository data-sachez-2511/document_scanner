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
        (JNIEnv *, jclass, jlong addrMatIn, jlong addrMatOut, jboolean clockwise){
    LOGD("Java_com_example_documentscanner_Filter_rotate -- BEGIN");
    cv::Mat& mIn = *(cv::Mat*)addrMatIn;
    cv::Mat& mOut = *(cv::Mat*)addrMatOut;
    cv::transpose(mIn, mOut);
    cv::flip(mOut, mOut, clockwise ? 1 : 0);
    LOGD("Java_com_example_documentscanner_Filter_rotate -- END");
}

extern "C" JNIEXPORT void JNICALL Java_com_example_documentscanner_Filter_cornerDetect
        (JNIEnv *, jclass, jlong addrMatIn, jlongArray points){
    LOGD("Java_com_example_documentscanner_Filter_rotate -- BEGIN");
    cv::Mat& mIn = *(cv::Mat*)addrMatIn;
    cv::Mat& mOut = *(cv::Mat*)addrMatOut;
    cv::transpose(mIn, mOut);
    cv::flip(mOut, mOut, clockwise ? 1 : 0);
    LOGD("Java_com_example_documentscanner_Filter_rotate -- END");
    // Converting the color image into grayscale
    cvtColor(image, gray, cv::COLOR_BGR2GRAY);

    // Detecting corners
    output = Mat::zeros(image.size(), CV_32FC1);
    cornerHarris(gray, output, 2, 3, 0.04);

    // Normalizing
    normalize(output, output_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
    convertScaleAbs(output_norm, output_norm_scaled);
}



