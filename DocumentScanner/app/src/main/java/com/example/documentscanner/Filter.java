package com.example.documentscanner;

import android.util.Log;

import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import org.opencv.utils.Converters;

import java.util.ArrayList;
import java.util.Arrays;

public class Filter {
    private final String TAG = "FILTER";
    private boolean conveer;
    Mat imageConveer;

    private static native void rotate(long matAddrIn, long matAddrOut, boolean clockwise);
    private static native void cornerDetect(long matAddrIn, long matAddrOut);

    public Filter(boolean conveer) {
        System.loadLibrary("OpenCvProcessImageLib");
        this.conveer = conveer;
    }

    public void setImageConveer(Mat image){
        imageConveer = image;
    }

    public Mat getImageConveer(){
        return imageConveer;
    }

    public Mat binarization(Mat image) {
        if (!conveer || imageConveer == null) {
            imageConveer = image;
        }
        Log.d(TAG, "image type: " + imageConveer.type());
        if(imageConveer.type() != CvType.CV_8UC1)
            Imgproc.cvtColor(imageConveer, imageConveer, Imgproc.COLOR_RGB2GRAY);
        Imgproc.threshold(imageConveer, imageConveer, 0, 255, Imgproc.THRESH_BINARY + Imgproc.THRESH_OTSU);
        return imageConveer;
    }

    public Mat rotate(Mat image, boolean clockwise){
        if (!conveer || imageConveer == null) {
            imageConveer = image;
        }
        Mat rotatedImage = new Mat(imageConveer.cols(), imageConveer.rows(), imageConveer.type());
        rotate(imageConveer.getNativeObjAddr(), rotatedImage.getNativeObjAddr(), clockwise);
        imageConveer = rotatedImage;
        return imageConveer;
    }

    public Mat cornerDetectAuto(Mat image){
        if (!conveer || imageConveer == null) {
            imageConveer = image;
        }
        Mat imageCorners = new Mat();
        cornerDetect(imageConveer.getNativeObjAddr(), imageCorners.getNativeObjAddr());
        return imageCorners;
    }

    public Mat perspectiveTransform(Mat image, ArrayList<Point> corners) {
        org.opencv.core.Point centroid = new org.opencv.core.Point(0, 0);
        for (org.opencv.core.Point point : corners) {
            centroid.x += point.x;
            centroid.y += point.y;
        }
        centroid.x /= corners.size();
        centroid.y /= corners.size();

        sortCorners(corners, centroid);
        if (!conveer || imageConveer == null) {
            imageConveer = image;
        }
        Mat srcPoints = Converters.vector_Point2f_to_Mat(corners);

        Mat destPoints = Converters.vector_Point2f_to_Mat(Arrays.asList(new Point(0, 0),
                new Point(imageConveer.cols(), 0),
                new Point(imageConveer.cols(), imageConveer.rows()),
                new Point(0, imageConveer.rows())));

        Mat transformation = Imgproc.getPerspectiveTransform(srcPoints, destPoints);
        Imgproc.warpPerspective(imageConveer, imageConveer, transformation, imageConveer.size());

        corners.clear();
        return imageConveer;
    }

    private void sortCorners(ArrayList<Point> corners, org.opencv.core.Point center) {
        ArrayList<org.opencv.core.Point> top = new ArrayList<org.opencv.core.Point>();
        ArrayList<org.opencv.core.Point> bottom = new ArrayList<org.opencv.core.Point>();

        for (int i = 0; i < corners.size(); i++) {
            if (corners.get(i).y < center.y)
                top.add(corners.get(i));
            else
                bottom.add(corners.get(i));
        }

        double topLeft = top.get(0).x;
        int topLeftIndex = 0;
        for (int i = 1; i < top.size(); i++) {
            if (top.get(i).x < topLeft) {
                topLeft = top.get(i).x;
                topLeftIndex = i;
            }
        }

        double topRight = 0;
        int topRightIndex = 0;
        for (int i = 0; i < top.size(); i++) {
            if (top.get(i).x > topRight) {
                topRight = top.get(i).x;
                topRightIndex = i;
            }
        }

        double bottomLeft = bottom.get(0).x;
        int bottomLeftIndex = 0;
        for (int i = 1; i < bottom.size(); i++) {
            if (bottom.get(i).x < bottomLeft) {
                bottomLeft = bottom.get(i).x;
                bottomLeftIndex = i;
            }
        }

        double bottomRight = bottom.get(0).x;
        int bottomRightIndex = 0;
        for (int i = 1; i < bottom.size(); i++) {
            if (bottom.get(i).x > bottomRight) {
                bottomRight = bottom.get(i).x;
                bottomRightIndex = i;
            }
        }

        org.opencv.core.Point topLeftPoint = top.get(topLeftIndex);
        org.opencv.core.Point topRightPoint = top.get(topRightIndex);
        org.opencv.core.Point bottomLeftPoint = bottom.get(bottomLeftIndex);
        org.opencv.core.Point bottomRightPoint = bottom.get(bottomRightIndex);

        corners.clear();
        corners.add(topLeftPoint);
        corners.add(topRightPoint);
        corners.add(bottomRightPoint);
        corners.add(bottomLeftPoint);
    }
}
