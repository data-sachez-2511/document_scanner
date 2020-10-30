package com.example.documentscanner;

import android.util.Log;

import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import org.opencv.utils.Converters;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.opencv.core.CvType.CV_8U;

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

    public Mat contrast(Mat image){
        if (!conveer || imageConveer == null) {
            imageConveer = image;
        }
        Log.d(TAG, "image type: " + imageConveer.type());
        if(imageConveer.type() != CvType.CV_8UC1)
            Imgproc.cvtColor(imageConveer,imageConveer, Imgproc.COLOR_RGB2GRAY);
        Mat out=new Mat();
        Core.MinMaxLocResult minMaxLocRes = Core.minMaxLoc(imageConveer);
        double minVal = minMaxLocRes.minVal;//+20;
        double maxVal = minMaxLocRes.maxVal;//-50;
        imageConveer.convertTo(out, CV_8U, 255.0 / (maxVal - minVal), -minVal * 255.0 / (maxVal - minVal));
        return imageConveer;
    }

    private byte saturate(double val) {
        int iVal = (int) Math.round(val);
        iVal = iVal > 255 ? 255 : (iVal < 0 ? 0 : iVal);
        return (byte) iVal;
    }

    public Mat gammaCorrection(Mat image){
        if (!conveer || imageConveer == null) {
            imageConveer = image;
        }
        Log.d(TAG, "image type: " + imageConveer.type());
        if(imageConveer.type() != CvType.CV_8UC1)
            Imgproc.cvtColor(imageConveer,imageConveer, Imgproc.COLOR_RGB2GRAY);
        double gammaValue = 1.3;
        Mat lookUpTable = new Mat(1, 256, CV_8U);
        byte[] lookUpTableData = new byte[(int) (lookUpTable.total() * lookUpTable.channels())];
        for (int i = 0; i < lookUpTable.cols(); i++) {
            lookUpTableData[i] = saturate(Math.pow(i / 255.0, gammaValue) * 255.0);
        }
        lookUpTable.put(0, 0, lookUpTableData);

        Core.LUT(imageConveer, lookUpTable, imageConveer);
        return imageConveer;
    }

    public Mat equalizeHisto(Mat image){
        if (!conveer || imageConveer == null) {
            imageConveer = image;
        }
        Log.d(TAG, "image type: " + imageConveer.type());
        if(imageConveer.type() != CvType.CV_8UC1)
            Imgproc.cvtColor(imageConveer,imageConveer, Imgproc.COLOR_RGB2GRAY);
        Imgproc.equalizeHist(imageConveer, imageConveer);
        return imageConveer;
    }

    public float fft(Mat image){
        if (!conveer || imageConveer == null) {
            imageConveer = image;
        }
        Log.d(TAG, "image type: " + imageConveer.type());
        if(imageConveer.type() != CvType.CV_8UC1)
            Imgproc.cvtColor(imageConveer,imageConveer, Imgproc.COLOR_RGB2GRAY);
        imageConveer.convertTo(imageConveer, CvType.CV_64FC1);

        int m = Core.getOptimalDFTSize(imageConveer.rows());
        int n = Core.getOptimalDFTSize(imageConveer.cols()); // on the border

        Mat padded = new Mat(new Size(n, m), CvType.CV_64FC1); // expand input

        Core.copyMakeBorder(imageConveer, padded, 0, m - imageConveer.rows(), 0,
                n - imageConveer.cols(), Core.BORDER_CONSTANT);

        List<Mat> planes = new ArrayList<Mat>();
        planes.add(padded);
        planes.add(Mat.zeros(padded.rows(), padded.cols(), CvType.CV_64FC1));
        Mat complexI = new Mat();
        Core.merge(planes, complexI); // Add to the expanded another plane with zeros
        Mat complexI2=new Mat();
        Core.dft(complexI, complexI2); // this way the result may fit in the source matrix

        // compute the magnitude and switch to logarithmic scale
        // => log(1 + sqrt(Re(DFT(I))^2 + Im(DFT(I))^2))
        Core.split(complexI2, planes); // planes[0] = Re(DFT(I), planes[1] =Im(DFT(I))
        Mat spectrum = new Mat();
        Core.magnitude(planes.get(0), planes.get(1), spectrum);
        Core.add(spectrum, new Scalar(1), spectrum);
        Core.log(spectrum, spectrum);
        float mean = 0;
        int count = 0 ;
        for(int i=0;i<spectrum.rows();i++){
            for(int j=0;j<spectrum.cols();j++){
                mean += spectrum.get(i, j)[0];
                count ++;
            }
        }
        mean = mean / count;
        Log.d(TAG, "fft: " + mean);
        return mean;
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
