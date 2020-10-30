package com.example.documentscanner;

import android.os.Bundle;
import android.widget.ImageView;

import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.Toolbar;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import android.content.Intent;
import android.content.pm.PackageInfo;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;

import androidx.exifinterface.media.ExifInterface;

import android.net.Uri;
import android.os.Bundle;
import android.os.SystemClock;
import android.util.DisplayMetrics;
import android.util.Log;
import android.view.Display;
import android.view.Menu;
import android.view.MenuItem;
import android.view.MotionEvent;
import android.view.View;
import android.view.WindowMetrics;
import android.widget.ImageView;
import android.widget.Toast;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.*;
import org.opencv.features2d.*;
import org.opencv.imgproc.Imgproc;
import org.opencv.utils.Converters;

import java.io.InputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;

public class Main extends AppCompatActivity {
    private ImageView imageView;
    private Filter filter;
    private final int REQUEST_CODE_ASK_MULTIPLE_PERMISSIONS = 124;
    private final String TAG = "Main DocumentScanner";
    private Mat sampledImage = null;
    ArrayList<Point> corners=new ArrayList<org.opencv.core.Point>();
    private boolean isSelectingCorner = false;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        Toolbar toolbar = (Toolbar) findViewById(R.id.my_toolbar);
        setSupportActionBar(toolbar);
        imageView = (ImageView) findViewById(R.id.inputImageView);

        if (!allPermissionsGranted()) {
            ActivityCompat.requestPermissions(this, getRequiredPermissions(), REQUEST_CODE_ASK_MULTIPLE_PERMISSIONS);
        }
    }

    private boolean allPermissionsGranted() {
        for (String permission : getRequiredPermissions()) {
            int status = ContextCompat.checkSelfPermission(this, permission);
            if (ContextCompat.checkSelfPermission(this, permission)
                    != PackageManager.PERMISSION_GRANTED) {
                return false;
            }
        }
        return true;
    }

    private String[] getRequiredPermissions() {
        try {
            PackageInfo info =
                    getPackageManager()
                            .getPackageInfo(getPackageName(), PackageManager.GET_PERMISSIONS);
            String[] ps = info.requestedPermissions;
            if (ps != null && ps.length > 0) {
                return ps;
            } else {
                return new String[0];
            }
        } catch (Exception e) {
            return new String[0];
        }
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        getMenuInflater().inflate(R.menu.toolbar_menu, menu);
        return true;
    }

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            if (status == LoaderCallbackInterface.SUCCESS) {
                Log.i(TAG, "OpenCV loaded successfully");
//                System.loadLibrary("OpenCvProcessImageLib");
//                Log.i(TAG, "After loading all libraries");
                Toast.makeText(getApplicationContext(),
                        "OpenCV loaded successfully",
                        Toast.LENGTH_SHORT).show();
            } else {
                super.onManagerConnected(status);
                Toast.makeText(getApplicationContext(),
                        "OpenCV error",
                        Toast.LENGTH_SHORT).show();
            }
        }
    };

    @Override
    public void onResume() {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_4_0, this, mLoaderCallback);
        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        if (requestCode == REQUEST_CODE_ASK_MULTIPLE_PERMISSIONS) {
            Map<String, Integer> perms = new HashMap<String, Integer>();
            boolean allGranted = true;
            for (int i = 0; i < permissions.length; i++) {
                perms.put(permissions[i], grantResults[i]);
                if (grantResults[i] != PackageManager.PERMISSION_GRANTED)
                    allGranted = false;
            }
            // Check for ACCESS_FINE_LOCATION
            if (!allGranted) {
                // Permission Denied
                Toast.makeText(Main.this, "Some Permission is Denied", Toast.LENGTH_SHORT)
                        .show();
                finish();
            }
        } else {
            super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        }
    }

    private static final int SELECT_PICTURE = 1;

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        isSelectingCorner = item.getItemId() == R.id.action_manuallyCornerSelect;
        Mat sampledImageClone = null;
        if(isImageLoaded()) {
            sampledImageClone = sampledImage.clone();
        }

        switch (item.getItemId()) {
            case R.id.action_openGallery:
                Intent intent = new Intent();
                intent.setType("image/*");
                intent.setAction(Intent.ACTION_GET_CONTENT);
                startActivityForResult(Intent.createChooser(intent, "Select Picture"),
                        SELECT_PICTURE);
                return true;
            case R.id.action_binary:
                Log.d(TAG, "start binarization");
                if(isImageLoaded()) {
                    sampledImageClone = filter.binarization(sampledImageClone);
                    displayImage(sampledImageClone);
                }
                return true;
            case R.id.action_rotate90:
                Log.d(TAG, "start rotate 90");
                if(isImageLoaded()) {
                    sampledImageClone = filter.rotate(sampledImageClone, true);
                    displayImage(sampledImageClone);
                }
                return true;
            case R.id.action_rotate270:
                Log.d(TAG, "start rotate 270");
                if(isImageLoaded()) {
                    sampledImageClone = filter.rotate(sampledImageClone, false);
                    displayImage(sampledImageClone);
                }
                return true;
            case R.id.action_manuallyCornerSelect:
                corners.clear();
                Log.d(TAG, "start manually corner selecting");
                if(isImageLoaded()) {
                    manuallyCornerSelect();
                    displayImage(filter.getImageConveer());
                }
                Log.d(TAG, "corner: " + corners.toString());
                return true;

            case R.id.action_autoCornerSelect:
                Log.d(TAG, "start auto corner selecting");
                if(isImageLoaded()) {
                    displayImage(filter.cornerDetectAuto(sampledImageClone));
                }
                return true;

            case R.id.action_perspectiveTransform:
                if(corners.size()!=4)
                    return true;
                if(isImageLoaded()) {
                    manuallyCornerSelect();
                    sampledImageClone = filter.perspectiveTransform(sampledImageClone, corners);
                    displayImage(sampledImageClone);
                }
                return true;
            case R.id.action_resetFilters:
                if(isImageLoaded()){
                    filter.setImageConveer(sampledImage.clone());
                    displayImage(sampledImage);
                }
                return true;
            default:
                return super.onOptionsItemSelected(item);
        }
    }

    private boolean isImageLoaded(){
        if(sampledImage==null)
            Toast.makeText(getApplicationContext(),
                    "It is necessary to open image firstly",
                    Toast.LENGTH_SHORT).show();
        return sampledImage!=null;
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == SELECT_PICTURE && resultCode == RESULT_OK) {
            Uri selectedImageUri = data.getData(); //The uri with the location of the file
            Log.d(TAG, "uri" + selectedImageUri);
            convertToMat(selectedImageUri);
            filter.setImageConveer(sampledImage.clone());
        }
    }

    private void manuallyCornerSelect(){
        imageView.setOnTouchListener(new View.OnTouchListener() {
            @Override
            public boolean onTouch(View view, MotionEvent event) {
                Log.i(TAG, "event.getX(), event.getY(): " + event.getX() +" "+ event.getY());
                Mat conveerImage = filter.getImageConveer();
                if(conveerImage!=null) {
                    Log.i(TAG, "sampledImage.width(), sampledImage.height(): " + conveerImage.width() +" "+ conveerImage.height());
                    Log.i(TAG, "view.getWidth(), view.getHeight(): " + view.getWidth() +" "+ view.getHeight());
                    int left=(view.getWidth()-conveerImage.width())/2;
                    int top=(view.getHeight()-conveerImage.height())/2;
                    int right=(view.getWidth()+conveerImage.width())/2;
                    int bottom=(view.getHeight()+conveerImage.height())/2;
                    Log.i(TAG, "left: " + left +" right: "+ right +" top: "+ top +" bottom:"+ bottom);
                    if(event.getX()>=left && event.getX()<=right && event.getY()>=top && event.getY()<=bottom) {
                        int projectedX = (int)event.getX()-left;
                        int projectedY = (int)event.getY()-top;
                        org.opencv.core.Point corner = new org.opencv.core.Point(projectedX, projectedY);
                        corners.add(corner);
                        if(corners.size()>4)
                            corners.remove(0);
                        Mat sampleImageCopy=conveerImage.clone();
                        for(org.opencv.core.Point c : corners)
                            Imgproc.circle(sampleImageCopy, c, (int) 5, new Scalar(0, 0, 255), 2);
                        if(isSelectingCorner)
                            displayImage(sampleImageCopy);
                    }
                }
                return false;
            }
        });
    }

    private void convertToMat(Uri selectedImageUri) {
        try {
            InputStream ims = getContentResolver().openInputStream(selectedImageUri);
            Bitmap bmp = BitmapFactory.decodeStream(ims);
            Mat rgbImage = new Mat();
            Utils.bitmapToMat(bmp, rgbImage);
            ims.close();
            ims = getContentResolver().openInputStream(selectedImageUri);
            ExifInterface exif = new ExifInterface(ims);//selectedImageUri.getPath());
            int orientation = exif.getAttributeInt(ExifInterface.TAG_ORIENTATION,
                    1);
            switch (orientation) {
                case ExifInterface.ORIENTATION_ROTATE_90:
                    //get the mirrored image
                    rgbImage = rgbImage.t();
                    //flip on the y-axis
                    Core.flip(rgbImage, rgbImage, 1);
                    break;
                case ExifInterface.ORIENTATION_ROTATE_270:
                    //get up side down image
                    rgbImage = rgbImage.t();
                    //Flip on the x-axis
                    Core.flip(rgbImage, rgbImage, 0);
                    break;
            }

            Display display = getWindowManager().getDefaultDisplay();
            android.graphics.Point size = new android.graphics.Point();
            display.getSize(size);
            int width = size.x;
            int height = size.y;
            double downSampleRatio = calculateSubSampleSize(rgbImage, width, height);
            sampledImage = new Mat();
            Imgproc.resize(rgbImage, sampledImage, new
                    Size(), downSampleRatio, downSampleRatio, Imgproc.INTER_AREA);
            displayImage(sampledImage);
        } catch (Exception e) {
            Log.e(TAG, "Exception thrown: " + e + " " + Log.getStackTraceString(e));
            sampledImage = null;
        }
        filter = new Filter(true);
    }

    private static double calculateSubSampleSize(Mat srcImage, int reqWidth,
                                                 int reqHeight) {
        final int height = srcImage.height();
        final int width = srcImage.width();
        double inSampleSize = 1;
        if (height > reqHeight || width > reqWidth) {
            final double heightRatio = (double) reqHeight / (double) height;
            final double widthRatio = (double) reqWidth / (double) width;
            inSampleSize = heightRatio < widthRatio ? heightRatio : widthRatio;
        }
        return inSampleSize;
    }

    private void displayImage(Mat image) {
        Bitmap bitmap = Bitmap.createBitmap(image.cols(),
                image.rows(), Bitmap.Config.RGB_565);
        Utils.matToBitmap(image, bitmap);
        displayImage(bitmap);
    }

    private void displayImage(Bitmap bitmap) {
        imageView.setImageBitmap(bitmap);
    }
}
