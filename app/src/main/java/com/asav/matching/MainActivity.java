package com.asav.matching;

import static android.app.PendingIntent.getActivity;

import static org.opencv.core.Core.BORDER_CONSTANT;
import static org.opencv.core.Core.DFT_SCALE;
import static org.opencv.core.CvType.CV_32FC1;
import static org.opencv.core.CvType.CV_8U;
import static org.opencv.core.CvType.CV_8UC1;
import static org.opencv.core.CvType.CV_8UC3;

import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.Toolbar;

import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import android.annotation.SuppressLint;
import android.app.Activity;
import android.content.ActivityNotFoundException;
import android.content.ContentValues;
import android.content.Context;
import android.content.ContextWrapper;
import android.content.Intent;
import android.content.pm.PackageInfo;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;

import androidx.core.content.FileProvider;
import androidx.exifinterface.media.ExifInterface;

import android.graphics.drawable.BitmapDrawable;
import android.media.MediaMetadataRetriever;
import android.media.MediaScannerConnection;
import android.net.Uri;
import android.os.Bundle;
import android.os.Environment;
import android.os.SystemClock;
import android.provider.MediaStore;
import android.util.Log;
import android.util.Rational;
import android.view.Display;
import android.view.Menu;
import android.view.MenuItem;
import android.view.MotionEvent;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.MediaController;
import android.widget.Toast;
import android.widget.VideoView;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraActivity;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.calib3d.Calib3d;
import org.opencv.core.*;
import org.opencv.features2d.*;
import org.opencv.imgproc.Imgproc;
import org.opencv.utils.Converters;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.text.SimpleDateFormat;
import java.util.*;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.Arrays;

public class MainActivity extends AppCompatActivity {
    /** Tag for the {@link Log}. */
    private static final String TAG = "MainActivity";
    private final int REQUEST_CODE_ASK_MULTIPLE_PERMISSIONS = 124;
    private ImageView imageView;
    private VideoView videoView;
    private Uri videoUri=null;
    private Button buttonSelectFrame;
    private Button buttonCloseVideo;
    private MediaMetadataRetriever mediaMetadataRetriever;
    private MediaController myMediaController;

    private Mat sampledImage=null;
    private TfLiteFeatureExtractor featureExtractor=null;
    private boolean useSuperpoint=true;
    private ArrayList<org.opencv.core.Point> corners=new ArrayList<org.opencv.core.Point>();

    private static native void extractPointsOfInterest(long matAddrIn, long matAddrOut);
    private static native void stitchImages(long matAddrIn1,long matAddrIn2, long matAddrOut);
    private static native void stitchMultipleImages(long[] matsAddrIn, long matAddrOut);


    private static native void niBlackThreshold(long matAddrIn, long matAddrOut);
    @SuppressLint("ClickableViewAccessibility")
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        Toolbar toolbar = (Toolbar) findViewById(R.id.my_toolbar);
        setSupportActionBar(toolbar);
        imageView=(ImageView)findViewById(R.id.inputImageView);
        imageView.setOnTouchListener(new View.OnTouchListener() {
            @Override
            public boolean onTouch(View view, MotionEvent event) {
                Log.i(TAG, "event.getX(), event.getY(): " + event.getX() +" "+ event.getY());
                if(sampledImage!=null) {
                    Log.i(TAG, "sampledImage.width(), sampledImage.height(): " + sampledImage.width() +" "+ sampledImage.height());
                    Log.i(TAG, "view.getWidth(), view.getHeight(): " + view.getWidth() +" "+ view.getHeight());
                    int left=(view.getWidth()-sampledImage.width())/2;
                    int top=(view.getHeight()-sampledImage.height())/2;
                    int right=(view.getWidth()+sampledImage.width())/2;
                    int bottom=(view.getHeight()+sampledImage.height())/2;
                    Log.i(TAG, "left: " + left +" right: "+ right +" top: "+ top +" bottom:"+ bottom);
                    if(event.getX()>=left && event.getX()<=right && event.getY()>=top && event.getY()<=bottom) {
                        int projectedX = (int)event.getX()-left;
                        int projectedY = (int)event.getY()-top;
                        org.opencv.core.Point corner = new org.opencv.core.Point(projectedX, projectedY);
                        corners.add(corner);
                        if(corners.size()>4)
                            corners.remove(0);
                        Mat sampleImageCopy=sampledImage.clone();
                        for(org.opencv.core.Point c : corners)
                            Imgproc.circle(sampleImageCopy, c, (int) 15, new Scalar(0, 0, 255), 2);
                        displayImage(sampleImageCopy);
                    }
                }
                return false;
            }
        });


        videoView=(VideoView) findViewById(R.id.inputVideoView);
        videoView.setVisibility(View.GONE);

        buttonSelectFrame=(Button)findViewById(R.id.button_selectFrame);
        buttonSelectFrame.setVisibility(View.GONE);

        buttonCloseVideo=(Button)findViewById(R.id.button_closeVideo);
        buttonCloseVideo.setVisibility(View.GONE);
        buttonCloseVideo.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                videoView.setVisibility(View.GONE);
                buttonSelectFrame.setVisibility(View.GONE);
                buttonCloseVideo.setVisibility(View.GONE);
            }
        });

        mediaMetadataRetriever = new MediaMetadataRetriever();
        buttonSelectFrame.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (videoUri != null){
                    int currentPosition = videoView.getCurrentPosition(); //in millisecond
                    Toast.makeText(MainActivity.this,
                            "Current Position: " + currentPosition + " (ms)",
                            Toast.LENGTH_LONG).show();
                    Bitmap bmFrame = mediaMetadataRetriever
                            .getFrameAtTime(currentPosition * 1000); //unit in microsecond
                    Uri imageUri = storeImage(bmFrame);
                    sampledImage=convertToMat(imageUri);
                    displayImage(bmFrame);
//                    if(sampledImage!=null)
//                        displayImage(sampledImage);
                } else {
                    Toast.makeText(MainActivity.this, "The video must be loaded!", Toast.LENGTH_SHORT).show();
                }
            }
        });

        if (!allPermissionsGranted()) {
            ActivityCompat.requestPermissions(this, getRequiredPermissions(), REQUEST_CODE_ASK_MULTIPLE_PERMISSIONS);

        }
        else
            init();
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        getMenuInflater().inflate(R.menu.toolbar_menu, menu);
        return true;
    }
    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {
                    Log.i(TAG, "OpenCV loaded successfully");
                    System.loadLibrary("ImageProcessLib");
                    Log.i(TAG, "After loading all libraries" );
                    Toast.makeText(getApplicationContext(),
                            "OpenCV loaded successfully",
                            Toast.LENGTH_SHORT).show();
                } break;
                default:
                {
                    super.onManagerConnected(status);
                    Toast.makeText(getApplicationContext(),
                            "OpenCV error",
                            Toast.LENGTH_SHORT).show();
                } break;
            }
        }
    };
    private void init(){
        try {
            featureExtractor=new TfLiteFeatureExtractor(getAssets());
        } catch (IOException e) {
            Log.e(TAG, "Exception thrown: " + e+" "+Log.getStackTraceString(e));
            featureExtractor=null;
            useSuperpoint=false;
        }
    }
    @Override
    public void onResume()
    {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION, this, mLoaderCallback);
        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
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
    private boolean allPermissionsGranted() {
        for (String permission : getRequiredPermissions()) {
            int status= ContextCompat.checkSelfPermission(this,permission);
            if (ContextCompat.checkSelfPermission(this,permission)
                    != PackageManager.PERMISSION_GRANTED) {
                return false;
            }
        }
        return true;
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        switch (requestCode) {
            case REQUEST_CODE_ASK_MULTIPLE_PERMISSIONS:
                Map<String, Integer> perms = new HashMap<String, Integer>();
                boolean allGranted = true;
                for (int i = 0; i < permissions.length; i++) {
                    perms.put(permissions[i], grantResults[i]);
                    if (grantResults[i] != PackageManager.PERMISSION_GRANTED)
                        allGranted = false;
                }
                // Check for ACCESS_FINE_LOCATION
                if (allGranted) {
                    // All Permissions Granted
                    init();
                } else {
                    // Permission Denied
                    Toast.makeText(MainActivity.this, "Some Permission is Denied", Toast.LENGTH_SHORT)
                            .show();
//                    finish();
                }
                break;
            default:
                super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        }
    }

    private static final int SELECT_PICTURE = 1;
    private static final int SELECT_TEMPLATE_PICTURE_MATCH = 2;
    private static final int SELECT_PICTURE_STITCHING = 3;

    private static final int REQUEST_IMAGE_CAPTURE = 4;
    private static final int SELECT_VIDEO = 5;
    private static final int SELECT_MULTIPLE_IMAGES = 6;
    private String mCurrentPhotoPath=null;
//    private List<long> matImgs = new ArrayList<long>();

    private File createImageFile() throws IOException {
        // Create an image file name
        String timeStamp = new SimpleDateFormat("yyyyMMdd_HHmmss").format(new Date());
        String imageFileName = "JPEG_" + timeStamp + "_";
        File storageDir = Environment.getExternalStoragePublicDirectory(
                Environment.DIRECTORY_PICTURES);
        File image = File.createTempFile(
                imageFileName,  /* prefix */
                ".jpg",         /* suffix */
                storageDir      /* directory */
        );

        // Save a file: path for use with ACTION_VIEW intents
        mCurrentPhotoPath = "file:" + image.getAbsolutePath();
        return image;
    }

    private void openImageFile(int requestCode){
        Intent intent = new Intent();
        intent.setType("image/*");
        intent.setAction(Intent.ACTION_GET_CONTENT);
        startActivityForResult(Intent.createChooser(intent,"Select Picture"),requestCode);
    }


    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        switch (item.getItemId()) {
            case R.id.action_openGallery:
                openImageFile(SELECT_PICTURE);
                return true;

            case R.id.action_takePhoto:
                Intent takePictureIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);

                File photoFile = null;
                mCurrentPhotoPath = null;
                try {
                    photoFile = createImageFile();
                } catch (IOException ex) {
                    // Error occurred while creating the File
                    Toast.makeText(this, "Error create File", Toast.LENGTH_LONG).show();
                }
                if (photoFile != null){
                    Uri photoURI = FileProvider.getUriForFile(MainActivity.this, BuildConfig.APPLICATION_ID + ".provider",photoFile);
                    mCurrentPhotoPath = photoURI.toString();
                    takePictureIntent.putExtra(MediaStore.EXTRA_OUTPUT, photoURI);
//                    Toast.makeText(MainActivity.this, "intent uri:" + mCurrentPhotoPath, Toast.LENGTH_LONG).show();
//                    Log.i(TAG, "inside intent called - mCurrentPhotoPath:" + mCurrentPhotoPath);

                    startActivityForResult(Intent.createChooser(takePictureIntent,"Photo saved to Gallery"),
                            REQUEST_IMAGE_CAPTURE);
                }

//                startActivityForResult(Intent.createChooser(takePictureIntent,"Take Picture from Camera"),
//                        REQUEST_IMAGE_CAPTURE);
                return true;

            case R.id.action_getFrameFromVideo:
//                Toast.makeText(this,"perform action_getFrameFromVideo", Toast.LENGTH_LONG).show();
                videoUri = null;

                Intent intent = new Intent();
                intent.setType("video/*");
                intent.setAction(Intent.ACTION_GET_CONTENT);
                startActivityForResult(Intent.createChooser(intent,"Select Video"),SELECT_VIDEO);
                return true;
            case R.id.action_saveImageToGallery:
                if (isImageLoaded()){
                    Bitmap bitmap = ((BitmapDrawable)imageView.getDrawable()).getBitmap();
                    storeImage(bitmap);
                    Toast.makeText(this,"Image saved to Gallery", Toast.LENGTH_LONG).show();
                } else {
                    Toast.makeText(this,"Image is not loaded :(", Toast.LENGTH_LONG).show();
                }
                return true;
            case R.id.action_binarization:
                Toast.makeText(this,"perform action_binarization", Toast.LENGTH_LONG).show();
                if (isImageLoaded()){
                    binary();
                }
                return true;
            case R.id.action_filtering:
                Toast.makeText(this,"perform action_filtering", Toast.LENGTH_LONG).show();
                if (isImageLoaded()){
                    blur();
                }
                return true;
            case R.id.action_contrastEnhancement:
                Toast.makeText(this,"perform action_contrastEnhancement", Toast.LENGTH_LONG).show();
                if (isImageLoaded()){
                    contrast();
                }
                return true;
            case R.id.action_noiseRemoving:
                Toast.makeText(this,"perform action_noiseRemoving", Toast.LENGTH_LONG).show();
                if (isImageLoaded()){
//                    bilateral();
                    noise_removing();
                }
                return true;
            case R.id.action_manual_perspective_transform:
                if(isImageLoaded()) {
                    Toast.makeText(this,"perform manual perspective transform", Toast.LENGTH_LONG).show();
                    // This function has some unknown error and not debug yet.
                    perspectiveTransform();
                }
                return true;
            case R.id.action_autoPerspectiveTransform:
                if(isImageLoaded()) {
                    Toast.makeText(this,"perform auto perspective transform", Toast.LENGTH_LONG).show();
//                    autoPerspectiveTransform();
                    findROI();
                }
                return true;
            case R.id.action_stitchimages:
                if(isImageLoaded()) {
                    openImageFile(SELECT_PICTURE_STITCHING);
                }
                return true;

            case R.id.action_stitchmultipleimages:
                Intent intentStitchMulti = new Intent(Intent.ACTION_GET_CONTENT);
                intentStitchMulti.setType("image/*"); //allows any image file type. Change * to specific extension to limit it
                //**The following line is the important one!
                intentStitchMulti.putExtra(Intent.EXTRA_ALLOW_MULTIPLE, true);
                startActivityForResult(Intent.createChooser(intentStitchMulti, "Select Multiple Picture"), SELECT_MULTIPLE_IMAGES);
                return true;
            case R.id.action_binarization_charThreshold:
                if (isImageLoaded()){
                    Toast.makeText(this,"perform binarization with char_threshold", Toast.LENGTH_LONG).show();
                    binary_with_char_threshold();
                }
                return true;
            default:
                // If we got here, the user's action was not recognized.
                // Invoke the superclass to handle it.
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
        if(resultCode==RESULT_OK) {
            if (requestCode == SELECT_PICTURE) {
                Uri selectedImageUri = data.getData(); //The uri with the location of the file
                Log.d(TAG, "uri" + selectedImageUri);
                sampledImage=convertToMat(selectedImageUri);
                if(sampledImage!=null)
                    displayImage(sampledImage);
            }
            else if(requestCode==SELECT_TEMPLATE_PICTURE_MATCH){
                Uri selectedImageUri = data.getData(); //The uri with the location of the file
                Mat imageToMatch=convertToMat(selectedImageUri);
                matchImages(imageToMatch);
            }
            else if(requestCode==SELECT_PICTURE_STITCHING){
                Uri selectedImageUri = data.getData(); //The uri with the location of the file
                Mat image2=convertToMat(selectedImageUri);
                Mat resImage;
                if(false){
                    resImage=new Mat();
                    long startTime = System.nanoTime();
                    stitchImages(sampledImage.getNativeObjAddr(),image2.getNativeObjAddr(),resImage.getNativeObjAddr());
                    long elapsedTime = System.nanoTime() - startTime;
                    elapsedTime = elapsedTime / 1000000; // Milliseconds (1:1000000)
                    Log.i(this.getClass().getSimpleName(), "OpenCV Stitching 2 images took " + elapsedTime + "ms");
                    if(resImage.rows()<=0 || resImage.cols()<=0) {
                        Toast.makeText(getApplicationContext(),
                                "Panorama not found",
                                Toast.LENGTH_SHORT).show();
                        return;
                    }
                }
                else
                    resImage=createPanorama(sampledImage,image2);
                displayImage(resImage);
            }
            else if (requestCode == REQUEST_IMAGE_CAPTURE) {
//                Bundle extras = data.getExtras();
//
//                Bitmap imageBitmap = (Bitmap) extras.get("data");
////                imageView.setImageBitmap(imageBitmap);
//                Uri imageUri = storeImage(imageBitmap);
//                sampledImage=convertToMat(imageUri);
//                displayImage(sampledImage);
//                Uri selectedImageUri = data.getData(); //The uri with the location of the file
//                Log.d(TAG, "uri" + selectedImageUri);
                sampledImage=convertToMat(Uri.parse(mCurrentPhotoPath));
                if(sampledImage!=null)
                    displayImage(sampledImage);
            }
            else if (requestCode == SELECT_VIDEO) {
                videoUri = data.getData();
                if (videoUri != null){
                    try {
                        Log.d(TAG, "videoUri: " + videoUri.toString());
                        mediaMetadataRetriever.setDataSource(MainActivity.this, videoUri);
                        videoView.setVideoURI(videoUri);
                        videoView.setVisibility(View.VISIBLE);
                        buttonSelectFrame.setVisibility(View.VISIBLE);
                        buttonCloseVideo.setVisibility(View.VISIBLE);
                        videoView.start();
                    } catch(Exception error){
                        Toast.makeText(MainActivity.this, "Something broken!", Toast.LENGTH_LONG).show();
                    }
                }

            }
            else if (requestCode == SELECT_MULTIPLE_IMAGES){
                Toast.makeText(MainActivity.this, "perform stitching multiple images", Toast.LENGTH_LONG).show();
                if (data.getClipData() != null){
                    try {
                        int count = data.getClipData().getItemCount(); //evaluate the count before the for loop --- otherwise, the count is evaluated every loop.
                        long[] matImgs = new long[count];

                        for(int i = 0; i < count; i++) {
                            Uri imageUri = data.getClipData().getItemAt(i).getUri();
                            //do something with the image (save it to some directory or whatever you need to do with it here)
                            Mat selectedImage = convertToMat(imageUri);
                            matImgs[i] = selectedImage.getNativeObjAddr();
                        }
                        Mat output = new Mat();
                        stitchMultipleImages(matImgs, output.getNativeObjAddr());
                        displayImage(output);
                    } catch (Exception error){
                        Toast.makeText(MainActivity.this, "Could not stitch multiple image :(", Toast.LENGTH_LONG).show();
                    }

                } else {
                    Toast.makeText(MainActivity.this, "Please select images!", Toast.LENGTH_LONG).show();
                }
            }
        }
    }

    private Uri storeImage(Bitmap finalBitmap) {
        String root = Environment.getExternalStoragePublicDirectory(
                Environment.DIRECTORY_PICTURES).toString();
        File myDir = new File(root + "/saved_images");
        myDir.mkdirs();
        Random generator = new Random();

        String timeStamp = new SimpleDateFormat("yyyyMMdd_HHmmss").format(new Date());
        String fname = "Image-"+ timeStamp +".jpg";
        File file = new File (myDir, fname);
        if (file.exists ()) file.delete ();
        try {
            FileOutputStream out = new FileOutputStream(file);
            finalBitmap.compress(Bitmap.CompressFormat.PNG, 100, out);
            // sendBroadcast(new Intent(Intent.ACTION_MEDIA_MOUNTED,
            //     Uri.parse("file://"+ Environment.getExternalStorageDirectory())));
            out.flush();
            out.close();

        } catch (Exception e) {
            e.printStackTrace();
        }
        // Tell the media scanner about the new file so that it is
        // immediately available to the user.
        MediaScannerConnection.scanFile(this, new String[]{file.toString()}, null,
            new MediaScannerConnection.OnScanCompletedListener() {
                public void onScanCompleted(String path, Uri uri) {
                    Log.i("ExternalStorage", "Scanned " + path + ":");
                    Log.i("ExternalStorage", "-> uri=" + uri);
                }
            }
        );
        return Uri.fromFile(file);
    }

    private Mat convertToMat(Uri selectedImageUri)
    {
        Mat resImage=null;
        try {
            InputStream ims = getContentResolver().openInputStream(selectedImageUri);
            Bitmap bmp=BitmapFactory.decodeStream(ims);
            Mat rgbImage=new Mat();
            Utils.bitmapToMat(bmp, rgbImage);
            ims.close();
            ims = getContentResolver().openInputStream(selectedImageUri);
            ExifInterface exif = new ExifInterface(ims);//selectedImageUri.getPath());
            int orientation = exif.getAttributeInt(ExifInterface.TAG_ORIENTATION,
                    1);
            switch (orientation)
            {
                case ExifInterface.ORIENTATION_ROTATE_90:
                    //get the mirrored image
                    rgbImage=rgbImage.t();
                    //flip on the y-axis
                    Core.flip(rgbImage, rgbImage, 1);
                    break;
                case ExifInterface.ORIENTATION_ROTATE_270:
                    //get up side down image
                    rgbImage=rgbImage.t();
                    //Flip on the x-axis
                    Core.flip(rgbImage, rgbImage, 0);
                    break;
            }

            Display display = getWindowManager().getDefaultDisplay();
            android.graphics.Point size = new android.graphics.Point();
            display.getSize(size);
            int width = size.x;
            int height = size.y;
            double downSampleRatio= calculateSubSampleSize(rgbImage,width,height);
            resImage=new Mat();
            Imgproc.resize(rgbImage, resImage, new
                    Size(),downSampleRatio,downSampleRatio,Imgproc.INTER_AREA);
        } catch (Exception e) {
            Log.e(TAG, "Exception thrown: " + e+" "+Log.getStackTraceString(e));
            resImage=null;
        }
        return resImage;
    }

    private static double calculateSubSampleSize(Mat srcImage, int reqWidth,
                                                 int reqHeight) {
        final int height = srcImage.height();
        final int width = srcImage.width();
        double inSampleSize = 1;
        if (height > reqHeight || width > reqWidth) {
            final double heightRatio = (double) reqHeight / (double) height;
            final double widthRatio = (double) reqWidth / (double) width;
            inSampleSize = heightRatio<widthRatio ? heightRatio :widthRatio;
        }
        return inSampleSize;
    }

    private final boolean isBinaryDetector=false;
    private Feature2D getDetector(){
        Feature2D detector;
        if(isBinaryDetector){
            //detector =FastFeatureDetector.create(50);
            detector=ORB.create();
            //detector= BriefDescriptorExtractor.create(256);
        }
        else {
            //detector= HarrisLaplaceFeatureDetector.create(1,0.02f);
            detector = SIFT.create();
            //detector=BRISK.create();
        }
        return detector;
    }
    private void extractFeatures(){
        Mat resImage=sampledImage.clone();
        long startTime = SystemClock.uptimeMillis();
        if(false)
            extractPointsOfInterest(sampledImage.getNativeObjAddr(),resImage.getNativeObjAddr());
        else{
            Mat grayImage=new Mat();
            Imgproc.cvtColor(sampledImage,grayImage, Imgproc.COLOR_RGB2GRAY);
            MatOfKeyPoint keyPoints=new MatOfKeyPoint();
            if(useSuperpoint) {
                Mat descriptors=featureExtractor.processImage(grayImage,keyPoints);
            }
            else {
                Feature2D detector = getDetector();
                detector.detect(grayImage, keyPoints);
            }
            Features2d.drawKeypoints(sampledImage, keyPoints, resImage,new Scalar(0,255,0));
        }
        Log.i(TAG, "Timecost to extractPointsOfInterest: " + Long.toString(SystemClock.uptimeMillis() - startTime));
        displayImage(resImage);
    }
    private void matchImages(Mat imageToMatch){
        //https://docs.opencv.org/3.4/d7/dff/tutorial_feature_homography.html
        long startTime = SystemClock.uptimeMillis();
        Mat imgScene=new Mat();
        Imgproc.cvtColor(sampledImage,imgScene, Imgproc.COLOR_RGB2GRAY);
        Mat imgObject=new Mat();
        Imgproc.cvtColor(imageToMatch,imgObject, Imgproc.COLOR_RGB2GRAY);

        Feature2D detector=getDetector();
        MatOfKeyPoint keypointsScene=new MatOfKeyPoint();

        Mat descriptors;
        descriptors=new Mat();
        detector.detectAndCompute(imgScene, new Mat(),keypointsScene,descriptors);

        MatOfKeyPoint keypointsObject=new MatOfKeyPoint();
        Mat descriptorsToMatch;
        descriptorsToMatch=new Mat();
        detector.detectAndCompute(imgObject, new Mat(),keypointsObject,descriptorsToMatch);

        DescriptorMatcher matcher =null;
        if(isBinaryDetector){
            matcher=DescriptorMatcher.create(DescriptorMatcher.BRUTEFORCE_HAMMING);
        }
        else{
            matcher=DescriptorMatcher.create(DescriptorMatcher.BRUTEFORCE);
        }
        List<MatOfDMatch> knnMatches = new ArrayList<>();
        matcher.knnMatch(descriptorsToMatch,descriptors, knnMatches, 2);

        ArrayList<DMatch> listOfGoodMatches =new ArrayList<DMatch>();
        //-- Filter matches using the Lowe's ratio test
        float ratioThresh = 0.75f;
        for (int i = 0; i < knnMatches.size(); i++) {
            if (knnMatches.get(i).rows() > 1) {
                DMatch[] matches = knnMatches.get(i).toArray();
                if (matches[0].distance < ratioThresh * matches[1].distance) {
                    listOfGoodMatches.add(matches[0]);
                }
            }
        }
        MatOfDMatch goodEnough=new MatOfDMatch();
        goodEnough.fromList(listOfGoodMatches );
        Mat imgMatches=new Mat();
        Features2d.drawMatches(imgObject, keypointsObject, imgScene,
                keypointsScene, goodEnough, imgMatches,Scalar.all(-1),Scalar.all(-1),new
                        MatOfByte(), Features2d.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS);

        //-- Localize the object
        List<Point> obj = new ArrayList<>();
        List<Point> scene = new ArrayList<>();
        List<KeyPoint> listOfKeypointsObject = keypointsObject.toList();
        List<KeyPoint> listOfKeypointsScene = keypointsScene.toList();
        for (int i = 0; i < listOfGoodMatches.size(); i++) {
            //-- Get the keypoints from the good matches
            obj.add(listOfKeypointsObject.get(listOfGoodMatches.get(i).queryIdx).pt);
            scene.add(listOfKeypointsScene.get(listOfGoodMatches.get(i).trainIdx).pt);
        }
        MatOfPoint2f objMat = new MatOfPoint2f(), sceneMat = new MatOfPoint2f();
        objMat.fromList(obj);
        sceneMat.fromList(scene);
        double ransacReprojThreshold = 3.0;
        Mat H = Calib3d.findHomography(objMat, sceneMat, Calib3d.RANSAC, ransacReprojThreshold);
        //-- Get the corners from the image_1 ( the object to be "detected" )
        Mat objCorners = new Mat(4, 1, CvType.CV_32FC2), sceneCorners = new Mat();
        float[] objCornersData = new float[(int) (objCorners.total() * objCorners.channels())];
        objCorners.get(0, 0, objCornersData);
        objCornersData[0] = 0;
        objCornersData[1] = 0;
        objCornersData[2] = imgObject.cols();
        objCornersData[3] = 0;
        objCornersData[4] = imgObject.cols();
        objCornersData[5] = imgObject.rows();
        objCornersData[6] = 0;
        objCornersData[7] = imgObject.rows();
        objCorners.put(0, 0, objCornersData);
        Core.perspectiveTransform(objCorners, sceneCorners, H);
        Log.i(TAG, "Timecost to matchImages: " + Long.toString(SystemClock.uptimeMillis() - startTime));

        float[] sceneCornersData = new float[(int) (sceneCorners.total() * sceneCorners.channels())];
        sceneCorners.get(0, 0, sceneCornersData);
        //-- Draw lines between the corners (the mapped object in the scene - image_2 )
        Imgproc.line(imgMatches, new Point(sceneCornersData[0] + imgObject.cols(), sceneCornersData[1]),
                new Point(sceneCornersData[2] + imgObject.cols(), sceneCornersData[3]), new Scalar(0, 255, 0), 4);
        Imgproc.line(imgMatches, new Point(sceneCornersData[2] + imgObject.cols(), sceneCornersData[3]),
                new Point(sceneCornersData[4] + imgObject.cols(), sceneCornersData[5]), new Scalar(0, 255, 0), 4);
        Imgproc.line(imgMatches, new Point(sceneCornersData[4] + imgObject.cols(), sceneCornersData[5]),
                new Point(sceneCornersData[6] + imgObject.cols(), sceneCornersData[7]), new Scalar(0, 255, 0), 4);
        Imgproc.line(imgMatches, new Point(sceneCornersData[6] + imgObject.cols(), sceneCornersData[7]),
                new Point(sceneCornersData[0] + imgObject.cols(), sceneCornersData[1]), new Scalar(0, 255, 0), 4);
        displayImage(imgMatches);
    }
    private Mat createPanorama(Mat...arg0) {
        // Base code extracted from: https://stackoverflow.com/questions/49357079/trying-to-get-opencv-to-work-in-java-sample-code-to-stitch-2-photos-together
        if (arg0.length != 2) {
            return null;
        }
        long startTime = System.nanoTime();
        // Convert the two bitmaps to OpenCV mats...
        Mat img1 = arg0[0];
        Mat img2 = arg0[1];

        Mat gray_image1 = new Mat();
        Mat gray_image2 = new Mat();
        Imgproc.cvtColor(img1, gray_image1, Imgproc.COLOR_RGB2GRAY);
        Imgproc.cvtColor(img2, gray_image2, Imgproc.COLOR_RGB2GRAY);

        MatOfKeyPoint keyPoints1 = new MatOfKeyPoint();
        MatOfKeyPoint keyPoints2 = new MatOfKeyPoint();
        Mat descriptors1 = new Mat();
        Mat descriptors2 = new Mat();
        Feature2D detector=getDetector();
        detector.detectAndCompute(gray_image1, new Mat(),keyPoints1, descriptors1);
        detector.detectAndCompute(gray_image2, new Mat(),keyPoints2, descriptors2);

        MatOfDMatch matches = new MatOfDMatch();
        DescriptorMatcher matcher =null;
        if(isBinaryDetector){
            matcher=DescriptorMatcher.create(DescriptorMatcher.BRUTEFORCE_HAMMING);
        }
        else{
            matcher=DescriptorMatcher.create(DescriptorMatcher.FLANNBASED);
        }
        matcher.match(descriptors1, descriptors2, matches);

        double max_dist = 0; double min_dist = 100;
        List<DMatch> listMatches = matches.toList();

        for( int i = 0; i < listMatches.size(); i++ ) {
            double dist = listMatches.get(i).distance;
            if( dist < min_dist ) min_dist = dist;
            if( dist > max_dist ) max_dist = dist;
        }

        // Reduce the list of matching keypoints to a list of good matches...
        LinkedList<DMatch> good_matches = new LinkedList<DMatch>();
        MatOfDMatch goodMatches = new MatOfDMatch();
        for(int i = 0; i < listMatches.size(); i++) {
            if(listMatches.get(i).distance < 2*min_dist) {
                good_matches.addLast(listMatches.get(i));
            }
        }
        goodMatches.fromList(good_matches);

        // Calculate the homograohy between the two images...
        LinkedList<Point> imgPoints1List = new LinkedList<Point>();
        LinkedList<Point> imgPoints2List = new LinkedList<Point>();
        List<KeyPoint> keypoints1List = keyPoints1.toList();
        List<KeyPoint> keypoints2List = keyPoints2.toList();

        for(int i = 0; i<good_matches.size(); i++) {
            imgPoints1List.addLast(keypoints1List.get(good_matches.get(i).queryIdx).pt);
            imgPoints2List.addLast(keypoints2List.get(good_matches.get(i).trainIdx).pt);
        }

        MatOfPoint2f obj = new MatOfPoint2f();
        obj.fromList(imgPoints1List);
        MatOfPoint2f scene = new MatOfPoint2f();
        scene.fromList(imgPoints2List);

        Mat H = Calib3d.findHomography(obj, scene, Calib3d.RANSAC,3);

        int imageWidth = img2.cols();
        int imageHeight = img2.rows();

        // To avoid missing some of the possible stitching scenarios, we offset the homography to the middle of a mat which has three time the size of one of the pictures.
        // Extracted from this: https://stackoverflow.com/questions/21618044/stitching-2-images-opencv
        Mat Offset = new Mat(3, 3, H.type());
        Offset.put(0,0, new double[]{1});
        Offset.put(0,1, new double[]{0});
        Offset.put(0,2, new double[]{imageWidth});
        Offset.put(1,0, new double[]{0});
        Offset.put(1,1, new double[]{1});
        Offset.put(1,2, new double[]{imageHeight});
        Offset.put(2,0, new double[]{0});
        Offset.put(2,1, new double[]{0});
        Offset.put(2,2, new double[]{1});

        // Multiply the homography mat with the offset.
        Core.gemm(Offset, H, 1, new Mat(), 0, H);

        Mat obj_corners = new Mat(4,1,CvType.CV_32FC2);
        Mat scene_corners = new Mat(4,1,CvType.CV_32FC2);

        obj_corners.put(0,0, new double[]{0,0});
        obj_corners.put(0,0, new double[]{imageWidth,0});
        obj_corners.put(0,0,new double[]{imageWidth,imageHeight});
        obj_corners.put(0,0,new double[]{0,imageHeight});

        Core.perspectiveTransform(obj_corners, scene_corners, H);

        // The resulting mat will be three times the size (width and height) of one of the source images. (We assume, that both images have the same size.
        Size s = new Size(imageWidth *3,imageHeight*3);
        Mat img_matches = new Mat(new Size(img1.cols()+img2.cols(),img1.rows()), CvType.CV_32FC2);

        // Perform the perspective warp of img1 with the given homography and place it on the large result mat.
        Imgproc.warpPerspective(img1, img_matches, H, s);

        // Create another mat which is used to hold the second image and place it in the middle of the large sized result mat.
        int m_xPos = (int)(img_matches.size().width/2 - img2.size().width/2);
        int m_yPos = (int)(img_matches.size().height/2 - img2.size().height/2);
        Mat m = new Mat(img_matches,new Rect(m_xPos, m_yPos, img2.cols(), img2.rows()));
        // Copy img2 to the mat in the middle of the large result mat
        img2.copyTo(m);

        long elapsedTime = System.nanoTime() - startTime;
        elapsedTime = elapsedTime / 1000000; // Milliseconds (1:1000000)
        Log.i(this.getClass().getSimpleName(), "Stitching 2 images took " + elapsedTime + "ms");

        // The resulting mat is way to big. It holds a lot of empty "transparent" space.
        // We will not crop the image, so that only the "region of interest" remains.
        Mat gray_out=new Mat();
        Imgproc.cvtColor(img_matches,gray_out,Imgproc.COLOR_BGR2GRAY);
        Mat mThreshold=new Mat();
        Imgproc.threshold(gray_out,mThreshold,1,255,Imgproc.THRESH_BINARY);
        Mat Points=new Mat();
        Core.findNonZero(mThreshold,Points);
        Rect imageBoundingBox3=Imgproc.boundingRect(Points);

        Mat regionOfInterest = img_matches.submat(imageBoundingBox3);

        return regionOfInterest;
    }
    private void displayImage(Mat image)
    {
        Bitmap bitmap = Bitmap.createBitmap(image.cols(),
                image.rows(),Bitmap.Config.RGB_565);
        Utils.matToBitmap(image, bitmap);
        displayImage(bitmap);
    }
    private void displayImage(Bitmap bitmap) {
        imageView.setImageBitmap(bitmap);
    }

    private void grayscale(){
        Mat grayImage=new Mat();
        Imgproc.cvtColor(sampledImage,grayImage, Imgproc.COLOR_RGB2GRAY);
        displayImage(grayImage);
    }
    private void binary(){
        Mat binImage = new Mat();
        if(true) {
            Mat grayImage = new Mat();
            Imgproc.cvtColor(sampledImage, grayImage, Imgproc.COLOR_RGB2GRAY);
            Imgproc.GaussianBlur(grayImage,grayImage,new Size(5,5),0,0);

            //Imgproc.adaptiveThreshold(grayImage, binImage, 255, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, Imgproc.THRESH_BINARY, 15, 12);
            //Imgproc.threshold(grayImage,binImage,128,255,Imgproc.THRESH_BINARY);
            Imgproc.threshold(grayImage,binImage,150,255,Imgproc.THRESH_BINARY+Imgproc.THRESH_OTSU);
        }
        else{
//            OpenCVNativeCaller.niBlackThreshold(sampledImage.getNativeObjAddr(),binImage.getNativeObjAddr());
            niBlackThreshold(sampledImage.getNativeObjAddr(),binImage.getNativeObjAddr());
        }
        displayImage(binImage);
    }

    private int char_threshold(int sigma, double percent, int k){
        Mat grayImage = new Mat();
        Imgproc.cvtColor(sampledImage, grayImage, Imgproc.COLOR_RGB2GRAY);
        Mat gHist = new Mat();
        int histSize = 256;
        float[] range = {0, 256}; //the upper boundary is exclusive
        MatOfFloat histRange = new MatOfFloat(range);
        boolean accumulate = false;
        List<Mat> bgrPlanes = new ArrayList<>();
        bgrPlanes.add(grayImage);
        Imgproc.calcHist(bgrPlanes, new MatOfInt(0), new Mat(), gHist, new MatOfInt(histSize), histRange, accumulate);


        for (int i = 0; i < k; i++){
            Imgproc.GaussianBlur(gHist, gHist, new Size(sigma,sigma),0,0, BORDER_CONSTANT);
        }
        Core.MinMaxLocResult res =Core.minMaxLoc(gHist);
        double minVal = res.minVal;
        double maxVal = res.maxVal;
        double maxLocX = res.maxLoc.x;
        double maxLocY = res.maxLoc.y;
        int threshold = (int)maxLocY;
        double lh = (double)(gHist.get(threshold, 0)[0]*100.0);
        double rh = (double)(maxVal*(100.0 - percent));
        while (lh >= rh){
            threshold -= (int)1;
            lh = (double)(gHist.get(threshold, 0)[0]*100.0);
            rh = (double)(maxVal*(100.0 - percent));

            if (threshold==0){
                break;
            }
        }
        return threshold;
    }
    private void binary_with_char_threshold() {
        int sigma = 5;
        double percent = 95.0;
        int k = 5;
        int threshold = char_threshold(sigma, percent, k);
        Mat grayImage = new Mat();
        Imgproc.cvtColor(sampledImage, grayImage, Imgproc.COLOR_RGB2GRAY);
        Imgproc.GaussianBlur(grayImage,grayImage,new Size(5,5),0,0);

        Mat binImage = new Mat();
        Imgproc.threshold(grayImage,binImage,0,255,Imgproc.THRESH_BINARY+Imgproc.THRESH_OTSU);
        displayImage(binImage);
    }

    private void noise_removing(){
        Mat out = new Mat();
        Imgproc.medianBlur(sampledImage, out, 5);
        displayImage(out);
    }
    private final boolean useColor=true;
    private void contrast(){
        Mat grayImage=new Mat();
        Imgproc.cvtColor(sampledImage,grayImage, Imgproc.COLOR_RGB2GRAY);
        Mat out=new Mat();
        if(useColor){
            Mat HSV=new Mat();
            Imgproc.cvtColor(sampledImage, HSV, Imgproc.COLOR_RGB2HSV);
            ArrayList<Mat> hsv_list = new ArrayList(3);
            Core.split(HSV,hsv_list);

            for(int channel=1;channel<=2;++channel) {
                Core.MinMaxLocResult minMaxLocRes = Core.minMaxLoc(hsv_list.get(channel));
                double minVal = minMaxLocRes.minVal;//+20;
                double maxVal = minMaxLocRes.maxVal;//-50;
                Mat corrected = new Mat();
                hsv_list.get(channel).convertTo(corrected, CV_8U, 255.0 / (maxVal - minVal), -minVal * 255.0 / (maxVal - minVal));
                hsv_list.set(channel, corrected);
            }
            Core.merge(hsv_list,HSV);
            Imgproc.cvtColor(HSV, out, Imgproc.COLOR_HSV2RGB);
        }
        else {
            Core.MinMaxLocResult minMaxLocRes = Core.minMaxLoc(grayImage);
            double minVal = minMaxLocRes.minVal;//+20;
            double maxVal = minMaxLocRes.maxVal;//-50;
            grayImage.convertTo(out, CV_8U, 255.0 / (maxVal - minVal), -minVal * 255.0 / (maxVal - minVal));
        }
        displayImage(out);
    }
    private void gammaCorrection(){
        double gammaValue = 1.3;
        Mat lookUpTable = new Mat(1, 256, CV_8U);
        byte[] lookUpTableData = new byte[(int) (lookUpTable.total() * lookUpTable.channels())];
        for (int i = 0; i < lookUpTable.cols(); i++) {
            lookUpTableData[i] = saturate(Math.pow(i / 255.0, gammaValue) * 255.0);
        }
        lookUpTable.put(0, 0, lookUpTableData);

        Mat out=new Mat();
        if(useColor){
            Mat HSV=new Mat();
            Imgproc.cvtColor(sampledImage, HSV, Imgproc.COLOR_RGB2HSV);
            ArrayList<Mat> hsv_list = new ArrayList(3);
            Core.split(HSV,hsv_list);

            for(int channel=1;channel<=2;++channel) {
                Mat corrected = new Mat();
                Core.LUT(hsv_list.get(channel), lookUpTable, corrected);
                hsv_list.set(channel, corrected);
            }
            Core.merge(hsv_list,HSV);
            Imgproc.cvtColor(HSV, out, Imgproc.COLOR_HSV2RGB);
        }
        else {
            Mat grayImage = new Mat();
            Imgproc.cvtColor(sampledImage, grayImage, Imgproc.COLOR_RGB2GRAY);

            Core.LUT(grayImage, lookUpTable, out);
        }
        displayImage(out);
    }
    private byte saturate(double val) {
        int iVal = (int) Math.round(val);
        iVal = iVal > 255 ? 255 : (iVal < 0 ? 0 : iVal);
        return (byte) iVal;
    }
    private void equalizeHisto(){
        Mat out=new Mat();
        if(useColor){
            Mat HSV=new Mat();
            Imgproc.cvtColor(sampledImage, HSV, Imgproc.COLOR_RGB2HSV);
            ArrayList<Mat> hsv_list = new ArrayList(3);
            Core.split(HSV,hsv_list);
            for(int channel=1;channel<=2;++channel) {
                Mat equalizedValue = new Mat();
                Imgproc.equalizeHist(hsv_list.get(channel), equalizedValue);
                hsv_list.set(channel, equalizedValue);
            }
            Core.merge(hsv_list,HSV);
            Imgproc.cvtColor(HSV, out, Imgproc.COLOR_HSV2RGB);
        }
        else {
            Imgproc.cvtColor(sampledImage, out, Imgproc.COLOR_RGB2GRAY);
            Imgproc.equalizeHist(out, out);
        }
        displayImage(out);
    }
    private void blur(){
        Mat out=new Mat();
        //Imgproc.cvtColor(sampledImage,out, Imgproc.COLOR_RGB2GRAY);
        Imgproc.GaussianBlur(sampledImage,out,new Size(7,7),0,0);
        displayImage(out);
    }

    //https://docs.opencv.org/3.4/d8/d01/tutorial_discrete_fourier_transform.html
    private void fft(){
        Mat grayImage = new Mat();
        Imgproc.cvtColor(sampledImage, grayImage, Imgproc.COLOR_RGB2GRAY);
        grayImage.convertTo(grayImage, CvType.CV_64FC1);

        int m = Core.getOptimalDFTSize(grayImage.rows());
        int n = Core.getOptimalDFTSize(grayImage.cols()); // on the border

        Mat padded = new Mat(new Size(n, m), CvType.CV_64FC1); // expand input

        Core.copyMakeBorder(grayImage, padded, 0, m - grayImage.rows(), 0,
                n - grayImage.cols(), Core.BORDER_CONSTANT);

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
        if(true) {
            Core.magnitude(planes.get(0), planes.get(1), spectrum);
            Core.add(spectrum, new Scalar(1), spectrum);
            Core.log(spectrum, spectrum);
        }
        else{
            Core.phase(planes.get(1), planes.get(0), spectrum);
        }

        Mat crop = new Mat(spectrum, new Rect(0, 0, spectrum.cols() & -2,
                spectrum.rows() & -2));

        Mat out = crop.clone();

        // rearrange the quadrants of Fourier image so that the origin is at the
        // image center
        int cx = out.cols() / 2;
        int cy = out.rows() / 2;

        Rect q0Rect = new Rect(0, 0, cx, cy);
        Rect q1Rect = new Rect(cx, 0, cx, cy);
        Rect q2Rect = new Rect(0, cy, cx, cy);
        Rect q3Rect = new Rect(cx, cy, cx, cy);

        Mat q0 = new Mat(out, q0Rect); // Top-Left - Create a ROI per quadrant
        Mat q1 = new Mat(out, q1Rect); // Top-Right
        Mat q2 = new Mat(out, q2Rect); // Bottom-Left
        Mat q3 = new Mat(out, q3Rect); // Bottom-Right

        Mat tmp = new Mat(); // swap quadrants (Top-Left with Bottom-Right)
        q0.copyTo(tmp);
        q3.copyTo(q0);
        tmp.copyTo(q3);

        q1.copyTo(tmp); // swap quadrant (Top-Right with Bottom-Left)
        q2.copyTo(q1);
        tmp.copyTo(q2);

        Core.normalize(out, out, 0, 255, Core.NORM_MINMAX);
        out.convertTo(out, CvType.CV_8UC1);
        displayImage(out);
    }
    //https://docs.opencv.org/master/de/dbc/tutorial_py_fourier_transform.html
    private void fftFilter(){
        Mat grayImage = new Mat();
        Imgproc.cvtColor(sampledImage, grayImage, Imgproc.COLOR_RGB2GRAY);
        grayImage.convertTo(grayImage, CvType.CV_64FC1);

        int m = Core.getOptimalDFTSize(grayImage.rows());
        int n = Core.getOptimalDFTSize(grayImage.cols()); // on the border

        Mat padded = new Mat(new Size(n, m), CvType.CV_64FC1); // expand input

        Core.copyMakeBorder(grayImage, padded, 0, m - grayImage.rows(), 0,
                n - grayImage.cols(), Core.BORDER_CONSTANT);

        List<Mat> planes = new ArrayList<Mat>();
        planes.add(padded);
        planes.add(Mat.zeros(padded.rows(), padded.cols(), CvType.CV_64FC1));
        Mat complexI = new Mat();
        Core.merge(planes, complexI); // Add to the expanded another plane with zeros
        Mat complexI2=new Mat();
        Core.dft(complexI, complexI2); // this way the result may fit in the source matrix

        //Mat mask=Mat.zeros(padded.rows(), padded.cols(), CvType.CV_64FC2);
        if(false){
            int cropSizeX=8;
            int cropSizeY=8;
            Mat mask=Mat.zeros(complexI2.size(),CV_8U);
            Mat crop = new Mat(mask, new Rect(cropSizeX, cropSizeY, complexI2.cols() - 2 * cropSizeX, complexI2.rows() - 2 * cropSizeY));
            crop.setTo(new Scalar(1));
            Mat tmp=new Mat();
            complexI2.copyTo(tmp, mask);
            complexI2=tmp;
        }
        else {
            int cropSizeX=complexI2.cols()/32;
            int cropSizeY=complexI2.rows()/32;
            Mat crop = new Mat(complexI2, new Rect(cropSizeX, cropSizeY, complexI2.cols() - 2 * cropSizeX, complexI2.rows() - 2 * cropSizeY));
            crop.setTo(new Scalar(0, 0));
        }
        Mat complexII=new Mat();
        Core.idft(complexI2, complexII,DFT_SCALE);
        Core.split(complexII, planes);
        Mat out = planes.get(0);

        Core.normalize(out, out, 0, 255, Core.NORM_MINMAX);
        out.convertTo(out, CvType.CV_8UC1);
        displayImage(out);
    }

    private Mat getNoisyImage(boolean add_noise){
        Mat noisyImage;
        if(add_noise) {
            Mat noise = new Mat(sampledImage.size(), sampledImage.type());
            MatOfDouble mean = new MatOfDouble ();
            MatOfDouble dev = new MatOfDouble ();
            Core.meanStdDev(sampledImage,mean,dev);
            Core.randn(noise,0, 1*dev.get(0,0)[0]);
            noisyImage = new Mat();
            Core.add(sampledImage, noise, noisyImage);
        }
        else{
            noisyImage=sampledImage;
        }
        return noisyImage;
    }
    private void addNoise(){
        Mat noisyImage=getNoisyImage(true);
        displayImage(noisyImage);
    }
    private void median(){
        Mat noisyImage=getNoisyImage(true);
        Mat blurredImage=new Mat();
        Imgproc.medianBlur(noisyImage,blurredImage, 7);
        displayImage(blurredImage);
    }
    private void bilateral(){
        Mat noisyImage=getNoisyImage(false);
        Mat outImage=new Mat();
        Mat rgb=new Mat();
        Imgproc.cvtColor(noisyImage, rgb, Imgproc.COLOR_RGBA2RGB);
        Imgproc.bilateralFilter(rgb,outImage,9,75,75);
        displayImage(outImage);
    }

    private void findROI(){
        // get 4 corners
        corners.clear();
        Mat grayImage = new Mat();
        Mat binImage = new Mat();
        Mat sampledImageCopied = new Mat();

        int morph_size = 2;
        Point anchor = new Point(-1,-1);
        Mat element = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(2*morph_size+1, 2*morph_size*1));
        Imgproc.morphologyEx(sampledImage, sampledImageCopied, Imgproc.MORPH_CLOSE, element, anchor, 5);
        Imgproc.morphologyEx(sampledImageCopied, sampledImageCopied, Imgproc.MORPH_OPEN, element, anchor, 5);

        // grabcut
        Rect rect = new Rect(5,5 ,sampledImage.cols()-20, sampledImage.rows() - 20);
        Mat result = new Mat(), bgModel = new Mat(), fgModel = new Mat();
        Mat img2 = new Mat();
//        sampledImageCopied.convertTo(img2, CV_8UC3);
//        Imgproc.grabCut(img2, result, rect, bgModel, fgModel, 1, Imgproc.GC_INIT_WITH_RECT);

        Imgproc.cvtColor(sampledImageCopied, grayImage, Imgproc.COLOR_RGB2GRAY);
        Imgproc.GaussianBlur(grayImage,grayImage,new Size(5,5),0,0);
        Imgproc.GaussianBlur(grayImage,grayImage,new Size(5,5),0,0);
        Imgproc.GaussianBlur(grayImage,grayImage,new Size(5,5),0,0);

        Imgproc.adaptiveThreshold(grayImage, binImage, 255, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, Imgproc.THRESH_BINARY, 15, 12);
        Imgproc.threshold(grayImage,binImage,128,255,Imgproc.THRESH_BINARY);
        Imgproc.threshold(grayImage,binImage,0,255,Imgproc.THRESH_BINARY+Imgproc.THRESH_OTSU);

        // edge detection
        Imgproc.cvtColor(sampledImageCopied, grayImage, Imgproc.COLOR_BGR2GRAY);
        Imgproc.GaussianBlur(grayImage,grayImage,new Size(3,3),0,0);
        Imgproc.GaussianBlur(grayImage,grayImage,new Size(3,3),0,0);
        Mat canny = new Mat();
        Imgproc.Canny(grayImage, canny, 50, 150);
        Imgproc.dilate(canny, canny, Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, new Size(5,5)));

        // contour detection
        List<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(canny, contours, hierarchy, Imgproc.RETR_TREE, Imgproc.CHAIN_APPROX_SIMPLE);

        if (contours.size() == 0){
            Toast.makeText(this, "No contour found :(", Toast.LENGTH_LONG).show();
            return;
        }
        contours.sort(new Comparator<MatOfPoint>() {
            public int compare(MatOfPoint c1, MatOfPoint c2) {
                return (int) (Imgproc.contourArea(c2)- Imgproc.contourArea(c1));
            }
        });

        Mat drawing = Mat.zeros(canny.size(), CvType.CV_8UC3);
        Random rng = new Random(12345);
        Scalar color = new Scalar(rng.nextInt(256), rng.nextInt(256), rng.nextInt(256));
        Imgproc.drawContours(drawing, contours, 0, new Scalar(255, 255, 255), -1, Imgproc.LINE_8, hierarchy, 0, new Point());
//        Imgproc.morphologyEx(drawing, drawing, Imgproc.MORPH_CLOSE, element, anchor, 3);
        Imgproc.erode(drawing, drawing, element, anchor, 2);
        Imgproc.dilate(drawing, drawing, element, anchor, 2);
//        Imgproc.morphologyEx(sampledImage, sampledImageCopied, Imgproc.MORPH_CLOSE, element, anchor, 5);

        MatOfPoint extractedCorners = new MatOfPoint();
        Mat drawingC1 = new Mat();
        Imgproc.cvtColor(drawing, drawingC1, Imgproc.COLOR_RGB2GRAY, 0);

        int minDistance = (int)(imageView.getWidth()*0.1);
        Imgproc.goodFeaturesToTrack(drawingC1, extractedCorners, 4, 0.01, minDistance, new Mat(),
                3, 3, true, 0.04);
        List<Point> extract = Arrays.asList(extractedCorners.toArray());
        Toast.makeText(this, "extract:" + extract.size(), Toast.LENGTH_LONG).show();

        int left=(imageView.getWidth()-sampledImage.width())/2;
        int top=(imageView.getHeight()-sampledImage.height())/2;
        int right=(imageView.getWidth()+sampledImage.width())/2;
        int bottom=(imageView.getHeight()+sampledImage.height())/2;

        for (Point c: extract){
            Point corner = new Point(c.x, c.y);
            corners.add(corner);
        }
        Mat out = sampledImage.clone();
        for (Point c: corners){
            Imgproc.circle(out, c, (int) 15, new Scalar(0, 0, 255), 3);
        }

        if (corners.size()==4){
            perspectiveTransform();
        } else {
            Toast.makeText(this, "Could not find enough 4 corners. Please select more corner!", Toast.LENGTH_LONG).show();
            displayImage(out);
        }

        // perform perspective transform

    }

    private void perspectiveTransform(){
        if(corners.size()<4){
            Toast.makeText(getApplicationContext(),
                    "It is necessary to choose 4 corners",
                    Toast.LENGTH_SHORT).show();
            return;
        }
        org.opencv.core.Point centroid=new org.opencv.core.Point(0,0);
        for(org.opencv.core.Point point:corners)
        {
            centroid.x+=point.x;
            centroid.y+=point.y;
        }
        centroid.x/=corners.size();
        centroid.y/=corners.size();

        sortCorners(corners,centroid);
        Mat correctedImage=new Mat(sampledImage.rows(),sampledImage.cols(),sampledImage.type());
        Mat srcPoints= Converters.vector_Point2f_to_Mat(corners);

        Mat destPoints=Converters.vector_Point2f_to_Mat(Arrays.asList(new org.opencv.core.Point[]{
                new org.opencv.core.Point(0, 0),
                new org.opencv.core.Point(correctedImage.cols(), 0),
                new org.opencv.core.Point(correctedImage.cols(),correctedImage.rows()),
                new org.opencv.core.Point(0,correctedImage.rows())}));

        Mat transformation=Imgproc.getPerspectiveTransform(srcPoints, destPoints);
        Imgproc.warpPerspective(sampledImage, correctedImage, transformation, correctedImage.size());

        corners.clear();
        displayImage(correctedImage);
    }

    void sortCorners(ArrayList<Point> corners, org.opencv.core.Point center)
    {
        ArrayList<org.opencv.core.Point> top=new ArrayList<org.opencv.core.Point>();
        ArrayList<org.opencv.core.Point> bottom=new ArrayList<org.opencv.core.Point>();

        for (int i = 0; i < corners.size(); i++)
        {
            if (corners.get(i).y < center.y)
                top.add(corners.get(i));
            else
                bottom.add(corners.get(i));
        }

        double topLeft=top.get(0).x;
        int topLeftIndex=0;
        for(int i=1;i<top.size();i++)
        {
            if(top.get(i).x<topLeft)
            {
                topLeft=top.get(i).x;
                topLeftIndex=i;
            }
        }

        double topRight=0;
        int topRightIndex=0;
        for(int i=0;i<top.size();i++)
        {
            if(top.get(i).x>topRight)
            {
                topRight=top.get(i).x;
                topRightIndex=i;
            }
        }

        double bottomLeft=bottom.get(0).x;
        int bottomLeftIndex=0;
        for(int i=1;i<bottom.size();i++)
        {
            if(bottom.get(i).x<bottomLeft)
            {
                bottomLeft=bottom.get(i).x;
                bottomLeftIndex=i;
            }
        }

        double bottomRight=bottom.get(0).x;
        int bottomRightIndex=0;
        for(int i=1;i<bottom.size();i++)
        {
            if(bottom.get(i).x>bottomRight)
            {
                bottomRight=bottom.get(i).x;
                bottomRightIndex=i;
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

    private void multipleStich(){

    }
}