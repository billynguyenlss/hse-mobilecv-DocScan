//https://pullrequest.opencv.org/buildbot/export/opencv_releases/master-contrib_pack-contrib-android/20200821-041002--11257/
#include <com_asav_matching_MainActivity.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/stitching.hpp>
#include <opencv2/ximgproc.hpp>
#include <opencv2/ximgproc.hpp>

#include <string>
#include <vector>

#include <android/log.h>

#define LOG_TAG "ImageMatching"
#define LOGD(...) ((void)__android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__))

using namespace std;
using namespace cv;


JNIEXPORT void JNICALL Java_com_asav_matching_MainActivity_extractPointsOfInterest
        (JNIEnv *, jclass, jlong addrMatIn, jlong addrMatOut)
{
    LOGD("Java_com_asav_matching_MainActivity_extractKeyFeatures -- BEGIN");
    cv::Mat& mIn = *(cv::Mat*)addrMatIn;
    cv::Mat grayImg;
    cv::cvtColor(mIn,grayImg,cv::COLOR_RGBA2GRAY);
    cv::Mat& mOut = *(cv::Mat*)addrMatOut;
    std::vector<cv::KeyPoint> v;
    cv::Ptr<cv::FastFeatureDetector> detector = cv::FastFeatureDetector::create(80);
    //cv::Ptr<cv::xfeatures2d::BriefDescriptorExtractor> detector=cv::xfeatures2d::BriefDescriptorExtractor::create();
    detector->detect(grayImg, v);
    //cv::drawKeypoints(mIn,v,mOut);
    for( unsigned int i = 0; i < v.size(); i++ )
    {
        const cv::KeyPoint& kp = v[i];
        cv::circle(mOut, cv::Point(kp.pt.x, kp.pt.y), 10,
        cv::Scalar(0,255,0,255));
    }
    LOGD("Java_com_asav_matching_MainActivity_extractKeyFeatures -- END");
}
JNIEXPORT void JNICALL Java_com_asav_matching_MainActivity_stitchImages
        (JNIEnv *, jclass, jlong addrMatIn1, jlong addrMatIn2, jlong addrMatOut){
    LOGD("Java_com_asav_matching_MainActivity_stitchImages -- BEGIN");
    cv::Mat& mIn1 = *(cv::Mat*)addrMatIn1;
    cv::Mat& mIn2 = *(cv::Mat*)addrMatIn2;
    cv::Mat& mOut = *(cv::Mat*)addrMatOut;
    cv::Mat rgb1,rgb2;
    cv::cvtColor(mIn1,rgb1,cv::COLOR_RGBA2RGB);
    cv::cvtColor(mIn2,rgb2,cv::COLOR_RGBA2RGB);

    std::vector<cv::Mat> natImgs;
    natImgs.push_back(rgb1);
    natImgs.push_back(rgb2);
    Ptr<cv::Stitcher> stitcher = cv::Stitcher::create(cv::Stitcher::SCANS);
    //Mat rgbOut;
    cv::Stitcher::Status sts=stitcher->stitch(natImgs, mOut);
    //cv::cvtColor(rgbOut,mOut,cv::COLOR_RGB2RGBA);
    LOGD("Java_com_asav_matching_MainActivity_stitchImages -- END");

}
JNIEXPORT void JNICALL Java_com_asav_matching_MainActivity_niBlackThreshold
        (JNIEnv *, jclass, jlong addrMatIn, jlong addrMatOut){
    LOGD("Java_com_asav_processimage_MainActivity_niBlackThreshold -- BEGIN");
    cv::Mat& mIn = *(cv::Mat*)addrMatIn;
    cv::Mat grayImg;
    cv::cvtColor(mIn,grayImg,cv::COLOR_RGBA2GRAY);
    cv::Mat& mOut = *(cv::Mat*)addrMatOut;
    cv::ximgproc::niBlackThreshold(grayImg,mOut,255, cv::THRESH_BINARY, 11,cv::ximgproc::BINARIZATION_NIBLACK  );
    LOGD("Java_com_asav_processimage_MainActivity_niBlackThreshold -- END");
}
JNIEXPORT void JNICALL Java_com_asav_matching_MainActivity_stitchMultipleImages
        (JNIEnv *env, jclass, jlongArray matsAddrIn, jlong matAddrOut){
    LOGD("Java_com_asav_matching_MainActivity_stitchImages -- BEGIN");

//    std::vector<cv::Mat> trainimgs;
    cv::Mat& mOut = *(cv::Mat*)matAddrOut;

    jint retVal=0;
    int num=0, temp=0;
    jclass alCls = env->FindClass("org/opencv/core/Mat");
    jmethodID jMatCons = env->GetMethodID(alCls,"<init>","()V");
    jmethodID alGetId  = env->GetMethodID(alCls, "getNativeObjAddr", "()J");
//    jmethodID sizeMethodID = env->GetMethodID(alCls, "size", "()I");

    jlong *traindata = env->GetLongArrayElements(matsAddrIn,0);

//    int intValue = *(int*) sizeMethodID;
    jsize a_len = env->GetArrayLength(matsAddrIn);

    std::vector<cv::Mat> natImgs;
    for(int i = 0; i < a_len; i++){
        try {
            cv::Mat& mIn = *(cv::Mat*)traindata[i];
            cv::Mat rgb;
            cv::cvtColor(mIn, rgb, cv::COLOR_RGBA2RGB);
            natImgs.push_back(rgb);
        } catch (...) {
            continue;
        }

    }
    env->ReleaseLongArrayElements(matsAddrIn,traindata,0);

    Ptr<cv::Stitcher> stitcher = cv::Stitcher::create(cv::Stitcher::SCANS);

    if (natImgs.size()>0){
        try {
            Mat rgbOut;
            cv::Stitcher::Status sts=stitcher->stitch(natImgs, rgbOut);
            cv::cvtColor(rgbOut,mOut,cv::COLOR_RGB2RGBA);
        } catch(...){
            return;
        }
    }

    LOGD("Java_com_asav_matching_MainActivity_stitchMultipleImages -- END");

}
