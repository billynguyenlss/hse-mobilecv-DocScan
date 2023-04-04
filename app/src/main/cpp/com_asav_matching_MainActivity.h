/* DO NOT EDIT THIS FILE - it is machine generated */
#include <jni.h>
/* Header for class com_asav_matching_MainActivity */

#ifndef _Included_com_asav_matching_MainActivity
#define _Included_com_asav_matching_MainActivity
#ifdef __cplusplus
extern "C" {
#endif
#undef com_asav_matching_MainActivity_SELECT_PICTURE
#define com_asav_matching_MainActivity_SELECT_PICTURE 1L
#undef com_asav_matching_MainActivity_SELECT_TEMPLATE_PICTURE_MATCH
#define com_asav_matching_MainActivity_SELECT_TEMPLATE_PICTURE_MATCH 2L
#undef com_asav_matching_MainActivity_SELECT_PICTURE_STITCHING
#define com_asav_matching_MainActivity_SELECT_PICTURE_STITCHING 3L
/*
 * Class:     com_asav_matching_MainActivity
 * Method:    extractPointsOfInterest
 * Signature: (JJ)V
 */
JNIEXPORT void JNICALL Java_com_asav_matching_MainActivity_extractPointsOfInterest
  (JNIEnv *, jclass, jlong, jlong);

/*
 * Class:     com_asav_matching_MainActivity
 * Method:    stitchImages
 * Signature: (JJJ)V
 */
JNIEXPORT void JNICALL Java_com_asav_matching_MainActivity_stitchImages
  (JNIEnv *, jclass, jlong, jlong, jlong);

/*
 * Class:     com_asav_processimage_OpenCVNativeCaller
 * Method:    niBlackThreshold
 * Signature: (JJ)V
 */
JNIEXPORT void JNICALL Java_com_asav_matching_MainActivity_niBlackThreshold
        (JNIEnv *, jclass, jlong, jlong);

#ifdef __cplusplus
}
#endif
#endif
