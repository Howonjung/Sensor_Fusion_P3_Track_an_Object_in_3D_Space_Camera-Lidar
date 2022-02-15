
#include <numeric>
#include "matching2D.hpp"

using namespace std;

// Find best matches for keypoints in two camera images based on several matching methods
void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource, cv::Mat &descRef,
                      std::vector<cv::DMatch> &matches, std::string descriptorType, std::string matcherType, std::string selectorType)
{
    // configure matcher
    bool crossCheck = false;
    cv::Ptr<cv::DescriptorMatcher> matcher;

    if (matcherType.compare("MAT_BF") == 0)
    {
        int normType = descriptorType.compare("DES_BINARY") == 0 ? cv::NORM_HAMMING : cv::NORM_L2;
        matcher = cv::BFMatcher::create(normType, crossCheck);
        cout << "BF matching" << "\n";
    }
    else if (matcherType.compare("MAT_FLANN") == 0)
    {   
        if (descSource.type() != CV_32F)
        { // OpenCV bug workaround : convert binary descriptors to floating point due to a bug in current OpenCV implementation
            descSource.convertTo(descSource, CV_32F);
            descRef.convertTo(descRef, CV_32F);
        }
        //... TODO : implement FLANN matching
        cout << "FLANN matching" << "\n";
        matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    }

    // perform matching task
    if (selectorType.compare("SEL_NN") == 0)
    { // nearest neighbor (best match)

        matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1
    }
    else if (selectorType.compare("SEL_KNN") == 0)
    {        
        // k nearest neighbors (k=2)
        std::vector<vector<cv::DMatch>> knn_matches;
        // knnMatch: Finds the k best matches for each descriptor from a query set.
        double t = (double)cv::getTickCount();
        matcher->knnMatch(descSource, descRef, knn_matches, 2);
        t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
        cout << " (KNN) with n=" << knn_matches.size() << " matches in " << 1000 * t / 1.0 << " ms" << endl;

        float descriptorDistanceRatio = 0.8;

        for (const auto& it : knn_matches ){
            // Consider using const for variables that are ensured not to change during the scope of a function.
            const auto d1 = it[0].distance;
            const auto d2 = it[1].distance;

            // /*Let d1 be the distance to the nearest neighbor and d2 be the distance to the next one. 
            // In order to accept the nearest neighbor as a “match”, d1/d2 ratio should be smaller than a given threshold (something like 0.8). 
            // The motivation behind this test is that we expect a good match to be much closer to the query feature than the second best match.
            // Because if both features are similarly close to the query, we cannot decide which one is really the best one.*/
            if(d1/d2 < descriptorDistanceRatio){ 
                matches.push_back(it[0]);
            }
        }
        cout << "# keypoints removed = " << knn_matches.size() - matches.size() << endl;
        cout << "# total matched points = " << matches.size() << endl;
    }
}

// Use one of several types of state-of-art descriptors to uniquely identify keypoints
void descKeypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, string descriptorType)
{
    // select appropriate descriptor
    cv::Ptr<cv::DescriptorExtractor> extractor;
    if (descriptorType.compare("BRISK") == 0)
    {

        int threshold = 30;        // FAST/AGAST detection threshold score.
        int octaves = 3;           // detection octaves (use 0 to do single scale)
        float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.

        extractor = cv::BRISK::create(threshold, octaves, patternScale);
    }
    else if (descriptorType.compare("BRIEF") == 0){
        int bytes = 32; // legth of the descriptor in bytes, valid values are: 16, 32 (default) or 64 .
        bool use_orientation = true; //	sample patterns using keypoints orientation, disabled by default.
        extractor = cv::xfeatures2d::BriefDescriptorExtractor::create();
    }
    else if (descriptorType.compare("ORB") == 0){
        // use all default values
        extractor = cv::ORB::create();
    }
    else if (descriptorType.compare("FREAK") == 0){
        // use all default values
        extractor = cv::xfeatures2d::FREAK::create();
        
    }
    else if (descriptorType.compare("AKAZE") == 0){
        // use all default values
        extractor = cv::AKAZE::create();

    }
    else if (descriptorType.compare("SIFT") == 0){
        // use all default values
        extractor = cv::xfeatures2d::SiftDescriptorExtractor::create();
    }
    
    else{ std::cout <<"descriptorType: " <<descriptorType<< " is not an available options" << "\n"; }

    // perform feature description
    double t = (double)cv::getTickCount();
    extractor->compute(img, keypoints, descriptors);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << descriptorType << " descriptor extraction in " << 1000 * t / 1.0 << " ms" << endl;
}

// Detect keypoints in image using ORB detector
void detKeypointsORB(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis){
    // Create ORB detector
    cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create();
    double t = (double)cv::getTickCount();
    // ORB detector input
    // - (input image, extracted keypoints)
    detector->detect(img, keypoints);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "ORB detector with n= " << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "ORB Detector Results";
        cv::namedWindow(windowName, 7);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}
// Detect keypoints in image using AKAZE detector
void detKeypointsAKAZE(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis){
     // Create AKAZE detector
    cv::Ptr<cv::FeatureDetector> detector = cv::AKAZE::create();
    double t = (double)cv::getTickCount();
    // AKAZE detector input
    // - (input image, extracted keypoints)
    detector->detect(img, keypoints);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "AKAZE detector with n= " << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "AKAZE Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}
// Detect keypoints in image using SIFT detector
void detKeypointsSIFT(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis){
     // Create SIFT detector
    cv::Ptr<cv::FeatureDetector> detector = cv::xfeatures2d::SIFT::create();
    double t = (double)cv::getTickCount();
    // SIFT detector input
    // - (input image, extracted keypoints)
    detector->detect(img, keypoints);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "SIFT detector with n= " << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "SIFT Detector Results";
        cv::namedWindow(windowName, 5);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}
// Detect keypoints in image using BRISK detector
void detKeypointsBRISK(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis){
    // Create BRISK detector
    cv::Ptr<cv::FeatureDetector> detector = cv::BRISK::create();
    double t = (double)cv::getTickCount();
    // BRISK detector input
    // - (input image, extracted keypoints)
    detector->detect(img, keypoints);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "BRISK detector with n= " << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "BRISK Detector Results";
        cv::namedWindow(windowName, 4);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}
// Detect keypoints in image using FAST detector
void detKeypointsFAST(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis){
    double t = (double)cv::getTickCount();
    // FAST input
    // - (input image, extracted keypoints, threshold(pixel diff between center and neighbor's), nonmaxSuppression)
    cv::FAST(img, keypoints, 70, true);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "FAST with n= " << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "FAST Detector Results";
        cv::namedWindow(windowName, 3);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}
// Detect keypoints in image using Harris corner detector
void detKeypointsHarris(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis){

    // Detector parameters
    int blockSize = 2;     // orig 2, for every pixel, a blockSize × blockSize neighborhood is considered
    int apertureSize = 3;  // aperture parameter for Sobel operator (must be odd)
    int minResponse = 100; // orig 100,minimum value for a corner in the 8bit scaled response matrix
    double k = 0.04;       // Harris parameter (see equation for details)

    double t = (double)cv::getTickCount();
    // Detect Harris corners and normalize output
    cv::Mat dst, dst_norm, dst_norm_scaled;
    dst = cv::Mat::zeros(img.size(), CV_32FC1);
    cv::cornerHarris(img, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT);
    cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
    cv::convertScaleAbs(dst_norm, dst_norm_scaled);

    // Perform a non-maximum suppression (NMS) in a local neighborhood around each maximum in the Harris response matrix.
    // Look for prominent corners and instantiate keypoints
    double maxOverlap = 0.0; // max. permissible overlap between two features in %, used during non-maxima suppression
    for (int i=0; i<dst_norm.rows; i++){
        for(int j=0; j<dst_norm.cols; j++){
            int response = (int)dst_norm.at<float>(i,j);
            // only store points above a threshold
            if(response > minResponse){
                cv::KeyPoint newKeypoint;
                newKeypoint.pt = cv::Point2f(j,i);
                newKeypoint.response = response;
                newKeypoint.size = 2*apertureSize;

                // perform non-maximum suppression (NMS) in local neighbourhood around new key point
                bool overlapFlag = false;
                for(auto it = keypoints.begin(); it<keypoints.end(); it++)
                {
                    float overlapRatio = cv::KeyPoint::overlap(newKeypoint, *it);
                    if(overlapRatio > maxOverlap){
                        overlapFlag = true;
                        if(newKeypoint.response > (*it).response){ // if overlap is >t AND response is higher for new kpt 
                            *it = newKeypoint;                     // replace old key point with new one
                            break;                                 // quit loop over keypoints
                        }

                    }
                }
                if(!overlapFlag)                        // only add new key point if no overlap has been found in previous NMS
                    keypoints.push_back(newKeypoint);   // store new keypoint in dynamic list
            }
 
        }
    }
    
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "Harris Corner detection with non-maximum suppression(NMS) n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Harris Corner Detector with non-maximum suppression(NMS) Results";
        cv::namedWindow(windowName, 2);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}
// Detect keypoints in image using the traditional Shi-Thomasi detector
void detKeypointsShiTomasi(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    // compute detector parameters based on image size
    int blockSize = 4;       //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
    double maxOverlap = 0.0; // max. permissible overlap between two features in %
    double minDistance = (1.0 - maxOverlap) * blockSize;
    int maxCorners = img.rows * img.cols / max(1.0, minDistance); // max. num. of keypoints

    double qualityLevel = 0.01; // minimal accepted quality of image corners
    double k = 0.04;

    // Apply corner detection
    double t = (double)cv::getTickCount();
    vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, false, k);

    // add corners to result vector
    for (auto it = corners.begin(); it != corners.end(); ++it)
    {

        cv::KeyPoint newKeyPoint;
        newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
        newKeyPoint.size = blockSize;
        keypoints.push_back(newKeyPoint);
    }
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "Shi-Tomasi detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Shi-Tomasi Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}