
/* INCLUDES FOR THIS PROJECT */
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <limits>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include "dataStructures.h"
#include "matching2D.hpp"
#include "objectDetection2D.hpp"
#include "lidarData.hpp"
#include "camFusion.hpp"

using namespace std;

/* MAIN PROGRAM */
int main(int argc, const char *argv[])
{
    /* INIT VARIABLES AND DATA STRUCTURES */

    // data location
    string dataPath = "../";

    // camera
    string imgBasePath = dataPath + "images/";
    string imgPrefix = "KITTI/2011_09_26/image_02/data/000000"; // left camera, color
    string imgFileType = ".png";
    int imgStartIndex = 0; // first file index to load (assumes Lidar and camera names have identical naming convention)
    int imgEndIndex = 18;   // last file index to load
    int imgStepWidth = 1; // 1 means that it will use every single image
    int imgFillWidth = 4;  // no. of digits which make up the file index (e.g. img-0001.png)

    // object detection based on YOLO Ver3
    string yoloBasePath = dataPath + "dat/yolo/";
    string yoloClassesFile = yoloBasePath + "coco.names";
    string yoloModelConfiguration = yoloBasePath + "yolov3.cfg";
    string yoloModelWeights = yoloBasePath + "yolov3.weights";

    // Lidar
    string lidarPrefix = "KITTI/2011_09_26/velodyne_points/data/000000";
    string lidarFileType = ".bin";

    // calibration data for camera and lidar
    cv::Mat P_rect_00(3,4,cv::DataType<double>::type); // 3x4 projection matrix after rectification (intrinsic - calibration mat)
    cv::Mat R_rect_00(4,4,cv::DataType<double>::type); // 3x3 rectifying rotation to make image planes co-planar (rotation mat relating left & right camera of stereo setup)
    cv::Mat RT(4,4,cv::DataType<double>::type); // rotation matrix and translation vector (extrinsic - lidar to camera coordinate system transformation mat)
    
    RT.at<double>(0,0) = 7.533745e-03; RT.at<double>(0,1) = -9.999714e-01; RT.at<double>(0,2) = -6.166020e-04; RT.at<double>(0,3) = -4.069766e-03;
    RT.at<double>(1,0) = 1.480249e-02; RT.at<double>(1,1) = 7.280733e-04; RT.at<double>(1,2) = -9.998902e-01; RT.at<double>(1,3) = -7.631618e-02;
    RT.at<double>(2,0) = 9.998621e-01; RT.at<double>(2,1) = 7.523790e-03; RT.at<double>(2,2) = 1.480755e-02; RT.at<double>(2,3) = -2.717806e-01;
    RT.at<double>(3,0) = 0.0; RT.at<double>(3,1) = 0.0; RT.at<double>(3,2) = 0.0; RT.at<double>(3,3) = 1.0;
    
    R_rect_00.at<double>(0,0) = 9.999239e-01; R_rect_00.at<double>(0,1) = 9.837760e-03; R_rect_00.at<double>(0,2) = -7.445048e-03; R_rect_00.at<double>(0,3) = 0.0;
    R_rect_00.at<double>(1,0) = -9.869795e-03; R_rect_00.at<double>(1,1) = 9.999421e-01; R_rect_00.at<double>(1,2) = -4.278459e-03; R_rect_00.at<double>(1,3) = 0.0;
    R_rect_00.at<double>(2,0) = 7.402527e-03; R_rect_00.at<double>(2,1) = 4.351614e-03; R_rect_00.at<double>(2,2) = 9.999631e-01; R_rect_00.at<double>(2,3) = 0.0;
    R_rect_00.at<double>(3,0) = 0; R_rect_00.at<double>(3,1) = 0; R_rect_00.at<double>(3,2) = 0; R_rect_00.at<double>(3,3) = 1;
    
    P_rect_00.at<double>(0,0) = 7.215377e+02; P_rect_00.at<double>(0,1) = 0.000000e+00; P_rect_00.at<double>(0,2) = 6.095593e+02; P_rect_00.at<double>(0,3) = 0.000000e+00;
    P_rect_00.at<double>(1,0) = 0.000000e+00; P_rect_00.at<double>(1,1) = 7.215377e+02; P_rect_00.at<double>(1,2) = 1.728540e+02; P_rect_00.at<double>(1,3) = 0.000000e+00;
    P_rect_00.at<double>(2,0) = 0.000000e+00; P_rect_00.at<double>(2,1) = 0.000000e+00; P_rect_00.at<double>(2,2) = 1.000000e+00; P_rect_00.at<double>(2,3) = 0.000000e+00;    

    // misc
    double sensorFrameRate = 10.0 / imgStepWidth; // frames per second for Lidar and camera
    int dataBufferSize = 2;       // no. of images which are held in memory (ring buffer) at the same time
    vector<DataFrame> dataBuffer; // list of data frames which are held in memory at the same time
    bool bVis = false;            // visualize results
    ofstream resOut;              // TTC result file 
    bool bresultFileSave = false;
    bool bFirstLine = false;
    if(bresultFileSave){resOut.open("../data/TTCresult.txt");}
    /* MAIN LOOP OVER ALL IMAGES */
    // available detectorTypes
    // std::vector<string> detectorTypeVec = {"SHITOMASI", "HARRIS", "FAST", "BRISK", "ORB", "AKAZE", "SIFT"};
    std::vector<string> detectorTypeVec = {"FAST"};
    // available descriptorTypes
    // std::vector<string> descriptorTypeVec = { "BRISK", "BRIEF", "ORB", "FREAK", "AKAZE", "SIFT"};
    std::vector<string> descriptorTypeVec = {"BRIEF"};
    std::vector<TTCresult> TTCresultVec;
    int TTCcalModel = 0; // 0 - CVM(Constant Velocity Model), 1 - CAM (Conatant Acceleration Model)
    for (auto it1 = detectorTypeVec.begin(); it1!=detectorTypeVec.end(); it1++)
    {  
        for (auto it2 = descriptorTypeVec.begin(); it2!=descriptorTypeVec.end(); it2++)
        {   
            // dataBuffer clear for new detector and descriptor set
            dataBuffer.clear();

            // detectorType AKAZE need to be matched with descriptorType AKAZE
            if ((*it1) == "AKAZE" && (*it2) != "AKAZE")
                continue; 
            if ((*it1) != "AKAZE" && (*it2) == "AKAZE")
                continue;
            // detectorType SIFT can't be used with descriptorType ORB
            if((*it1) == "SIFT" && (*it2) == "ORB")
                continue;
            TTCresult TTCresult;
            TTCresult.detectorType = (*it1);
            TTCresult.descriptorType = (*it2);
            cout << "============================================="<<endl;
            cout << "TTCresult.detectorType: " << TTCresult.detectorType << endl;
            cout << "TTCresult.descriptorType " << TTCresult.descriptorType << endl;
            cout << "============================================="<<endl;

            // for constant acceleration model
            double vehicleVel = -1e9;
            double vehicleAcc = -1e9;

            for (size_t imgIndex = 0; imgIndex <= imgEndIndex - imgStartIndex; imgIndex+=imgStepWidth)
            {
                /* LOAD IMAGE INTO BUFFER */

                // assemble filenames for current index
                ostringstream imgNumber;
                imgNumber << setfill('0') << setw(imgFillWidth) << imgStartIndex + imgIndex;
                string imgFullFilename = imgBasePath + imgPrefix + imgNumber.str() + imgFileType;

                // load image from file 
                cv::Mat img = cv::imread(imgFullFilename);

                // push image into data frame buffer
                DataFrame frame;
                frame.cameraImg = img;

                // Ring buffer
                // - If dataBuffer.size() has same size as dataBufferSize, 
                //   erase first element in dataBuffer and add next frame at the end
                if (dataBuffer.size() >= dataBufferSize){
                    std::vector<DataFrame>::iterator it = dataBuffer.begin();
                    dataBuffer.erase(it);
                }

                dataBuffer.push_back(frame);

                cout << "#1 : LOAD IMAGE INTO BUFFER done" << endl;


                /* DETECT & CLASSIFY OBJECTS */

                float confThreshold = 0.2;
                float nmsThreshold = 0.4;
                // DETECT & CLASSIFY OBJECTS based on YOLO
                // output -> boundingBoxes        
                detectObjects((dataBuffer.end() - 1)->cameraImg, (dataBuffer.end() - 1)->boundingBoxes, confThreshold, nmsThreshold,
                            yoloBasePath, yoloClassesFile, yoloModelConfiguration, yoloModelWeights, bVis);

                cout << "#2 : DETECT & CLASSIFY OBJECTS done" << endl;


                /* CROP LIDAR POINTS */

                // load 3D Lidar points from file
                string lidarFullFilename = imgBasePath + lidarPrefix + imgNumber.str() + lidarFileType;
                std::vector<LidarPoint> lidarPoints;
                loadLidarFromFile(lidarPoints, lidarFullFilename);

                // remove Lidar points based on distance properties
                float minZ = -1.5, maxZ = -0.9, minX = 2.0, maxX = 20.0, maxY = 2.0, minR = 0.1; // focus on ego lane
                cropLidarPoints(lidarPoints, minX, maxX, maxY, minZ, maxZ, minR);
            
                (dataBuffer.end() - 1)->lidarPoints = lidarPoints;

                cout << "#3 : CROP LIDAR POINTS done" << endl;


                /* CLUSTER LIDAR POINT CLOUD */

                // associate Lidar points with camera-based ROI
                float shrinkFactor = 0.10; // shrinks each bounding box by the given percentage to avoid 3D object merging at the edges of an ROI
                clusterLidarWithROI((dataBuffer.end()-1)->boundingBoxes, (dataBuffer.end() - 1)->lidarPoints, shrinkFactor, P_rect_00, R_rect_00, RT);

                // Visualize 3D objects
                bVis = false;
                if(bVis)
                {
                    show3DObjects((dataBuffer.end()-1)->boundingBoxes, cv::Size(4.0, 20.0), cv::Size(2000, 2000), true);
                }
                bVis = false;

                cout << "#4 : CLUSTER LIDAR POINT CLOUD done" << endl;
                
                
                // REMOVE THIS LINE BEFORE PROCEEDING WITH THE FINAL PROJECT
                // continue; // skips directly to the next image without processing what comes beneath

                /* DETECT IMAGE KEYPOINTS */

                // convert current image to grayscale
                cv::Mat imgGray;
                cv::cvtColor((dataBuffer.end()-1)->cameraImg, imgGray, cv::COLOR_BGR2GRAY);

                // extract 2D keypoints from current image
                vector<cv::KeyPoint> keypoints; // create empty feature list for current image
                
                    // Available detectorType options: SHITOMASI, HARRIS, FAST, BRISK, ORB, AKAZE, SIFT
                    // string detectorType = "FAST";
                    // double detKeypoingTime, descriptorExtractTime;
                    // int matchSize;

                    if ((*it1).compare("SHITOMASI") == 0)
                        detKeypointsShiTomasi(keypoints, imgGray, false);
                    else if((*it1).compare("HARRIS") == 0)
                        detKeypointsHarris(keypoints, imgGray, false);
                    else if((*it1).compare("FAST") == 0)
                        detKeypointsFAST(keypoints, imgGray, false);
                    else if((*it1).compare("BRISK") == 0)
                        detKeypointsBRISK(keypoints, imgGray, false);
                    else if((*it1).compare("ORB") == 0)
                        detKeypointsORB(keypoints, imgGray, false);
                    else if((*it1).compare("AKAZE") == 0)
                        detKeypointsAKAZE(keypoints, imgGray, false);
                    else if((*it1).compare("SIFT") == 0)
                        detKeypointsSIFT(keypoints, imgGray, false);
                    else{
                        cout <<"detectorType: " << (*it1) << " is not in available options s" << "\n";
                    }

                    // optional : limit number of keypoints (helpful for debugging and learning)
                    bool bLimitKpts = false;
                    if (bLimitKpts)
                    {
                        int maxKeypoints = 50;

                        if ((*it1).compare("SHITOMASI") == 0)
                        { // there is no response info, so keep the first 50 as they are sorted in descending quality order
                            keypoints.erase(keypoints.begin() + maxKeypoints, keypoints.end());
                        }
                        cv::KeyPointsFilter::retainBest(keypoints, maxKeypoints);
                        cout << " NOTE: Keypoints have been limited!" << endl;
                    }

                    // push keypoints and descriptor for current frame to end of data buffer
                    (dataBuffer.end() - 1)->keypoints = keypoints;

                    cout << "#5 : DETECT KEYPOINTS done" << endl;


                    /* EXTRACT KEYPOINT DESCRIPTORS */

                    cv::Mat descriptors;
                    descKeypoints((dataBuffer.end() - 1)->keypoints, (dataBuffer.end() - 1)->cameraImg, descriptors, (*it2));
                    

                    // push descriptors for current frame to end of data buffer
                    (dataBuffer.end() - 1)->descriptors = descriptors;

                    cout << "#6 : EXTRACT DESCRIPTORS done" << endl;


                    if (dataBuffer.size() > 1) // wait until at least two images have been processed
                    {

                        /* MATCH KEYPOINT DESCRIPTORS */

                        vector<cv::DMatch> matches;
                        string matcherType = "MAT_BF";        // MAT_BF, MAT_FLANN
                        string descriptorDataType = "DES_BINARY"; // DES_BINARY, DES_HOG
                        string selectorType = "SEL_KNN";       // SEL_NN, SEL_KNN
                        
                        // change descriptorDataType into DES_HOG when descriptorType is SIFT.
                        if((*it2) == "SIFT"){ descriptorDataType = "DES_HOG"; }

                        matchDescriptors((dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints,
                                        (dataBuffer.end() - 2)->descriptors, (dataBuffer.end() - 1)->descriptors,
                                        matches, descriptorDataType, matcherType, selectorType);

                        // store matches in current data frame
                        (dataBuffer.end() - 1)->kptMatches = matches;

                        cout << "#7 : MATCH KEYPOINT DESCRIPTORS done" << endl;

                        // visualize matches between current and previous image
                        bVis = false;
                        if (bVis)
                        {
                            cv::Mat matchImg = ((dataBuffer.end() - 1)->cameraImg).clone();
                            cv::drawMatches((dataBuffer.end() - 2)->cameraImg, (dataBuffer.end() - 2)->keypoints,
                                            (dataBuffer.end() - 1)->cameraImg, (dataBuffer.end() - 1)->keypoints,
                                            matches, matchImg,
                                            cv::Scalar::all(-1), cv::Scalar::all(-1),
                                            vector<char>(), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

                            string windowName = (*it1) + "-"+ (*it2) + " Matching keypoints between two camera images";
                            cv::namedWindow(windowName, 7);
                            cv::imshow(windowName, matchImg);
                            cout << "Press key to continue to next image" << endl << endl;
                            cv::waitKey(0); // wait for key to be pressed
                        }
                        bVis = false;

                        
                        /* TRACK 3D OBJECT BOUNDING BOXES */

                        //// STUDENT ASSIGNMENT
                        //// TASK FP.1 -> match list of 3D objects (vector<BoundingBox>) between current and previous frame (implement ->matchBoundingBoxes)
                        map<int, int> bbBestMatches;
                        matchBoundingBoxes(matches, bbBestMatches, *(dataBuffer.end()-2), *(dataBuffer.end()-1)); // associate bounding boxes between current and previous frame using keypoint matches
                        //// EOF STUDENT ASSIGNMENT
                        
                        // Visualize matched bounding boxes
                        if(bVis){
                                char str1[20],str2[20];
                                for(auto it1= (dataBuffer.end()-2)->boundingBoxes.begin(); it1!=(dataBuffer.end()-2)->boundingBoxes.end(); it1++){
                                    cv::rectangle((dataBuffer.end() - 2)->cameraImg, it1->roi, cv::Scalar(0,0,255), 2);
                                    sprintf(str1, "id=%d ", it1->boxID);
                                    putText((dataBuffer.end() - 2)->cameraImg, str1, cv::Point2f(it1->roi.x + (it1->roi.width)/2, it1->roi.y+(it1->roi.height)/2), cv::FONT_ITALIC, 0.5, cv::Scalar(0,0,255));
                                }

                                for(auto it2= (dataBuffer.end()-1)->boundingBoxes.begin(); it2!=(dataBuffer.end()-1)->boundingBoxes.end(); it2++){
                                    cv::rectangle((dataBuffer.end() - 1)->cameraImg, it2->roi, cv::Scalar(255,0,0), 2);
                                    sprintf(str2, "id=%d ", it2->boxID);
                                    putText((dataBuffer.end() - 1)->cameraImg, str2, cv::Point2f(it2->roi.x + (it2->roi.width)/2, it2->roi.y+(it2->roi.height)/2), cv::FONT_ITALIC, 0.5, cv::Scalar(0,255,0));
                                }
                                for (map<int, int>::iterator iter= bbBestMatches.begin(); iter != bbBestMatches.end(); iter++){
                                    cout <<"(previous, current frame boxId) :" << "(" <<iter->first <<", " << iter->second <<")" << endl; 
                                }
                                // display image
                                string previousWindow = "Bounding boxes on previous frame";
                                cv::namedWindow(previousWindow, 2);
                                cv::imshow(previousWindow, (dataBuffer.end() - 2)->cameraImg);

                                string currentWindow = "Bounding boxes on current frame";
                                cv::namedWindow(currentWindow, 2);
                                cv::imshow(currentWindow, (dataBuffer.end() - 1)->cameraImg);
                                cv::waitKey(0); 
                        }

                        // store matches in current data frame
                        (dataBuffer.end()-1)->bbMatches = bbBestMatches;

                        cout << "#8 : TRACK 3D OBJECT BOUNDING BOXES done" << endl;


                        /* COMPUTE TTC ON OBJECT IN FRONT */

                        // loop over all BB match pairs
                        for (auto it1 = (dataBuffer.end() - 1)->bbMatches.begin(); it1 != (dataBuffer.end() - 1)->bbMatches.end(); ++it1)
                        {
                            // find bounding boxes associates with current match
                            BoundingBox *prevBB, *currBB;
                            for (auto it2 = (dataBuffer.end() - 1)->boundingBoxes.begin(); it2 != (dataBuffer.end() - 1)->boundingBoxes.end(); ++it2)
                            {
                                if (it1->second == it2->boxID) // check wether current match partner corresponds to this BB
                                {
                                    currBB = &(*it2);
                                }
                            }

                            for (auto it2 = (dataBuffer.end() - 2)->boundingBoxes.begin(); it2 != (dataBuffer.end() - 2)->boundingBoxes.end(); ++it2)
                            {
                                if (it1->first == it2->boxID) // check wether current match partner corresponds to this BB
                                {
                                    prevBB = &(*it2);
                                }
                            }

                            // compute TTC for current match
                            if( currBB->lidarPoints.size()>0 && prevBB->lidarPoints.size()>0 ) // only compute TTC if we have Lidar points
                            {
                                //// STUDENT ASSIGNMENT
                                //// TASK FP.2 -> compute time-to-collision based on Lidar data (implement -> computeTTCLidar)
                                double ttcLidar; 
                                // computeTTCLidar(prevBB->lidarPoints, currBB->lidarPoints, sensorFrameRate, ttcLidar);
                                computeTTCLidar(prevBB->lidarPoints, currBB->lidarPoints, sensorFrameRate, ttcLidar, vehicleVel, vehicleAcc, TTCcalModel);
                                //// EOF STUDENT ASSIGNMENT

                                //// STUDENT ASSIGNMENT
                                //// TASK FP.3 -> assign enclosed keypoint matches to bounding box (implement -> clusterKptMatchesWithROI)
                                //// TASK FP.4 -> compute time-to-collision based on camera (implement -> computeTTCCamera)
                                double ttcCamera;
                                clusterKptMatchesWithROI(*currBB, (dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints, (dataBuffer.end() - 1)->kptMatches);                    
                                computeTTCCamera((dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints, currBB->kptMatches, sensorFrameRate, ttcCamera);
                                // //// EOF STUDENT ASSIGNMENT
                                
                                TTCresult.lidarBasedTTC.push_back(ttcLidar);
                                TTCresult.cameraBasedTTC.push_back(ttcCamera);
                                bVis = true;
                                bool bimgFileSave = false;
                                char buf[256];
                                if (bVis)
                                {
                                    cv::Mat visImg = (dataBuffer.end() - 1)->cameraImg.clone();
                                    showLidarImgOverlay(visImg, currBB->lidarPoints, P_rect_00, R_rect_00, RT, &visImg);
                                    cv::rectangle(visImg, cv::Point(currBB->roi.x, currBB->roi.y), cv::Point(currBB->roi.x + currBB->roi.width, currBB->roi.y + currBB->roi.height), cv::Scalar(0, 255, 0), 2);
                                    
                                    char str[200];
                                    sprintf(str, "TTC Lidar : %.3f s, TTC Camera : %.3f s", ttcLidar, ttcCamera);
                                    putText(visImg, str, cv::Point2f(80, 50), cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(0,0,255));

                                    string windowName = "Final Results : TTC";
                                    cv::namedWindow(windowName, 4);
                                    cv::imshow(windowName, visImg);
                                    if(bimgFileSave){
                                        // sprintf(buf, "../data/TTC-Lidar-Nofiltering_%ld.png",imgIndex);
                                        // sprintf(buf, "../data/TTC-Lidar-Distfiltering_%ld.png",imgIndex);
                                        // sprintf(buf, "../data/TTC-Lidar-Dist&Intensityfiltering_%ld.png",imgIndex);
                                        sprintf(buf, "../data/TTC-Lidar-Dist&Intensityfiltering_CAM%ld.png",imgIndex);
                                        // sprintf(buf, "../data/TTC-Camera-NofilteringMeanBased_%ld.png",imgIndex);
                                        // sprintf(buf, "../data/TTC-Camera-SB&Distfiltering_%ld.png",imgIndex);
                                        cv::imwrite(buf, visImg);
                                    }
                                    cout << "Press key to continue to next frame" << endl;
                                    cv::waitKey(0);
                                }
                                bVis = false;

                            } // eof TTC computation
                        } // eof loop over all BB matches            

                    }

            } // eof loop over all images
            TTCresultVec.push_back(TTCresult);
        }// eof loop over all descriptor options 
    }// eof loop over all detector options 

    if(bresultFileSave){
        for (auto it1= TTCresultVec.begin(); it1!=TTCresultVec.end(); it1++){
            TTCresult TTCresultTemp = (*it1);
            // Write doewn lidarBasedTTC
            if(!bFirstLine){
                resOut << "lidarBasedTTC: ";
            }

            for (int i=0; i<TTCresultTemp.lidarBasedTTC.size(); i++)
            {   
                if (i == TTCresultTemp.lidarBasedTTC.size()-1)
                    resOut << TTCresultTemp.lidarBasedTTC[i] << endl;
                else
                    resOut << TTCresultTemp.lidarBasedTTC[i] << "\t" ;
            }
            // Write doewn cameraBasedTTC
            if(!bFirstLine){
                resOut << it1->detectorType << "\t" << it1->descriptorType << endl;
                resOut << "cameraBasedTTC: ";
                bFirstLine = true;
            }
            for (int i=0; i<TTCresultTemp.lidarBasedTTC.size(); i++)
            {   
                if (i == TTCresultTemp.cameraBasedTTC.size()-1)
                    resOut << TTCresultTemp.cameraBasedTTC[i]  << endl;
                else
                    resOut << TTCresultTemp.cameraBasedTTC[i] << "\t";
            }
            bFirstLine = false;
        }
    }
    resOut.close();
    return 0;
}
