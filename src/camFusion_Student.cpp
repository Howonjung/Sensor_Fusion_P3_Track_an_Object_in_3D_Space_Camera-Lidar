
#include <iostream>
#include <algorithm>
#include <numeric>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camFusion.hpp"
#include "dataStructures.h"

using namespace std;


// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT)
{
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1)
    {
        // assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        // pixel coordinates
        pt.x = Y.at<double>(0, 0) / Y.at<double>(2, 0); 
        pt.y = Y.at<double>(1, 0) / Y.at<double>(2, 0); 

        vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point
        for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
        {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // check wether point is within current bounding box
            if (smallerBox.contains(pt))
            {
                enclosingBoxes.push_back(it2);
            }

        } // eof loop over all bounding boxes

        // check wether point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1)
        { 
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*it1);
        }

    } // eof loop over all Lidar points
}

/* 
* The show3DObjects() function below can handle different output image sizes, but the text output has been manually tuned to fit the 2000x2000 size. 
* However, you can make this function work for other sizes too.
* For instance, to use a 1000x1000 size, adjusting the text positions by dividing them by 2.
*/
void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait)
{
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for(auto it1=boundingBoxes.begin(); it1!=boundingBoxes.end(); ++it1)
    {
        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0,150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top=1e8, left=1e8, bottom=0.0, right=0.0; 
        float xwmin=1e8, ywmin=1e8, ywmax=-1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            xwmin = xwmin<xw ? xwmin : xw;
            ywmin = ywmin<yw ? ywmin : yw;
            ywmax = ywmax>yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top<y ? top : y;
            left = left<x ? left : x;
            bottom = bottom>y ? bottom : y;
            right = right>x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom),cv::Scalar(0,0,0), 2);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left-250, bottom+50), cv::FONT_ITALIC, 2, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax-ywmin);
        putText(topviewImg, str2, cv::Point2f(left-250, bottom+125), cv::FONT_ITALIC, 2, currColor);  
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    string windowName = "3D Objects";
    cv::namedWindow(windowName, 1);
    cv::imshow(windowName, topviewImg);

    if(bWait)
    {
        cv::waitKey(0); // wait for key to be pressed
    }
}

// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{   
    // calculate distance mean and standard deviation between matched points
    double distSum = 0, distSqSum = 0; 
    double distMean, distStd, dist;
    for(auto it = kptMatches.begin(); it!=kptMatches.end(); it++){
        // get current keypoint and its matched partner in the prev. frame
        cv::KeyPoint kpCurr = kptsCurr.at(it->trainIdx);
        cv::KeyPoint kpPrev = kptsPrev.at(it->queryIdx);
        // compute distances and distance ratios
        distSum += cv::norm(kpCurr.pt - kpPrev.pt);
        distSqSum += std::pow(cv::norm(kpCurr.pt - kpPrev.pt),2);
    }
    distMean = distSum / kptMatches.size();
    distStd = std::sqrt(distSqSum/kptMatches.size() - std::pow(distMean,2));

    // Variables for distance filtering
    float distThreshold = 1.7; // 1e9, 1.7

    // Variables for small box filtering
    float shrinkFactor = 0.10; 
    cv::Rect smallerBox;
    smallerBox.x = boundingBox.roi.x + shrinkFactor * boundingBox.roi.width / 2.0;
    smallerBox.y = boundingBox.roi.y + shrinkFactor * boundingBox.roi.height / 2.0;
    smallerBox.width = boundingBox.roi.width * (1 - shrinkFactor);
    smallerBox.height = boundingBox.roi.height * (1 - shrinkFactor);
    for(auto it = kptMatches.begin(); it!=kptMatches.end(); it++){
        // get current keypoint and its matched partner in the prev. frame
        cv::KeyPoint kpCurr = kptsCurr.at(it->trainIdx);
        cv::KeyPoint kpPrev = kptsPrev.at(it->queryIdx);

        // add kptMatches only if meets small box and distance condition
        if(smallerBox.contains(kpCurr.pt) && smallerBox.contains(kpPrev.pt)){
            // compute distances and distance ratios
            dist = cv::norm(kpCurr.pt - kpPrev.pt);
            if( (dist > distMean - distThreshold*distStd) && (dist < distMean + distThreshold*distStd)){
                boundingBox.kptMatches.push_back(*it);
            }
        }
        
    }
    cout << "=======================================" << endl;
    cout << "boundingBox.kptMatches size " << boundingBox.kptMatches.size() << endl;
    cout << "=======================================" << endl;

}

// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, 
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{
    // compute distance ratios between all matched keypoints
    vector<double> distRatios; // stores the distance ratios for all keypoints between curr. and prev. frame
    for (auto it1 = kptMatches.begin(); it1 != kptMatches.end() - 1; ++it1)
    { // outer keypoint loop

        // get current keypoint and its matched partner in the prev. frame
        cv::KeyPoint kpOuterCurr = kptsCurr.at(it1->trainIdx);
        cv::KeyPoint kpOuterPrev = kptsPrev.at(it1->queryIdx);

        for (auto it2 = kptMatches.begin() + 1; it2 != kptMatches.end(); ++it2)
        { // inner keypoint loop

            double minDist = 100.0; // min. required distance

            // get next keypoint and its matched partner in the prev. frame
            cv::KeyPoint kpInnerCurr = kptsCurr.at(it2->trainIdx);
            cv::KeyPoint kpInnerPrev = kptsPrev.at(it2->queryIdx);

            // compute distances and distance ratios
            double distCurr = cv::norm(kpOuterCurr.pt - kpInnerCurr.pt);
            double distPrev = cv::norm(kpOuterPrev.pt - kpInnerPrev.pt);

            if (distPrev > std::numeric_limits<double>::epsilon() && distCurr >= minDist)
            { // avoid division by zero

                double distRatio = distCurr / distPrev;
                distRatios.push_back(distRatio);
            }
        } // eof inner loop over all matched kpts
    }     // eof outer loop over all matched kpts

    // only continue if list of distance ratios is not empty
    if (distRatios.size() == 0)
    {
        TTC = NAN;
        return;
    }

    // compute camera-based TTC from distance ratios (mean & median based)
    // double meanDistRatio = std::accumulate(distRatios.begin(), distRatios.end(), 0.0) / distRatios.size();
    double medianDistRatio;
    std::sort(distRatios.begin(), distRatios.end());
    if (distRatios.size() %2 == 1){
        medianDistRatio = distRatios[int(distRatios.size()/2)];
    }
    else{
        medianDistRatio = (distRatios[int(distRatios.size()/2)] + distRatios[int((distRatios.size()-1)/2)])/2.0 ;
    }

    double dT = 1 / frameRate;
    // TTC = -dT / (1 - meanDistRatio);
    TTC = -dT / (1 - medianDistRatio);
}

// Compute time-to-collision (TTC) based on lidar minX and intensity values
void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC, double &vehicleVel, double &vehicleAcc, int TTCcalModel)
{

    // Calculate mean & standard deviation of lidarPointsPrev and lidarPointsCurr x & intensity values 
    // to get rid of outliers which are not witin ± (Dist or Intensity)Threshold * 1 standard deviation vlaue.

    double lidarPrevXMean,lidarCurrXMean,lidarPrevXStd,lidarCurrXStd;
    double lidarPrevXSum =0 , lidarCurrXSum = 0, lidarPrevXSqSum =0 , lidarCurrXSqSum = 0;
    double lidarPrevISum =0, lidarPrevISqSum =0, lidarCurrISum =0, lidarCurrISqSum =0  ;
    double lidarPrevIMean, lidarPrevIStd, lidarCurrIMean, lidarCurrIStd;
    for (auto it1 = lidarPointsPrev.begin(); it1 != lidarPointsPrev.end(); ++it1){ 
        lidarPrevXSum += it1->x;
        lidarPrevXSqSum += std::pow(it1->x,2);
        lidarPrevISum += it1->r;
        lidarPrevISqSum += std::pow(it1->r,2);

    }

    for (auto it2 = lidarPointsCurr.begin(); it2 != lidarPointsCurr.end(); ++it2){
        lidarCurrXSum += it2->x;
        lidarCurrXSqSum += std::pow(it2->x,2);
        lidarCurrISum += it2->r;
        lidarCurrISqSum += std::pow(it2->r,2);
    }
    lidarPrevXMean = lidarPrevXSum / lidarPointsPrev.size();
    lidarCurrXMean = lidarCurrXSum / lidarPointsCurr.size();
    lidarPrevXStd = std::sqrt(lidarPrevXSqSum/lidarPointsPrev.size() - std::pow(lidarPrevXMean,2));
    lidarCurrXStd = std::sqrt(lidarCurrXSqSum/lidarPointsCurr.size() - std::pow(lidarCurrXMean,2));
    lidarPrevIMean = lidarPrevISum / lidarPointsPrev.size();
    lidarCurrIMean = lidarCurrISum / lidarPointsCurr.size();
    lidarPrevIStd = std::sqrt(lidarPrevISqSum/lidarPointsPrev.size() - std::pow(lidarPrevIMean,2));
    lidarCurrIStd = std::sqrt(lidarCurrISqSum/lidarPointsCurr.size() - std::pow(lidarCurrIMean,2));

    // auxiliary variables
    float dT = 1.0/frameRate; // time between two measurements in seconds
    float DistThreshold = 2; //orig 2 
    float intensityThreshold = 1.6; // orig 1.6
    // find closest distance to Lidar points within ego lane
    double minXPrev = 1e9, minXCurr = 1e9;
    double secondminXPrev = 1e9, secondminXCurr = 1e9;
    for (auto it = lidarPointsPrev.begin(); it != lidarPointsPrev.end(); ++it)
    {   
        if( it->x > (lidarPrevXMean - DistThreshold*lidarPrevXStd) && it->x < (lidarPrevXMean + DistThreshold*lidarPrevXStd) &&
        it->r > (lidarPrevIMean - intensityThreshold*lidarPrevIStd) && it->r < (lidarPrevIMean + intensityThreshold*lidarPrevIStd) )
            minXPrev = (minXPrev > it->x) ? it->x : minXPrev;
    }

    for (auto it = lidarPointsCurr.begin(); it != lidarPointsCurr.end(); ++it)
    {   
        if( it->x > (lidarCurrXMean - DistThreshold*lidarCurrXStd) && it->x < (lidarCurrXMean + DistThreshold*lidarCurrXStd) &&
        it->r > (lidarCurrIMean - intensityThreshold*lidarCurrIStd) && it->r < (lidarCurrIMean + intensityThreshold*lidarCurrIStd) )
            minXCurr = (minXCurr > it->x)  ? it->x : minXCurr;
    }

    // if vehicleAcc is not yet initialized, update vehicleVel
    if (vehicleVel != -1e9 && vehicleAcc == -1e9){
        double vehicleVelCurr = (minXPrev - minXCurr) / dT;
        // Acceleration = (Δvelocity) / (Δ t)
        vehicleAcc = (vehicleVelCurr - vehicleVel ) / dT;
        cout << "====================================" << endl;
        cout <<"vehicleVelCurr: "<<vehicleVelCurr << " PrevVehicleVel: " << vehicleVel << " dT: "<< dT <<" vehicleAcc " << vehicleAcc << endl;
        cout << "====================================" << endl;
    }
    if(TTCcalModel == 0) // 0 - Constant Velocity Model 
        TTC = minXCurr * dT / (minXPrev - minXCurr);
    else{ // 1 - Constant Acceleration Model
        // compute TTC from both measurements
        if(vehicleAcc == -1e9)
            TTC = minXCurr * dT / (minXPrev - minXCurr);
        else{
            
            double a,b,c,d;
            a = 1.0;
            b = vehicleVel/(0.5*vehicleAcc);
            c = -minXCurr/(0.5*vehicleAcc);

            /* Quadratic equation: f(x) = ax² + bx + c 
                discriminant: d = b*b - 4ac 
                x1 = (-b + sqrt(b * b - 4 * a * c)) / (2 * a); 
                x2 = (-b - sqrt(b * b - 4 * a * c)) / (2 * a); 
            */
            d = b*b - 4*a*c;
            if(d>0){
                double TTC1, TTC2;
                TTC1 = (-b - sqrt(b*b - 4*a*c)) / 2*a;
                TTC2 = (-b + sqrt(b*b - 4*a*c)) / 2*a;
                if (TTC1>0 && TTC2>0)
                    TTC = std::min(TTC1, TTC2);
                else if ((TTC1>0 && TTC2<0))
                    TTC = TTC1;
                else if ((TTC1<0 && TTC2>0))
                    TTC = TTC2;
                else
                    cout << "TTC is not able to calculate!" << endl;
            }
            else if(d==0)
                TTC = b / (-2*a);
            else
                cout << "TTC is not able to calculate!" << endl;
        }

        // cout << "vehicleAcc: " <<  vehicleAcc  << " vehicleVel: " <<vehicleVel  <<" minXCurr: "<< minXCurr <<" TTC: " << TTC << endl ;
        
        // if vehicleVel is not yet initialized, update vehicleVel based on constant velocity model
        if (vehicleVel == -1e9 && vehicleAcc == -1e9)
                vehicleVel = (minXPrev - minXCurr) / dT;
        // if both vehicleVel & vehicleAcc are initialized, update vehicleVel based on constant acceraltion model
        else
            // V(t+ Δt) = v(t) + a*Δt
            vehicleVel =  vehicleVel + vehicleAcc*dT;
    }
    
}

// Compute matching bounding box pair between prvious & current frame
void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{   
    float IOURatio, maxIOURatio; // Intersection over union(IOU)
    float IOUThreshold = 0.7;
    std::vector<pair<vector<BoundingBox>::iterator, vector<BoundingBox>::iterator>> boundingBoxPairs;
    for (auto it1 = prevFrame.boundingBoxes.begin(); it1<prevFrame.boundingBoxes.end(); it1++){
        maxIOURatio = -1e9;
        for (auto it2 = currFrame.boundingBoxes.begin(); it2<currFrame.boundingBoxes.end(); it2++){
            
            // Calculate Intersection over union(IOU) ratio
            IOURatio = ((it1->roi) & (it2->roi)).area() / float( ((it1->roi) | (it2->roi) ).area() );
            // Replace maxIOURatio value, if it meets condition.
            if(IOURatio > maxIOURatio) {
                maxIOURatio = IOURatio;
                boundingBoxPairs.push_back(make_pair(it1,it2));
            }
        }
        // cout <<"maxIOURatio: "<<maxIOURatio<<endl;
        // only if maxIOURatio is larger than IOUThreshold, insert matched box which has maximum IOURatio id pair data.
        if(maxIOURatio > IOUThreshold){
            bbBestMatches.insert(pair<int,int>(boundingBoxPairs.back().first->boxID, boundingBoxPairs.back().second->boxID));
        }
        boundingBoxPairs.clear();
    }
}
