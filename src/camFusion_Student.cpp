
#include <algorithm>
#include <iostream>
// #include <math.h>
#include <numeric>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camFusion.hpp"
#include "dataStructures.h"

using namespace std;

// Create groups of Lidar points whose projection into the camera falls into the
// same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes,
                         std::vector<LidarPoint> &lidarPoints,
                         float shrinkFactor, cv::Mat &P_rect_xx,
                         cv::Mat &R_rect_xx, cv::Mat &RT) {

  // loop over all Lidar points and associate them to a 2D bounding box
  cv::Mat X(4, 1, cv::DataType<double>::type);
  cv::Mat Y(3, 1, cv::DataType<double>::type);

  for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1) {
    // assemble vector for matrix-vector-multiplication
    X.at<double>(0, 0) = it1->x;
    X.at<double>(1, 0) = it1->y;
    X.at<double>(2, 0) = it1->z;
    X.at<double>(3, 0) = 1;

    // project Lidar point into camera
    Y = P_rect_xx * R_rect_xx * RT * X;
    cv::Point pt;
    pt.x = Y.at<double>(0, 0) / Y.at<double>(0, 2); // pixel coordinates
    pt.y = Y.at<double>(1, 0) / Y.at<double>(0, 2);

    vector<vector<BoundingBox>::iterator>
        enclosingBoxes; // pointers to all bounding boxes which enclose the
                        // current Lidar point
    for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin();
         it2 != boundingBoxes.end(); ++it2) {
      // shrink current bounding box slightly to avoid having too many outlier
      // points around the edges
      cv::Rect smallerBox;
      smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
      smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
      smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
      smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

      // check wether point is within current bounding box
      if (smallerBox.contains(pt)) {
        enclosingBoxes.push_back(it2);
      }

    } // eof loop over all bounding boxes

    // check wether point has been enclosed by one or by multiple boxes
    if (enclosingBoxes.size() == 1) {
      // add Lidar point to bounding box
      enclosingBoxes[0]->lidarPoints.push_back(*it1);
    }

  } // eof loop over all Lidar points
}

void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize,
                   cv::Size imageSize, bool bWait) {
  // create topview image
  cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

  for (auto it1 = boundingBoxes.begin(); it1 != boundingBoxes.end(); ++it1) {
    // create randomized color for current 3D object
    cv::RNG rng(it1->boxID);
    cv::Scalar currColor = cv::Scalar(rng.uniform(0, 150), rng.uniform(0, 150),
                                      rng.uniform(0, 150));

    // plot Lidar points into top view image
    int top = 1e8, left = 1e8, bottom = 0.0, right = 0.0;
    float xwmin = 1e8, ywmin = 1e8, ywmax = -1e8;
    for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end();
         ++it2) {
      // world coordinates
      float xw =
          (*it2).x; // world position in m with x facing forward from sensor
      float yw = (*it2).y; // world position in m with y facing left from sensor
      xwmin = xwmin < xw ? xwmin : xw;
      ywmin = ywmin < yw ? ywmin : yw;
      ywmax = ywmax > yw ? ywmax : yw;

      // top-view coordinates
      int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
      int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

      // find enclosing rectangle
      top = top < y ? top : y;
      left = left < x ? left : x;
      bottom = bottom > y ? bottom : y;
      right = right > x ? right : x;

      // draw individual point
      cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
    }

    // draw enclosing rectangle
    cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom),
                  cv::Scalar(0, 0, 0), 2);

    // augment object with some key data
    char str1[200], str2[200];
    sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
    putText(topviewImg, str1, cv::Point2f(left - 250, bottom + 50),
            cv::FONT_ITALIC, 2, currColor);
    sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax - ywmin);
    putText(topviewImg, str2, cv::Point2f(left - 250, bottom + 125),
            cv::FONT_ITALIC, 2, currColor);
  }

  // plot distance markers
  float lineSpacing = 2.0; // gap between distance markers
  int nMarkers = floor(worldSize.height / lineSpacing);
  for (size_t i = 0; i < nMarkers; ++i) {
    int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) +
            imageSize.height;
    cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y),
             cv::Scalar(255, 0, 0));
  }

  // display image
  string windowName = "3D Objects";
  cv::namedWindow(windowName, 1);
  cv::imshow(windowName, topviewImg);

  if (bWait) {
    cv::waitKey(0); // wait for key to be pressed
  }
}

// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &boundingBox,
                              std::vector<cv::KeyPoint> &kptsPrev,
                              std::vector<cv::KeyPoint> &kptsCurr,
                              std::vector<cv::DMatch> &kptMatches) {

  /*
We need 2 tolerance ratio to filter out the outliers keypoints, kpt distance >
maxRatio * mean distance or kpt distance < minRatio * mean distance would be
filtered out.
*/
  float maxToleranceRatio = 1.5;
  float minToleranceRatio = 0.5;

  int RoiX = boundingBox.roi.x;
  int RoiY = boundingBox.roi.y;
  int RoiWidth = boundingBox.roi.width;
  int RoiHeight = boundingBox.roi.height;

  double distance = 0; // We use it to accumulate the sum of euclidian
  distance
      // of every match points, then compute the mean of it
      vector<double>
          distanceList;

  for (auto it = kptMatches.begin(); it != kptMatches.end(); it++) {
    if (kptsCurr[it->trainIdx].pt.x >= RoiX &&
        kptsCurr[it->trainIdx].pt.x <= (RoiX + RoiWidth) &&
        kptsCurr[it->trainIdx].pt.y >= RoiY &&
        kptsCurr[it->trainIdx].pt.y <= (RoiY + RoiHeight)) {
      boundingBox.kptMatches.push_back(*it);
      distance += sqrt(
          pow(((kptsCurr[it->trainIdx].pt.x - kptsPrev[it->queryIdx].pt.x) *
               1.0),
              2.0) +
          pow(((kptsCurr[it->trainIdx].pt.y - kptsPrev[it->queryIdx].pt.y) *
               1.0),
              2.0));
      distanceList.push_back(distance);
    }
  }

  distance /= (boundingBox.kptMatches.size() * 1.0);
  double maxDistance = distance * maxToleranceRatio;
  double minDistance = distance * minToleranceRatio;
  size_t i = 0; // we use i to iterate distanceList, while using iterator it
  to
      // iterate kptMatches.
      for (auto it = boundingBox.kptMatches.begin();
           it != boundingBox.kptMatches.end(); i++) {
    if (distanceList[i] < minDistance || distanceList[i] > maxDistance) {
      it = boundingBox.kptMatches.erase(it);
    } else {
      it++;
    }
  }
}

// Compute time-to-collision (TTC) based on keypoint correspondences in
// successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev,
                      std::vector<cv::KeyPoint> &kptsCurr,
                      std::vector<cv::DMatch> kptMatches, double frameRate,
                      double &TTC, cv::Mat *visImg) {
  // ...
}

void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate,
                     double &TTC) {
  // frameRate is frames per second for Lidar and camera

  double dT = 1.0 / frameRate;
  vector<double> XPrev;
  vector<double> XCurr;
  double minXPrev, minXCurr;
  for (auto it = lidarPointsPrev.begin(); it != lidarPointsPrev.end(); ++it) {
    XPrev.push_back(it->x);
  }
  for (auto it = lidarPointsCurr.begin(); it != lidarPointsCurr.end(); ++it) {
    XCurr.push_back(it->x);
  }
  sort(XPrev.begin(), XPrev.end());
  sort(XCurr.begin(), XCurr.end());
  minXPrev = accumulate(XPrev.begin(), XPrev.begin() + 5, 0.0) / 5.0;
  minXCurr = accumulate(XCurr.begin(), XCurr.begin() + 5, 0.0) / 5.0;
  // compute TTC from both measurements
  TTC = minXCurr * dT / (minXPrev - minXCurr);
}

void matchBoundingBoxes(std::vector<cv::DMatch> &matches,
                        std::map<int, int> &bbBestMatches, DataFrame &prevFrame,
                        DataFrame &currFrame) {
  int PrevBoxNumber = prevFrame.boundingBoxes.size();
  vector<vector<int>> RawMatch; // elem in vector is <current match bbox,
                                // previous match bbox, kpt match number>.

  // Push the kptMatches to each BoundingBox in the currFrame, and pop out the
  // element from matches.
  // should design better way to match points in boxes form each frame. I knew
  // such a method in python...
  for (auto &box : currFrame.boundingBoxes) {
    vector<int> MatchNumber(PrevBoxNumber, 0);
    int RoiX = box.roi.x;
    int RoiY = box.roi.y;
    int RoiWidth = box.roi.width;
    int RoiHeight = box.roi.height;
    for (auto it = matches.begin(); it != matches.end();) {
      int currIndex = it->trainIdx;
      int ptX = currFrame.keypoints[currIndex].pt.x;
      int ptY = currFrame.keypoints[currIndex].pt.y;
      if (ptX >= RoiX && ptX <= (RoiX + RoiWidth) && ptY >= RoiY &&
          ptY <= (RoiY + RoiHeight)) {
        box.kptMatches.push_back(*it);
        int prevIndex = it->queryIdx;
        int prevPtX = prevFrame.keypoints[prevIndex].pt.x;
        int prevptY = prevFrame.keypoints[prevIndex].pt.y;
        for (int i = 0; i < prevFrame.boundingBoxes.size(); i++) {
          if (prevPtX >= prevFrame.boundingBoxes[i].roi.x &&
              prevPtX <= (prevFrame.boundingBoxes[i].roi.x +
                          prevFrame.boundingBoxes[i].roi.width) &&
              prevptY >= prevFrame.boundingBoxes[i].roi.y &&
              prevptY <= (prevFrame.boundingBoxes[i].roi.y +
                          prevFrame.boundingBoxes[i].roi.height)) {
            MatchNumber[i] += 1;
            break;
          }
        }
        it = matches.erase(it);
      } else {
        ++it;
      }
    }
    auto maxMatchItr = std::max_element(MatchNumber.begin(), MatchNumber.end());
    int maxMatchNum = *maxMatchItr;
    int PervBoxId = std::distance(MatchNumber.begin(), maxMatchItr);
    vector<int> temp{box.boxID, PervBoxId, maxMatchNum};
    bool flag =
        (maxMatchNum > 0) ? true : false; // flag is true if the current bbox
                                          // has matching kpt in previous bbox.
    if (flag == true) {
      for (auto i = RawMatch.begin(); i != RawMatch.end(); i++) {
        if ((*i)[1] == PervBoxId && maxMatchNum > (*i)[2]) {
          RawMatch.erase(i);
          break;
        } else if ((*i)[1] == PervBoxId && maxMatchNum < (*i)[2]) {
          flag = false;
        }
      }
    }

    if (flag == true) {
      RawMatch.push_back(temp);
    }
    /*
    std::cout << "Bbox: " << box.boxID << " has: " << box.kptMatches.size()
              << " matching points. Its corresponding previous box ID "
                 "is: No."
              << PervBoxId << " " << maxMatchNum << " points" << std::endl;
    */
  }

  // Define map< int PreviousBoxId,  int currentBoxID bbBestMatches
  for (auto i : RawMatch) {
    std::cout << "Bbox: " << i[0]
              << "'s corresponding previous box ID "
                 "is: No."
              << i[1] << " "
              << "has kptMatch of: " << i[2] << std::endl;
    bbBestMatches.insert(pair<int, int>(i[1], i[0]));
  }
}
