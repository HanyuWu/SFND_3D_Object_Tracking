# SFND 3D Object Tracking

## FP.1 Match 3D Objects
My program can print out the matching bounding box ID for both current and previous frame. The way I match 3D obkects are to iterate each current bounding box, and find the corresponding keypoints inside the box's ROI. After a keypoint has been found, I loop through the previous bounding boxes, and find if its correspondence is inside a certain bounding box from the previous frame. A vector would be defined to record those info (the number of kptmatch in different previous bounding box). At the end of each iteration, we find out which 2 box have the most correspondence, and record the number of keypoints match too.
```c++
auto maxMatchItr = std::max_element(MatchNumber.begin(), MatchNumber.end());
int maxMatchNum = *maxMatchItr;
int PervBoxId = std::distance(MatchNumber.begin(), maxMatchItr);
```
Here, MatchNumber is the vector which record the number of keypoints matches for each previous bounding boxes in order.
After we iterated through the current bounding boxes. We compared the every bounding box and find out the pair (current and previous bounding box) with the highest number of keypoint correspondences. 

To speed up the program a little, I erase the key points matches which have been recognized as the keypoints inside the box in line 327 of camFusion_Student.cpp.
```c++
it = matches.erase(it);
```

## FP.2 Compute Lidar-based TTC
We get the lidar points from the two corresponding bounding boxes (from previous and current frame). Then I sort out the lidar points from those two boxes, finding out the 5 lidar points with the smallest x value from both boxes. Then I compute their x values' means, and thus get the estimated TTC we need. I choose computing the 5 most nearest lidar points' mean to reduce the impact of outliners.
```c++
minXPrev = accumulate(XPrev.begin(), XPrev.begin() + 5, 0.0) / 5.0;   //  based on the last 5 samllest x values.
minXCurr = accumulate(XCurr.begin(), XCurr.begin() + 5, 0.0) / 5.0;
// compute TTC from both measurements
TTC = minXCurr * dT / (minXPrev - minXCurr);
```

## FP.3 Associate Keypoint Correspondences with Bounding Boxes
For this task, we have to rule out the outlier matches based on the euclidean distance between them in relation to all the matches in the bounding box. I decide two ratios which are max tolerance ratio and min tolerance ratio. I first compute the euclian distance for all the Keypoint matching pairs, then compute the mean of those distances. We define:  

* max distance = mean distance * max tolerance ratio;
* min distance = mean distance * min tolerance ratio;

Distance for each pari should be below the max distance and over the min Distance. Unqualified points would be removed and the qualified ones would be pushed into the kptMatches vector of the current bounding box. That's how I filter the Keypoint Correspondences. 

## FP.4 Compute Camera-based TTC
The method for deriving camera-based TTC is as follow:<br>
<div style="width:800px; margin:0 auto;">
<img src="../SFND_3D_Object_Tracking/examples/draggedimage.png" width="800" height="400">
<img src="../SFND_3D_Object_Tracking/examples/draggedimage-1.png" width="800" height="400">
</div><br>
The tricky part is that we don't know how well our detectors/decriptors are. We are not sure that the matching keypoints are the exact matching pair like the picture shown above. I think the angle between two cars' directions will change while driving, and that can cause system error.<br>

I extract 4 or 5 distance ratios right at the middle of all the distance ratios (even or odd), and compute their mean as the distance ratio we need to reduce outliers' influence.

```c++
if (distRatios.size() < 5) {
    medDistRatio = accumulate(distRatios.begin(), distRatios.end(), 0.0) /
                   (double)distRatios.size();
  } else {
    medDistRatio = distRatios.size() % 2 == 0
                       ? (accumulate(distRatios.begin() + medIndex - 2,
                                     distRatios.begin() + medIndex + 2, 0.0) /
                          4.0)
                       : (accumulate(distRatios.begin() + medIndex - 2,
                                     distRatios.begin() + medIndex + 3, 0.0) /
                          5.0);
  } 

  double dT = 1 / frameRate;
  TTC = -dT / (1 - medDistRatio);
```

## FP.5 Performance Evaluation 1
The result of the TTC estimate based on the Lidar sensor is shown [here](./output/LidarTTC.csv). The estimates seems off at the frame 11,12 and 16,17. I can clearly see there is outlier points in the frame 11 and 16. I think that is why it make the estimate of TTC off at the time frame No.11 and No.12. It also affect the estimate at the time frame right after them because the TTC are based on 2 successive frames, which make the TTC at frame 12 and 17 off as well.

## FP.6 Performance Evaluation 2
The result of the TTC estimates based on the camera is shown [here](./output/camera.csv). I think alomst all the TTC estimates based on the camera perform really poor. The best one is AKAZE/SIFT as detector/descriptor. The comparison between the results from AKAZE/SIFT and the lidars can be found [here](./output/AkazeSiftLidar.csv).

I think the reason why some TTC estimates based on camera are off is that feature point extraction is somewhat not robust. The angle between two cars' direction, all kinds of occlusion, light and many others can affect the performance of feature point extraction/matching. The TTC estimates at Frame No. 17 is off, and there is an outliner lidar point in that frame as we can recall. I think maybe some kind of occlusion exist at that particular time frame.

