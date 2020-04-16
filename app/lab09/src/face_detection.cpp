#include <iostream>
#include <vector>
#include "opencv2/highgui.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/imgproc.hpp"

using namespace std;
using namespace cv;

int main()
{
    //Read image and xml face_cascade
    Mat FaceImage = imread("../data/faceimage.jpg");

    String face_cascade_name = "../data/haarcascade_frontalface_default.xml";
    String eye_cascade_name = "../data/haarcascade_eye.xml";

    CascadeClassifier face_cascade;
    CascadeClassifier eye_cascade;

    face_cascade.load(face_cascade_name);
    eye_cascade.load(eye_cascade_name);
    
    vector<Rect> faces;
    Mat gray_Image;
    //Converts to a gray image & Histogram equalization
    cv::cvtColor(FaceImage, gray_Image, CV_BGR2GRAY);
    cv::equalizeHist(gray_Image, gray_Image);

    //face Detection code
    face_cascade.detectMultiScale(gray_Image, faces, 1.1, 5);
    for (size_t i = 0; i < faces.size(); i++)
    {
        rectangle(FaceImage, faces[i], Scalar(255, 0, 0), 2); //draw Face detection


        //Draw FaceROI
        Mat FaceROI = gray_Image(faces[i]); 
        std::vector<Rect> eyes;

        eye_cascade.detectMultiScale(FaceROI, eyes);
        //draw Circle
        for (size_t j = 0; j < eyes.size(); j++)
        {
            Point center(faces[i].x + eyes[j].x + eyes[j].width*0.5, faces[i].y + eyes[j].y + eyes[j].height*0.5);
            circle(FaceImage, center, 13, Scalar(0, 255, 0), 2);
        }
    }
    cv::imshow("Detect Image", FaceImage);
    cv::waitKey(0);
    cv::destroyAllWindows();
}
    