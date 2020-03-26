#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <cmath>
#include <iostream>

using namespace std;
using namespace cv;


bool hiddenWaterMark(Mat &input_img, Mat &output_img, int mode)
{
    try {
        cv::imshow("input_img", input_img);
        cv::waitKey(0);

        Mat hsv_img = Mat::zeros( input_img.size(), input_img.type() );
        cvtColor(input_img, hsv_img, COLOR_BGR2HSV);

//        cout << input_img.size() << endl;
//        cout << input_img.type() << endl;

        int h = input_img.rows;
        int w = input_img.cols;
        int center_x = (h - 1) / 2;
        int center_y = (w - 1) / 2;
//        cout << center_x << "\t" << center_y << endl;
        int bandwidth = 100;

        for (int i = 0; i < h; i++)
            for (int j = 0; j < w; j++) {
                if ( (int(sqrt(pow(i - center_x, 2) + pow(j - center_y, 2)) / bandwidth)) % 2 == 0 ) {
                    hsv_img.at<Vec3b>(i, j)[0] = hsv_img.at<Vec3b>(i, j)[0] * 0.9;
                    hsv_img.at<Vec3b>(i, j)[1] = hsv_img.at<Vec3b>(i, j)[1] * 0.9;
                }
            }

        cv::imshow("hsv_img", hsv_img);
        cv::waitKey(0);

        cvtColor(hsv_img, output_img, COLOR_HSV2BGR);

        cv::imshow("output_img", output_img);
        cv::waitKey(0);


        imwrite("/Users/nolan/Desktop/w1.png", output_img);

        return true;
    } catch (...) {
        return false;
    }
}


int main( int argc, char** argv)
{
    Mat input_img = imread("/Users/nolan/Desktop/1.png");
    Mat out_img = Mat::zeros( input_img.size(), input_img.type() );
    bool result = hiddenWaterMark(input_img, out_img, 0);
    cout << result << endl;
    return 0;
}
