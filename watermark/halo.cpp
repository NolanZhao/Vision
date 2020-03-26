#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cmath>
#include <iostream>

using namespace std;
using namespace cv;


bool hiddenWaterMark(Mat &input_img, Mat &output_img, int strength, int mode)
{
    try {
        cv::imshow("input_img", input_img);
        cv::waitKey(0);

        int h = input_img.rows;
        int w = input_img.cols;
        int center_x = w / 2;
        int center_y = h / 2;
        int radius = min(center_x, center_y);
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int distance = pow( y - center_y, 2) + pow(x - center_x, 2);
                if (distance < radius * radius) {
                    int result = (int) (strength * (1.0 - sqrt(distance) / radius));
                    output_img.at<Vec3b>(y, x)[0] = min(255, max(0, output_img.at<Vec3b>(y, x)[0] + result));
                    output_img.at<Vec3b>(y, x)[1] = min(255, max(0, output_img.at<Vec3b>(y, x)[1] + result));
                    output_img.at<Vec3b>(y, x)[2] = min(255, max(0, output_img.at<Vec3b>(y, x)[2] + result));
                }
            }
        }

        cv::imshow("output_img", output_img);
        cv::waitKey(0);

        imwrite("/Users/nolan/Desktop/w2.jpg", output_img);

        return true;
    } catch (...) {
        return false;
    }
}


int main( int argc, char** argv)
{
    Mat input_img = imread("/Users/nolan/Desktop/2.jpg");
    Mat out_img = Mat::zeros( input_img.size(), input_img.type() );
    input_img.copyTo(out_img);
    bool result = hiddenWaterMark(input_img, out_img, 80, 0);
    cout << result << endl;
    return 0;
}
