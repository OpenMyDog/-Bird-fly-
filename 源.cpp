#include <opencv2/opencv.hpp>
#include <iostream>

constexpr auto LowWindows = "原视频";
constexpr auto fangsheWindows = "仿射变换";
constexpr auto NowWindows = "处理后视频";

constexpr auto PhoneW = 1920;//手机分辨率
constexpr auto PhoneH = 1080;//手机分辨率

using namespace cv;
using namespace std;

int g_Canny = 100;//边缘检测因子
int g_zuizhi = 10;//方框面积滑动条值
int g_zuizhiMax = 100;//最大描绘方框面积
bool fangshe = false;//是否开启仿射变换
Rect g_rectangle;//记录鼠标位置
int g_click = 0;//点击次数

vector<vector<Point>>contours;
vector<Vec4i> hierarchy;
Point2f srcTriangle[4];
Point2f dstTriangle[4];

void on_zuizhi(int, void*);
void on_click(int event, int x, int y, int flags, void* param);

int main() {

	Mat srcimg = imread("1.jpg");

	VideoCapture capture(0);
	capture >> srcimg;

	if (!srcimg.data) { cout << "图片错误" << endl; cv::waitKey(0); return -1; }

	//创建视图窗口
	namedWindow(LowWindows, WINDOW_NORMAL);
	namedWindow(NowWindows, WINDOW_NORMAL);

	//初始化鼠标位置参数
	g_rectangle = Rect(-1, -1, 0, 0);

	//创建鼠标回调事件
	setMouseCallback(LowWindows, on_click, (void*)&srcimg);

	//创建进度条回调事件
	createTrackbar("阀值", NowWindows, &g_Canny, 254, NULL);
	createTrackbar("最值", NowWindows, &g_zuizhi, 100, on_zuizhi);

	//仿射变换参数初始化
	dstTriangle[0] = Point2f(0, 0);
	dstTriangle[1] = Point2f(static_cast<float>(srcimg.cols - 1),0);
	dstTriangle[2] = Point2f(0, static_cast<float>(srcimg.rows - 1));
	dstTriangle[3] = Point2f(static_cast<float>(srcimg.cols - 1), static_cast<float>(srcimg.rows - 1));

	while (true)
	{

		Mat grayimg;
		capture >> srcimg;

		Mat threshold_output;
		Mat warpMat;
		Mat dstImg;

		//逆时针旋转90度
		//transpose(srcimg, srcimg);
		//flip(srcimg, srcimg, 0);

		

		if (fangshe)//如果开始仿射
		{
			warpMat = getAffineTransform(srcTriangle, dstTriangle);
			warpAffine(srcimg, dstImg, warpMat, dstImg.size());
		}else {
			dstImg = srcimg;
		}

		imshow(fangsheWindows, dstImg);

		//描绘点击的点
		if (fangshe||g_click) {//为0时不进行描绘

			for (size_t i = 0; i < g_click; i++)
			{
				circle(srcimg, srcTriangle[i], 3, Scalar(0, 0, 255), -1);
			}
		}
		
		imshow(LowWindows, srcimg);

		//转换成灰度图
		cvtColor(dstImg, grayimg, COLOR_BGR2GRAY);
		//方框模糊
		blur(grayimg, grayimg, Size(3, 3));
		//二值化
		threshold(grayimg, threshold_output, g_Canny, 255, THRESH_BINARY);
		//寻找轮廓
		findContours(threshold_output, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));

		//定义参数
		vector<vector<Point>>contours_poly(contours.size());
		vector<Rect>boundRect(contours.size());

		//记录轮廓
		for (size_t i = 0; i < contours.size(); i++)
		{
			approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);
			boundRect[i] = boundingRect(Mat(contours_poly[i]));
		}

		for (size_t i = 0; i < contours.size(); i++)
		{
			//描绘轮廓
			drawContours(grayimg, contours_poly, i, Scalar(0, 255, 0), 1, 8, vector<Vec4i>(), 0, Point());
			//选择描绘特定大值轮廓边框
			if (boundRect[i].area() > g_zuizhiMax)
				rectangle(grayimg, boundRect[i].tl(), boundRect[i].br(), Scalar(0, 255, 0), 2, 8, 0);
		}

		//imshow(LowWindows, LowSrcimg);
		imshow(NowWindows, grayimg);


		waitKey(10);

	}


	waitKey(0);

	return 0;
}

void on_zuizhi(int, void*) {

	g_zuizhiMax = g_zuizhi * g_zuizhi;

}

void on_click(int event, int x, int y, int flags, void* param) {
	Mat& image = *(cv::Mat*)param;

	switch (event)
	{
		case EVENT_LBUTTONUP://左键弹起
			
			cout << g_click << endl;

			if (g_click < 4) {//1~4时保存
				fangshe = false;
				srcTriangle[g_click] = Point2f(static_cast<float>(x), static_cast<float>(y));
				g_click++;
			}
			else { 
				g_click = 0;
				
			}

			if (g_click >= 4) {
				fangshe = true;
			}
			
			break;

		case EVENT_RBUTTONDOWN://右键按下
			g_click = 0;
			fangshe = false;
			break;
	}

}