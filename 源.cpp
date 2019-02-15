#include <opencv2/opencv.hpp>
#include <iostream>
#include <opencv2/tracking.hpp>
#include "Adbshell.h"
#include <librealsense2/rs.hpp>
#include "cv-helpers.hpp"

constexpr auto LowWindows = "ԭ��Ƶ";
constexpr auto fangsheWindows = "����任";
constexpr auto NowWindows = "�������Ƶ";

constexpr auto PhoneW = 1920;//�ֻ��ֱ���
constexpr auto PhoneH = 1080;//�ֻ��ֱ���

using namespace rs2;
using namespace cv;
using namespace std;

const size_t inWidth = 700;
const size_t inHeight = 480;
const float WHRatio = inWidth / (float)inHeight;

int g_Canny = 100;//��Ե�������
int g_zuizhi = 10;//�������������ֵ
int g_zuizhiMax = 100;//�����淽�����
bool fangshe = false;//�Ƿ�������任
bool Drawgenzong = false;//�Ƿ������ROI����
bool genzong = false;//�Ƿ�ʼ����
Rect2d g_ROI;//����ģ��
Rect g_rectangle;//��¼���λ��
int g_click = 0;//�������

Mat dstImg;//�����ͼ��

vector<vector<Point>>contours;
vector<Vec4i> hierarchy;
Point2f srcTriangle[4];
Point2f dstTriangle[4];

//��������ģ��
Ptr<TrackerTLD> tracker = TrackerTLD::create();

void on_zuizhi(int, void*);
void on_click_Low(int event, int x, int y, int flags, void* param);
void on_click_fangshe(int event, int x, int y, int flags, void* param);

int main() {

	Mat srcimg = imread("1.jpg");

	//VideoCapture capture(0);
	//capture >> srcimg;

	if (!srcimg.data) { cout << "ͼƬ����" << endl; cv::waitKey(0); return -1; }

	// ��Ӣ�ض�ʵ������ͷ��ȡ��Ƶ��
	pipeline pipe;
	auto config = pipe.start();
	auto profile = config.get_stream(RS2_STREAM_COLOR)
		.as<video_stream_profile>();

	rs2::align align_to(RS2_STREAM_COLOR);

	Size cropSize;
	if (profile.width() / (float)profile.height() > WHRatio) {
		cropSize = Size(static_cast<int>(profile.height() * WHRatio),
						profile.height());
	} else {
		cropSize = Size(profile.width(),
						static_cast<int>(profile.width() / WHRatio));
	}

	Rect crop(Point((profile.width() - cropSize.width) / 2,
		(profile.height() - cropSize.height) / 2),
			  cropSize);


	//������ͼ����
	namedWindow(LowWindows, WINDOW_AUTOSIZE);
	namedWindow(fangsheWindows, WINDOW_AUTOSIZE);
	namedWindow(NowWindows, WINDOW_AUTOSIZE);

	//��ʼ�����λ�ò���
	g_rectangle = Rect(-1, -1, 0, 0);
	g_ROI = Rect2d(-1, -1, 0, 0);

	//�������ص��¼�
	setMouseCallback(LowWindows, on_click_Low);
	setMouseCallback(fangsheWindows, on_click_fangshe);

	//�����������ص��¼�
	createTrackbar("��ֵ", NowWindows, &g_Canny, 254, NULL);
	createTrackbar("��ֵ", NowWindows, &g_zuizhi, 100, on_zuizhi);

	//����任������ʼ��
	dstTriangle[0] = Point2f(0, 0);
	dstTriangle[1] = Point2f(static_cast<float>(srcimg.cols - 1), 0);
	dstTriangle[2] = Point2f(0, static_cast<float>(srcimg.rows - 1));
	dstTriangle[3] = Point2f(static_cast<float>(srcimg.cols - 1), static_cast<float>(srcimg.rows - 1));

	
	CAdbshell shell;
	if (shell.Start()) {
		cout << "�����ɹ�" << endl;
		if (shell.RunCmd("dumpsys battery")) {
			cout << "���ͳɹ�" << endl;
		}
		shell.Stop();
		cout << "adb:" << shell.GetOutput() << endl;
	}
	

	while (true) {

		Mat grayimg;
		//capture >> srcimg;

		Mat threshold_output;
		Mat warpMat;

		// �ȴ���һ֡����
		auto data = pipe.wait_for_frames();
		// ȷ������ڿռ��϶���
		data = align_to.process(data);

		auto color_frame = data.get_color_frame();

		// ���û���յ����ݣ�����
		static int last_frame_number = 0;
		if (color_frame.get_frame_number() == last_frame_number) continue;
		last_frame_number = color_frame.get_frame_number();

		// �� RealSense ת��Ϊ OpenCV ����:
		auto color_mat = frame_to_mat(color_frame);

		// Crop both color and depth frames
		color_mat = color_mat(crop);

		//imshow("������", color_mat);

		//��ʱ����ת90��
		//transpose(srcimg, srcimg);
		//flip(srcimg, srcimg, 0);

		//������ĵ�
		if (fangshe || g_click) {//Ϊ0ʱ���������

			for (size_t i = 0; i < g_click; i++) {
				circle(color_mat, srcTriangle[i], 3, Scalar(0, 0, 255), -1);
			}
		}

		imshow(LowWindows, color_mat);


		if (fangshe){//�����ʼ����
			warpMat = getAffineTransform(srcTriangle, dstTriangle);
			warpAffine(color_mat, dstImg, warpMat, color_mat.size());
		} else {
			dstImg = color_mat;
		}

		//���ROI����
		//if (!Drawgenzong)
		rectangle(dstImg, g_ROI, Scalar(255, 0, 0), 2, 1);

		if (genzong)
			tracker->update(dstImg, g_ROI);//����Ŀ�����

		imshow(fangsheWindows, dstImg);



		//ת���ɻҶ�ͼ
		cvtColor(dstImg, grayimg, COLOR_BGR2GRAY);
		//����ģ��
		blur(grayimg, grayimg, Size(3, 3));
		//��ֵ��
		threshold(grayimg, threshold_output, g_Canny, 255, THRESH_BINARY);
		//Ѱ������
		findContours(threshold_output, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));

		//�������
		vector<vector<Point>>contours_poly(contours.size());
		vector<Rect>boundRect(contours.size());

		//��¼����
		for (size_t i = 0; i < contours.size(); i++) {
			approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);
			boundRect[i] = boundingRect(Mat(contours_poly[i]));
		}

		for (size_t i = 0; i < contours.size(); i++) {
			//�������
			drawContours(grayimg, contours_poly, i, Scalar(0, 255, 0), 1, 8, vector<Vec4i>(), 0, Point());
			//ѡ������ض���ֵ�����߿�
			if (boundRect[i].area() > g_zuizhiMax)
				rectangle(grayimg, boundRect[i].tl(), boundRect[i].br(), Scalar(0, 255, 0), 2, 8, 0);
		}

		imshow(NowWindows, grayimg);


		waitKey(1);

	}


	waitKey(0);

	return 0;
}

void on_zuizhi(int, void*) {

	g_zuizhiMax = g_zuizhi * g_zuizhi;

}

void on_click_Low(int event, int x, int y, int flags, void* param) {//ԭ��Ƶ���ڵĵ���¼�����4����ȷ�������

	switch (event) {
		case EVENT_LBUTTONUP://�������

			cout << g_click << endl;

			if (g_click < 4) {//1~4ʱ����
				fangshe = false;
				srcTriangle[g_click] = Point2f(static_cast<float>(x), static_cast<float>(y));
				g_click++;
			} else {
				g_click = 0;

			}

			if (g_click >= 4) {
				fangshe = true;
			}

			break;

		case EVENT_RBUTTONDOWN://�Ҽ�����
			g_click = 0;
			fangshe = false;
			break;
	}
}

void on_click_fangshe(int event, int x, int y, int flags, void* param) {//ѡ��ROI����

	switch (event) {
		case EVENT_MOUSEMOVE://����ƶ�

			if (Drawgenzong) {//�Ƿ���м�¼
				g_ROI.width = x - g_ROI.x;
				g_ROI.height = y - g_ROI.y;

				if (g_ROI.width < 0) {
					g_ROI.x += g_ROI.width;
					g_ROI.width *= -1;
				}

				if (g_ROI.height < 0) {
					g_ROI.y += g_ROI.height;
					g_ROI.height *= -1;
				}


			}

			break;

		case EVENT_LBUTTONDOWN://�������

			Drawgenzong = true;
			g_ROI = Rect2d(x, y, 0, 0);//��¼��ʼ��
			cout << x << "," << y << endl;

			break;

		case EVENT_LBUTTONUP://�������

			genzong = true;
			Drawgenzong = false;
			tracker->init(dstImg, g_ROI);

			break;

		case EVENT_RBUTTONDOWN://�Ҽ�����

			g_ROI = Rect2d(-1, -1, 0, 0);
			genzong = false;

			break;
	}
}