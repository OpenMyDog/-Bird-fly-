#include <opencv2/opencv.hpp>
#include <iostream>
#include <opencv2/tracking.hpp>

constexpr auto LowWindows = "ԭ��Ƶ";
constexpr auto fangsheWindows = "����任";
constexpr auto NowWindows = "�������Ƶ";

constexpr auto PhoneW = 1920;//�ֻ��ֱ���
constexpr auto PhoneH = 1080;//�ֻ��ֱ���

using namespace cv;
using namespace std;

int g_Canny = 100;//��Ե�������
int g_zuizhi = 10;//�������������ֵ
int g_zuizhiMax = 100;//�����淽�����
bool fangshe = false;//�Ƿ�������任
bool genzong =false;//�Ƿ������ROI����
Rect2d g_ROI;//����ģ��
Rect g_rectangle;//��¼���λ��
int g_click = 0;//�������

Mat dstImg;//�����ͼ��

vector<vector<Point>>contours;
vector<Vec4i> hierarchy;
Point2f srcTriangle[4];
Point2f dstTriangle[4];

//��������ģ��
Ptr<TrackerMOSSE> tracker = TrackerMOSSE::create();

void on_zuizhi(int, void*);
void on_click_Low(int event, int x, int y, int flags, void* param);
void on_click_fangshe(int event, int x, int y, int flags, void* param);

int main() {

	Mat srcimg = imread("1.jpg");

	VideoCapture capture(0);
	capture >> srcimg;

	
	

	if (!srcimg.data) { cout << "ͼƬ����" << endl; cv::waitKey(0); return -1; }

	//������ͼ����
	namedWindow(LowWindows, WINDOW_NORMAL);
	namedWindow(fangsheWindows, WINDOW_NORMAL);
	namedWindow(NowWindows, WINDOW_NORMAL);

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
	dstTriangle[1] = Point2f(static_cast<float>(srcimg.cols - 1),0);
	dstTriangle[2] = Point2f(0, static_cast<float>(srcimg.rows - 1));
	dstTriangle[3] = Point2f(static_cast<float>(srcimg.cols - 1), static_cast<float>(srcimg.rows - 1));

	while (true)
	{

		Mat grayimg;
		capture >> srcimg;

		Mat threshold_output;
		Mat warpMat;
		

		//��ʱ����ת90��
		//transpose(srcimg, srcimg);
		//flip(srcimg, srcimg, 0);

				//������ĵ�
		if (fangshe||g_click) {//Ϊ0ʱ���������

			for (size_t i = 0; i < g_click; i++)
			{
				circle(srcimg, srcTriangle[i], 3, Scalar(0, 0, 255), -1);
			}
		}
		
		imshow(LowWindows, srcimg);


		if (fangshe)//�����ʼ����
		{
			warpMat = getAffineTransform(srcTriangle, dstTriangle);
			warpAffine(srcimg, dstImg, warpMat, dstImg.size());
		}else {
			dstImg = srcimg;
		}

		if(!genzong)
			tracker->update(dstImg, g_ROI);//����Ŀ�����

		//���ROI����
		if (g_ROI.width > 0 && g_ROI.height > 0)
			rectangle(dstImg, g_ROI, Scalar(255, 0, 0), 2, 1);

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
		for (size_t i = 0; i < contours.size(); i++)
		{
			approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);
			boundRect[i] = boundingRect(Mat(contours_poly[i]));
		}

		for (size_t i = 0; i < contours.size(); i++)
		{
			//�������
			drawContours(grayimg, contours_poly, i, Scalar(0, 255, 0), 1, 8, vector<Vec4i>(), 0, Point());
			//ѡ������ض���ֵ�����߿�
			if (boundRect[i].area() > g_zuizhiMax)
				rectangle(grayimg, boundRect[i].tl(), boundRect[i].br(), Scalar(0, 255, 0), 2, 8, 0);
		}

		imshow(NowWindows, grayimg);


		waitKey(10);

	}


	waitKey(0);

	return 0;
}

void on_zuizhi(int, void*) {

	g_zuizhiMax = g_zuizhi * g_zuizhi;

}

void on_click_Low(int event, int x, int y, int flags, void* param) {//ԭ��Ƶ���ڵĵ���¼�����4����ȷ�������
	

	switch (event)
	{
		case EVENT_LBUTTONUP://�������
			
			cout << g_click << endl;

			if (g_click < 4) {//1~4ʱ����
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

		case EVENT_RBUTTONDOWN://�Ҽ�����
			g_click = 0;
			fangshe = false;
			break;
	}

}

void on_click_fangshe(int event, int x, int y, int flags, void* param) {//ѡ��ROI����

	switch (event)
	{
		case EVENT_MOUSEMOVE://����ƶ�

			if (genzong) {//�Ƿ���м�¼
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
			
		case EVENT_LBUTTONDOWN://��갴��

			genzong = true;
			g_ROI = Rect(x, y, 0, 0);//��¼��ʼ��

			break;

		case EVENT_LBUTTONUP://��굯��

			genzong = false;
			tracker->init(dstImg, g_ROI);

			break;
	}


}