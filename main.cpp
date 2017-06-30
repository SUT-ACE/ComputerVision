#include <fstream>  
#include "opencv2/opencv.hpp"  
#include <vector>  

using namespace std;
using namespace cv;

#define SHOW_PROCESS  1 
#define ON_STUDY      0  

#define TRAIN   1  
#define TEST    0  

#define SAMPLE_NUM        9    //0-9数字样本       最大为9 
#define SAMPLE_MAX_NUM    5000 //训练数字样本最大编号  5000
#define TEST_MAX_NUM    800  //评估测试数字样本最大编号 800

/*
************************************************************************
*
*类：class NumTrainData
*功能：创建训练样本矩阵，包括标签和样本数据
*参数：float data[64];int result;
*
*
************************************************************************
*/
class NumTrainData
{
public:
	NumTrainData()
	{
		memset(data, 0, sizeof(data));
		result = -1;
	}
public:
	float data[64];
	int result;
};

vector<NumTrainData> buffer;//整理后得到的训练数据样本容器
int featureLen = 64;

/*
********************************************************
*
*
*函数功能：得到文件位置字符串，读取文件夹下指定数目的图片
*
********************************************************
*/
void ReadImage(Mat &image ,unsigned char pathNum, int num, bool flag)
{
	char filename[100];
	if (flag)//flag为1 则为训练样本路径
	{
		sprintf(filename, "E:/WIN10 Document/桌面/MNIST图片库/trainimage/pic2/%d/%d.bmp", pathNum, num);//用这个函数来转换图片名称，存放在filename中。
	}
	else   //反之则为测试样本路径
	{
		sprintf(filename, "E:/WIN10 Document/桌面/MNIST图片库/testimage/pic2/%d/%d.bmp", pathNum, num);//用这个函数来转换图片名称，存放在filename中。
	}
	
	cout << filename << endl;
	image = imread(filename, 0);
}

/*
******************************************************************************
*函数名：void GetROI(Mat& src, Mat& dst)
*功能：从输入图像中分割出数字有效区域，并将有效区放置在创建正方形中心处
*入口参数:Mat src    数字库中的图像，随意大小
*返回参数:Mat dest   由输入决定大小的正方形图像 
******************************************************************************
*/
void GetROI(Mat& src, Mat& dst)
{
	int left, right, top, bottom;
	left = src.cols;
	right = 0;
	top = src.rows;
	bottom = 0;

	//Get valid area 左右上下扫描，得到有效区 
	for (int i = 0; i<src.rows; i++)
	{
		for (int j = 0; j<src.cols; j++)
		{
			if (src.at<uchar>(i, j) > 0)
			{
				if (j<left) left = j;
				if (j>right) right = j;
				if (i<top) top = i;
				if (i>bottom) bottom = i;
			}
		}
	}

	//Point center;  
	//center.x = (left + right) / 2;  
	//center.y = (top + bottom) / 2;  

	int width = right - left;
	int height = bottom - top;
	int len = (width < height) ? height : width;//得到长边

	//Create a squre  
	dst = Mat::zeros(len, len, CV_8UC1);//创建正方形输出图像原型

	//Copy valid data to squre center  
	Rect dstRect((len - width) / 2, (len - height) / 2, width, height);//将有效区放入正方形输出图像中心处
	Rect srcRect(left, top, width, height);
	Mat dstROI = dst(dstRect);
	Mat srcROI = src(srcRect);
	srcROI.copyTo(dstROI);
}

int ReadTrainData(void)
{
	Mat src;//载入原图
	Mat temp = Mat::zeros(8, 8, CV_8UC1);//降维度后的统一尺度8*8
	Mat img, dst;//dst为方形有效区域

	//Create source and show image matrix  //矩阵
	Scalar templateColor(255, 0, 255);
	NumTrainData rtd;//8*8=64个字节全至0，result为-1

	for (int m = 0; m <= SAMPLE_NUM; m++)//外层循环，0-9个数字
	{
		for (int n = 1; n <= SAMPLE_MAX_NUM; n++)//内层循环，1-5000个数字样本
		{
				//Read source data  
				ReadImage(src, m, n,TRAIN);
				GetROI(src, dst);//分割并调整图像输出

				#if(SHOW_PROCESS) //如果要显示过程，则将调整后的图像横纵都放大十倍来显示 
					//Too small to watch  
					img = Mat::zeros(dst.rows * 10, dst.cols * 10, CV_8UC1);
					resize(dst, img, img.size());

					stringstream ss;
					ss << "Number " << m<<","<<n;
					string text = ss.str();
					putText(img, text, Point(10, 50), FONT_HERSHEY_SIMPLEX, 1.0, templateColor);

					imshow("img", img);  
				#endif  

				rtd.result = m;//标签值
				resize(dst, temp, temp.size());//将调整后的图像统一再次调整为8*8的图像
				//threshold(temp, temp, 10, 1, CV_THRESH_BINARY);  

				//将8*8的图像调整为1*64的数组存储
				for (int i = 0; i<8; i++)
				{
					for (int j = 0; j<8; j++)
					{
						rtd.data[i * 8 + j] = temp.at<uchar>(i, j);
					}
				}

				buffer.push_back(rtd);//将调整的数组放到容器里

				//if(waitKey(0)==27) //ESC to quit  
				//  break;  
	     }
   }
	
	cout <<"读取训练数据完成" << endl;
	waitKey(0);
	return 0;
}

void newSvmStudy(vector<NumTrainData>& trainData)
{
	int testCount = trainData.size();

	Mat m = Mat::zeros(1, featureLen, CV_32FC1);
	Mat data = Mat::zeros(testCount, featureLen, CV_32FC1);
	Mat res = Mat::zeros(testCount, 1, CV_32SC1);

	for (int i = 0; i< testCount; i++)
	{

		NumTrainData td = trainData.at(i);
		memcpy(m.data, td.data, featureLen*sizeof(float));
		normalize(m, m);
		memcpy(data.data + i*featureLen*sizeof(float), m.data, featureLen*sizeof(float));

		res.at<unsigned int>(i, 0) = td.result;

		cout <<i<< endl;
	}

	/////////////START SVM TRAINNING//////////////////  
	CvSVM svm ;//= CvSVM()
	CvSVMParams param;
	CvTermCriteria criteria;//迭代终止条件

	criteria = cvTermCriteria(CV_TERMCRIT_EPS, 1000, FLT_EPSILON);
	param = CvSVMParams(CvSVM::C_SVC, CvSVM::RBF, 10.0, 8.0, 1.0, 10.0, 0.5, 0.1, NULL, criteria);

	svm.train(data, res, Mat(), Mat(), param);
	svm.save("SVM_DATA.xml");

	cout << "训练结束"<< endl;
	waitKey(0);
}


int newSvmPredict()
{
	CvSVM svm ;//= CvSVM()
	svm.load("SVM_DATA.xml");

	Mat src;
	Mat temp = Mat::zeros(8, 8, CV_8UC1);
	Mat LEN = Mat::zeros(1, featureLen, CV_32FC1);
	Mat img, dst;

	int    right = 0;
	float  rightPercent[10] = {0};//正确率

	Scalar templateColor(255, 0, 0);
	NumTrainData rtd;

	for (int m = 0; m <= SAMPLE_NUM; m++)//外层循环，0-9个数字
	{
		for (int n = 1; n <= TEST_MAX_NUM; n++)//内层循环，1-800个数字样本
		{
				ReadImage(src, m, n,TEST);
				GetROI(src, dst);

				//Too small to watch  
				img = Mat::zeros(dst.rows * 30, dst.cols * 30, CV_8UC3);
				resize(dst, img, img.size());

				rtd.result = m;
				resize(dst, temp, temp.size());

				//threshold(temp, temp, 10, 1, CV_THRESH_BINARY);  
				for (int i = 0; i<8; i++)
				{
					for (int j = 0; j<8; j++)
					{
						LEN.at<float>(0, j + i * 8) = temp.at<uchar>(i, j);
					}
				}

				normalize(LEN, LEN);
				char ret = (char)svm.predict(LEN);

				if (ret == m)
				{
					right++;
				}

				#if(SHOW_PROCESS)  
						stringstream ss;
						ss << "Number " << m << "," << n;
						string text = ss.str();
						putText(img, text, Point(10, 50), FONT_HERSHEY_SIMPLEX, 1.0, templateColor);

						imshow("img", img);
						if (waitKey(0) == 27) //ESC to quit  
							break;
				#endif  

		}

		rightPercent[m] = right/800.0;//正确率
		cout << "数字" << m << "正确率：" << rightPercent[m] << endl;
		right = 0;//计数清零
	}

//	stringstream ss;
//	ss << ", right " << right/800.0 <<endl;
//	string text = ss.str();
//	putText(img, text, Point(50, 50), FONT_HERSHEY_SIMPLEX, 1.0, templateColor);
//	imshow("img", img);
	cout <<"测试结束" << endl;
	waitKey(0);
	return 0;
}

int main(int argc, char *argv[]) 
{
	#if(ON_STUDY)  
		ReadTrainData();
		newSvmStudy(buffer);
	#else  
		newSvmPredict();
	#endif  
		return 0;
}
