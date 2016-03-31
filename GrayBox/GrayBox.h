#pragma once

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <cminpack.h>
#define real __cminpack_real__

// 観測データを定義する構造体
typedef struct {
	int m;
	real *y;
	cv::Mat_<double> X;
	cv::Mat_<double> U;
	cv::Mat_<double> C;
	cv::Mat_<double> D;
	double W;				// window ratio
	double Ap;				// area of perimeter zone
	double Ac;				// area of core zone
} fcndata_t;

int fcn(void *p, int m, int n, const real *x, real *fvec, int iflag);

class GrayBox {
private:
	cv::Mat_<double> X;
	cv::Mat_<double> U;
	cv::Mat_<double> Y;
	cv::Mat_<double> A;
	cv::Mat_<double> B;
	cv::Mat_<double> C;
	cv::Mat_<double> D;

public:
	GrayBox();

	void loadTrainingData(const std::string& filename);
	void inverse(double& R1, double& R2, double& R3, double& R4, double& Rwin, double& Ce, double& Cp, double& Ci, double& Cc, double W, double P, double H, double Ap, double Ac);
	void forward(cv::Mat_<double>& predictedY);
};

