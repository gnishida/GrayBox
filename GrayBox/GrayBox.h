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
	double H;				// height
	double P;				// perimeter
} fcndata_t;

int fcn(void *p, int m, int n, const real *x, real *fvec, int iflag);

class Range {
public:
	double minimum;
	double maximum;
	double step;

public:
	Range();
	Range(double minimum, double maximum);
};

class GrayBox {
public:
	cv::Mat_<double> X;
	cv::Mat_<double> U;
	cv::Mat_<double> Y;
	cv::Mat_<double> A;
	cv::Mat_<double> B;
	cv::Mat_<double> C;
	cv::Mat_<double> D;
	std::map<std::string, Range> ranges;
	double W;				// window ratio
	double Ap;				// area of perimeter zone
	double Ac;				// area of core zone
	double H;				// height
	double P;				// perimeter

public:
	GrayBox();

	void loadTrainingData(const std::string& simulation_filename, const std::string& range_filename);
	void inverse(double& R1, double& R2, double& R3, double& R4, double& Rwin, double& Ce, double& Cp, double& Ci, double& Cc);
	void forward(cv::Mat_<double>& predictedY);
};



