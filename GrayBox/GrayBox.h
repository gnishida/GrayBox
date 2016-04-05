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
	//cv::Mat_<double> X;
	cv::Mat_<double> U;
	cv::Mat_<double> C;
	cv::Mat_<double> D;
	double W;				// window ratio
	double Ap;				// area of perimeter zone
	double Ac;				// area of core zone
	double H;				// height
	double P;				// perimeter
	double Rwin;			// Rwin;
} fcndata_t;

int fcn(void *p, int m, int n, const real *x, real *fvec, int iflag);
void buildMatrices(double R1, double R2, double R3, double R4, double Ce, double Cp, double Ci, double Cc, double W, double Ap, double Ac, double H, double P, double Rwin, cv::Mat_<double>& A, cv::Mat_<double>& B);

class Range {
public:
	double minimum;
	double maximum;
	double step;

public:
	Range();
	Range(double minimum, double maximum);
};

class GrayBoxResult {
public:
	double fnorm;
	double R1;
	double R2;
	double R3;
	double R4;
	double Ce;
	double Cp;
	double Ci;
	double Cc;

public:
	GrayBoxResult(double fnorm, double R1, double R2, double R3, double R4, double Ce, double Cp, double Ci, double Cc);
};

class GrayBox {
public:
	//cv::Mat_<double> X;
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
	double Rwin;			// Rwin

public:
	GrayBox();

	void loadTrainingData(const std::string& simulation_filename, const std::string& range_filename);
	GrayBoxResult inverse(double R1, double R2, double R3, double R4, double Ce, double Cp, double Ci, double Cc);
	double forward(double R1, double R2, double R3, double R4, double Ce, double Cp, double Ci, double Cc, cv::Mat_<double>& predictedY);
};



