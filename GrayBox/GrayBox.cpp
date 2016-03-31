#include "GrayBox.h"
#include <boost/algorithm/string.hpp>
#include <fstream>

/**
* 自分の関数を記述し、真値と観測データとの差を計算する。
*
* @param p		観測データが入った構造体オブジェクト
* @param m		観測データの数
* @param n		パラメータの数
* @param x		パラメータ配列
* @param fvec	真値と観測データとの差を格納する配列
* @param iflag	lmdifから返されるフラグ (0なら終了?)
* @return		0を返却する
*/
int fcn(void *p, int m, int n, const real *x, real *fvec, int iflag) {
	const real *y = ((fcndata_t*)p)->y;
	//assert(m == 15 && n == 3);

	if (iflag == 0) {
		/* insert print statements here when nprint is positive. */
		/* if the nprint parameter to lmdif is positive, the function is
		called every nprint iterations with iflag=0, so that the
		function may perform special operations, such as printing
		residuals. */
		return 0;
	}

	// build matrix A and B
	cv::Mat_<double> A(4, 4), B(4, 3);
	A(0, 0) = -1.0 / x[5] / x[0] - 1.0 / x[5] / x[1];						// -1/Ce/R1 - 1/Ce/R2
	A(0, 1) = 1.0 / x[5] / x[1];											// 1/Ce/R2
	A(1, 0) = 1.0 / x[6] / x[1];											// 1/Cp/R2
	A(1, 1) = -1.0 / x[1] / x[6] - 1.0 / x[2] / x[6] - 1.0 / x[4] / x[6];	// -1/R2Cp - 1/R3/Cp - 1/Rwin/Cp
	A(1, 2) = 1.0 / x[2] / x[6];											// 1/Cp/R3
	A(2, 1) = 1.0 / x[2] / x[7];											// 1/Ci/R3
	A(2, 2) = -1.0 / x[2] / x[7] - 1.0 / x[3] / x[7];						// -1/Ci/R3 - 1/Ci/R4
	A(2, 3) = 1.0 / x[3] / x[7];											// 1/Ci/R4
	A(3, 2) = 1.0 / x[3] / x[8];											// 1/Cc/R4
	A(3, 3) = -1.0 / x[3] / x[8];											// -1/Cc/R4

	B(0, 0) = (1.0 - W) / x[5];												// (1-W)/Ce
	B(0, 1) = ((fcndata_t*)p)->Ap * 0.4 / 2.0 / x[5];						// Ap*0.4/2/Ce
	B(0, 2) = 1.0 / x[0] / x[5];											// 1/Ce/R1
	B(1, 1) = ((fcndata_t*)p)->Ap * 0.6 / x[6];								// Ap*0.6/Cp
	B(1, 2) = 1.0 / x[4] / x[6];											// 1/Ce/R1
	B(2, 0) = ((fcndata_t*)p)->W / x[7];									// W/Ci
	B(2, 1) = ((fcndata_t*)p)->Ap * 0.4 / 2.0 / x[7];						// Ap*0.4/2/Ci
	B(3, 1) = Ac * 0.7 / x[8];												// Ac*0.7/Cc

	// Xの初期化
	cv::Mat_<double> X(4, 1);
	for (int i = 0; i < 4; ++i) {
		X(i, 0) = 20 + 273.15;
	}
	
	// 真値と観測データの差を計算する
	for (int t = 0; t < ((fcndata_t*)p)->U.rows; ++t) {
		// dx/dt = Ax + Bu
		cv::Mat_<double> dX = A * X + B * ((fcndata_t*)p)->U.col(t);
		X += dX * 1; // 1 hour

		// y = Cx + Du
		cv::Mat_<double> Y = ((fcndata_t*)p)->C * X + ((fcndata_t*)p)->D * ((fcndata_t*)p)->U.col(t);

		// 誤差を格納する
		for (int k = 0; k < 2; ++k) {
			fvec[t * 2 + k] = y[t * 2 + k] - Y(k, 0);
		}
	}

	return 0;
}

GrayBox::GrayBox() {
}

void GrayBox::loadTrainingData(const std::string& filename) {
	std::ifstream file(filename);
	std::vector<std::vector<float> > dataset;

	// read dataset from the file
	std::string line;
	while (std::getline(file, line)) {
		dataset.push_back(std::vector<float>());

		std::vector<std::string> strs;
		boost::split(strs, line, boost::is_any_of(","));

		for (auto it = strs.begin(); it != strs.end(); ++it) {
			dataset.back().push_back(std::stod(*it));
		}
	}

	// copy the dataset to X, U, and Y
	X = cv::Mat_<double>(4, dataset.size());
	U = cv::Mat_<double>(3, dataset.size());
	Y = cv::Mat_<double>(2, dataset.size());

	for (int r = 0; r < dataset.size(); ++r) {
		// read X
		/*
		for (int k = 0; k < 4; ++k) {
			X.at<double>(k, r) = dataset[r][k];
		}
		*/

		// read U
		U.at<double>(0, r) = dataset[r][29];			// Q_sol
		U.at<double>(1, r) = dataset[r][30];			// Q_IHG
		U.at<double>(2, r) = dataset[r][28] + 273.15;	// T_out [Celcius] -> [K]

		// read Y (ground)
		Y.at<double>(0, r) = dataset[r][32] + 273.15;	// T_p [Celcius] -> [K]
		Y.at<double>(1, r) = dataset[r][31] + 273.15;	// T_c [Celcius] -> [K]
	}

	// initialize A, B, C, D
	A = cv::Mat_<double>(4, 4, 0.0);
	B = cv::Mat_<double>(4, 3, 0.0);
	C = cv::Mat_<double>(2, 4) << (0, 1, 0, 0, 0, 0, 0, 1);
	D = cv::Mat_<double>(2, 3) << (0, 0, 0, 0, 0, 0);
}

void GrayBox::inverse(double& R1, double& R2, double& R3, double& R4, double& Rwin, double& Ce, double& Cp, double& Ci, double& Cc, double W, double P, double H, double Ap, double Ac) {
	// パラメータの数
	const int NUM_PARAMS = 9;

	// 観測データの数
	int m = Y.rows * Y.cols;

	// パラメータ（行列A, B）
	real x[NUM_PARAMS];

	// パラメータの初期推定値
	x[0] = R1;
	x[1] = R2;
	x[2] = R3;
	x[3] = R4;
	x[4] = Rwin;
	x[5] = Ce;
	x[6] = Cp;
	x[7] = Ci;
	x[8] = Cc;

	// 観測データ（m個）
	real* y = new real[m];
	int count = 0;
	for (int c = 0; c < Y.cols; ++c) {
		for (int r = 0; r < Y.rows; ++r) {
			y[count++] = Y(r, c);
		}
	}

	// 真値と観測データとの誤差が格納される配列
	real* fvec = new real[m];

	// 結果のヤコビ行列
	real* fjac = new real[m * NUM_PARAMS];

	// lmdif内部使用パラメータ
	int ipvt[NUM_PARAMS];

	real diag[NUM_PARAMS], qtf[NUM_PARAMS], wa1[NUM_PARAMS], wa2[NUM_PARAMS], wa3[NUM_PARAMS];
	real* wa4 = new real[m];

	// 観測データを格納する構造体オブジェクト
	fcndata_t data;
	data.m = m;
	data.y = y;
	data.X = X;
	data.U = U;
	data.C = C;
	data.D = D;
	data.W = W;
	data.Ap = Ap;
	data.Ac = Ac;

	// 観測データの数と同じ値にすることを推奨する
	int ldfjac = m;

	// 各種パラメータ（推奨値のまま）
	real ftol = sqrt(__cminpack_func__(dpmpar)(1));
	real xtol = sqrt(__cminpack_func__(dpmpar)(1));
	real gtol = 0.;

	// 何回繰り返すか？
	int maxfev = 1600;

	// 収束チェック用の微小値
	real epsfcn = 1e-010;
	int mode = 1;

	// 1が推奨されている？
	real factor = 1;//1.e2;

	// 実際に繰り返した回数
	int nfev;

	int nprint = 0;
	int info = __cminpack_func__(lmdif)(fcn, &data, m, NUM_PARAMS, x, fvec, ftol, xtol, gtol, maxfev, epsfcn,
		diag, mode, factor, nprint, &nfev, fjac, ldfjac, ipvt, qtf, wa1, wa2, wa3, wa4);
	real fnorm = __cminpack_func__(enorm)(m, fvec);

	printf(" final l2 norm of the residuals%15.7g\n\n", (double)fnorm);
	printf(" number of function evaluations%10i\n\n", nfev);
	printf(" exit parameter %10i\n\n", info);

	// メモリ解放
	delete[] y;
	delete[] fvec;
	delete[] fjac;
	delete[] wa4;
}

void GrayBox::forward(cv::Mat_<double>& predictedY) {
	predictedY = cv::Mat_<double>(2, U.cols);

	// Xの初期化
	cv::Mat_<double> x(4, 1);
	for (int i = 0; i < 4; ++i) {
		x(i, 0) = X(i, 0);
	}

	// Yを予測する
	for (int t = 0; t < U.cols; ++t) {
		// dx/dt = Ax + Bu
		cv::Mat_<double> dx = A * x + B * U.col(t);
		x += dx * 60; // 60 min

		// y = Cx + Du
		predictedY.col(t) = C * x + D * U.col(t);
	}
}