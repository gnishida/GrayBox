#include "GrayBox.h"
#include <boost/algorithm/string.hpp>
#include <fstream>

#define SQR(x)					((x) * (x))
#define	Feet2Meter(x)			((x) * 0.3048)
#define C2K(x)					((x) + 273.15)

Range::Range() {
	this->minimum = 0;
	this->maximum = 0;
	this->step = 0;
}

Range::Range(double minimum, double maximum) {
	this->minimum = minimum;
	this->maximum = maximum;
	this->step = (maximum - minimum) / 3.0;
}

GrayBoxResult::GrayBoxResult(double fnorm, double R1, double R2, double R3, double R4, double Ce, double Cp, double Ci, double Cc) {
	this->fnorm = fnorm;
	this->R1 = R1;
	this->R2 = R2;
	this->R3 = R3;
	this->R4 = R4;
	this->Ce = Ce;
	this->Cp = Cp;
	this->Ci = Ci;
	this->Cc = Cc;
}

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

	if (iflag == 0) {
		/* insert print statements here when nprint is positive. */
		/* if the nprint parameter to lmdif is positive, the function is
		called every nprint iterations with iflag=0, so that the
		function may perform special operations, such as printing
		residuals. */
		return 0;
	}

	// build matrix A and B
	cv::Mat_<double> A(4, 4, 0.0);
	cv::Mat_<double> B(4, 3, 0.0);
	buildMatrices(x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], ((fcndata_t*)p)->W, ((fcndata_t*)p)->Ap, ((fcndata_t*)p)->Ac, ((fcndata_t*)p)->H, ((fcndata_t*)p)->P, ((fcndata_t*)p)->Rwin, A, B);
	/*
	A(0, 0) = -1.0 / x[4] / x[0] - 1.0 / x[4] / x[1];										// -1/Ce/R1 - 1/Ce/R2
	A(0, 1) = 1.0 / x[4] / x[1];															// 1/Ce/R2
	A(1, 0) = 1.0 / x[5] / x[1];															// 1/Cp/R2
	A(1, 1) = -1.0 / x[1] / x[5] - 1.0 / x[2] / x[5] - 1.0 / ((fcndata_t*)p)->Rwin / x[5];	// -1/R2Cp - 1/R3/Cp - 1/Rwin/Cp
	A(1, 2) = 1.0 / x[2] / x[5];															// 1/Cp/R3
	A(2, 1) = 1.0 / x[2] / x[6];															// 1/Ci/R3
	A(2, 2) = -1.0 / x[2] / x[6] - 1.0 / x[3] / x[6];										// -1/Ci/R3 - 1/Ci/R4
	A(2, 3) = 1.0 / x[3] / x[6];															// 1/Ci/R4
	A(3, 2) = 1.0 / x[3] / x[7];															// 1/Cc/R4
	A(3, 3) = -1.0 / x[3] / x[7];															// -1/Cc/R4

	B(0, 0) = ((fcndata_t*)p)->P * ((fcndata_t*)p)->H * (1.0 - ((fcndata_t*)p)->W) / x[4];	// PH(1-W)/Ce
	B(0, 1) = ((fcndata_t*)p)->Ap * 0.4 / 2.0 / x[4];										// Ap*0.4/2/Ce
	B(0, 2) = 1.0 / x[0] / x[4];															// 1/Ce/R1
	B(1, 1) = ((fcndata_t*)p)->Ap * 0.6 / x[5];												// Ap*0.6/Cp
	B(1, 2) = 1.0 / x[5] / ((fcndata_t*)p)->Rwin;											// 1/Cp/Rwin
	B(2, 0) = ((fcndata_t*)p)->P * ((fcndata_t*)p)->H * ((fcndata_t*)p)->W / x[6];			// PHW/Ci
	B(2, 1) = ((fcndata_t*)p)->Ap * 0.4 / 2.0 / x[6] + ((fcndata_t*)p)->Ac * 0.3 / x[6];	// Ap*0.4/2/Ci + Ac*0.3/Ci
	B(3, 1) = ((fcndata_t*)p)->Ac * 0.7 / x[7];												// Ac*0.7/Cc
	*/

	// Xの初期化
	cv::Mat_<double> X(4, 1);
	for (int i = 0; i < 4; ++i) {
		X(i, 0) = C2K(20);
	}
	
	// 真値と観測データの差を計算する
	for (int t = 0; t < ((fcndata_t*)p)->U.cols; ++t) {
		//////////////// DEBUG //////////////////////////////////////////////////
		/*
		std::cout << "A: " << A << std::endl;
		std::cout << "B: " << B << std::endl;
		std::cout << "U: " << ((fcndata_t*)p)->U.col(t) << std::endl;
		*/

		// dx/dt = Ax + Bu
		cv::Mat_<double> dX = A * X + B * ((fcndata_t*)p)->U.col(t);
		X += dX * 3600; // 1 hour = 3600 sec

		//////////////// DEBUG //////////////////////////////////////////////////
		/*
		std::cout << dX << std::endl;
		std::cout << X << std::endl;
		*/

		// y = Cx + Du
		cv::Mat_<double> Y = ((fcndata_t*)p)->C * X + ((fcndata_t*)p)->D * ((fcndata_t*)p)->U.col(t);

		//std::cout << "y0: " << Y(0, 0) << " (True: " << y[t * 2] << "), y1: " << Y(1, 0) << " (True: " << y[t * 2 + 1] << ")" << std::endl;

		// 誤差を格納する
		for (int k = 0; k < 2; ++k) {
			fvec[t * 2 + k] = SQR(y[t * 2 + k] - Y(k, 0));
		}
	}

	return 0;
}

/**
 * Build matrix A and B
 */
void buildMatrices(double R1, double R2, double R3, double R4, double Ce, double Cp, double Ci, double Cc, double W, double Ap, double Ac, double H, double P, double Rwin, cv::Mat_<double>& A, cv::Mat_<double>& B) {
	A = cv::Mat_<double>(4, 4, 0.0);
	B = cv::Mat_<double>(4, 3, 0.0);

	A(0, 0) = -1.0 / Ce / R1 - 1.0 / Ce / R2;					// -1/Ce/R1 - 1/Ce/R2
	A(0, 1) = 1.0 / Ce / R2;									// 1/Ce/R2
	A(1, 0) = 1.0 / Cp / R2;									// 1/Cp/R2
	A(1, 1) = -1.0 / R2 / Cp - 1.0 / R3 / Cp - 1.0 / Rwin / Cp;	// -1/R2Cp - 1/R3/Cp - 1/Rwin/Cp
	A(1, 2) = 1.0 / R3 / Cp;									// 1/Cp/R3
	A(2, 1) = 1.0 / R3 / Ci;									// 1/Ci/R3
	A(2, 2) = -1.0 / R3 / Ci - 1.0 / R4 / Ci;					// -1/Ci/R3 - 1/Ci/R4
	A(2, 3) = 1.0 / R4 / Ci;									// 1/Ci/R4
	A(3, 2) = 1.0 / R4 / Cc;									// 1/Cc/R4
	A(3, 3) = -1.0 / R4 / Cc;									// -1/Cc/R4

	B(0, 0) = P * H * (1.0 - W) / Ce;						// PH(1-W)/Ce
	B(0, 1) = Ap * 0.4 / 2.0 / Ce;								// Ap*0.4/2/Ce
	B(0, 2) = 1.0 / R1 / Ce;									// 1/Ce/R1
	B(1, 1) = Ap * 0.6 / Cp;									// Ap*0.6/Cp
	B(1, 2) = 1.0 / Cp / Rwin;									// 1/Cp/Rwin
	B(2, 0) = P * H * W / Ci;								// PHW/Ci
	B(2, 1) = Ap * 0.4 / 2.0 / Ci + Ac * 0.3 / Ci;				// Ap*0.4/2/Ci + Ac*0.3/Ci
	B(3, 1) = Ac * 0.7 / Cc;									// Ac*0.7/Cc
}

GrayBox::GrayBox() {
}

void GrayBox::loadTrainingData(const std::string& simulation_filename, const std::string& range_filename) {
	// read simulation results
	{
		std::ifstream sim_file(simulation_filename);
		std::vector<std::vector<float> > dataset;

		// skip 2 lines
		std::string line;
		std::getline(sim_file, line);
		std::getline(sim_file, line);

		// read dataset from the file
		int line_num = 0;
		while (std::getline(sim_file, line)) {
			dataset.push_back(std::vector<float>());

			std::vector<std::string> strs;
			boost::split(strs, line, boost::is_any_of(","));

			int col_num = 0;
			for (auto it = strs.begin(); it != strs.end(); ++it, ++col_num) {
				try {
					dataset.back().push_back(std::stod(*it));
				}
				catch (std::invalid_argument ex) {
					std::cerr << "Invalid argument: " << *it << " at line " << line_num + 3 << " column " << col_num + 1 << std::endl;
				}
			}
			line_num++;

			// read only the first 30 days
			if (line_num >= 24 * 30) break;
		}
		sim_file.close();

		// copy the dataset to X, U, and Y
		//X = cv::Mat_<double>(4, dataset.size());
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
			U.at<double>(0, r) = dataset[r][64];		// Q_sol
			//U.at<double>(0, r) = dataset[r][6] * 0.293071 / 0.3048 / 0.3048;	// use horizontal component only for Q_sol
			U.at<double>(1, r) = dataset[r][65];		// Q_IHG
			U.at<double>(2, r) = C2K(dataset[r][63]);	// T_out [Celcius] -> [K]

			// read Y (ground)
			Y.at<double>(0, r) = C2K(dataset[r][67]);	// T_p [Celcius] -> [K]
			Y.at<double>(1, r) = C2K(dataset[r][66]);	// T_c [Celcius] -> [K]
		}

		// initialize A, B, C, D
		A = cv::Mat_<double>(4, 4, 0.0);
		B = cv::Mat_<double>(4, 3, 0.0);
		C = (cv::Mat_<double>(2, 4) << 0, 1, 0, 0, 0, 0, 0, 1);
		D = (cv::Mat_<double>(2, 3) << 0, 0, 0, 0, 0, 0);
	}

	////////////////////////////////////////////////////////////////
	// read range from the file
	{
		std::ifstream range_file(range_filename);
		std::string line;

		// skip the first 3 lines
		std::getline(range_file, line);
		std::getline(range_file, line);
		std::getline(range_file, line);

		// read range values
		std::getline(range_file, line);

		std::vector<std::string> strs;
		std::vector<double> values;
		boost::split(strs, line, boost::is_any_of(","));

		for (auto it = strs.begin(); it != strs.end(); ++it) {
			values.push_back(std::stod(*it));
		}
		range_file.close();

		W = values[12];
		Ap = Feet2Meter(values[0] - 15) * Feet2Meter(15) * 4;
		//Ap = Feet2Meter(Feet2Meter(values[2]));
		Ac = SQR(Feet2Meter(values[0] - 15 * 2));
		//Ac = Feet2Meter(Feet2Meter(values[3]));
		H = Feet2Meter(10.0);
		P = Feet2Meter(values[0]) * 4;

		ranges["Ce"] = Range(values[13] * 1055 / 5.0 * 9.0, values[14] * 1055 / 5.0 * 9.0);
		ranges["Cp"] = Range(values[15] * 1055 / 5.0 * 9.0, values[16] * 1055 / 5.0 * 9.0);
		ranges["Ci"] = Range(values[17] * 1055 / 5.0 * 9.0, values[18] * 1055 / 5.0 * 9.0);
		ranges["Cc"] = Range(values[19] * 1055 / 5.0 * 9.0, values[20] * 1055 / 5.0 * 9.0);
		ranges["R1"] = Range(values[21] * 5.0 / 9.0 / 0.293071, values[22] * 5.0 / 9.0 / 0.293071);
		ranges["R2"] = Range(values[23] * 5.0 / 9.0 / 0.293071, values[24] * 5.0 / 9.0 / 0.293071);
		ranges["R3"] = Range(values[25] * 5.0 / 9.0 / 0.293071, values[26] * 5.0 / 9.0 / 0.293071);
		ranges["R4"] = Range(values[27] * 5.0 / 9.0 / 0.293071, values[28] * 5.0 / 9.0 / 0.293071);
		//Rwin = values[29] * 5.0 / 9.0 / 0.293071;
		//Rwin = 0.003;
	}
}

GrayBoxResult GrayBox::inverse(double R1, double R2, double R3, double R4, double Ce, double Cp, double Ci, double Cc, double Rwin) {
	/*
	std::cout << "///////////////////////////////////////////////////////////" << std::endl;
	std::cout << "Inverse computation" << std::endl;
	std::cout << "initial values: R1=" << R1 << ", R2=" << R2 << ", R3=" << R3 << ", R4=" << R4 << ", Ce=" << Ce << ", Cp=" << Cp << ", Ci=" << Ci << ", Cc=" << Cc << std::endl;
	*/

	// パラメータの数
	const int NUM_PARAMS = 8;

	// 観測データの数
	int m = Y.rows * Y.cols;

	// パラメータ（行列A, B）
	real x[NUM_PARAMS];

	// パラメータの初期推定値
	x[0] = R1;
	x[1] = R2;
	x[2] = R3;
	x[3] = R4;
	x[4] = Ce;
	x[5] = Cp;
	x[6] = Ci;
	x[7] = Cc;

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
	//data.X = X;
	data.U = U;
	data.C = C;
	data.D = D;
	data.W = W;
	data.Ap = Ap;
	data.Ac = Ac;
	data.H = H;
	data.P = P;
	data.Rwin = Rwin;

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

	/*
	printf(" final l2 norm of the residuals%15.7g\n\n", (double)fnorm);
	printf(" number of function evaluations%10i\n\n", nfev);
	printf(" exit parameter %10i\n\n", info);
	*/

	// 結果を格納する
	R1 = x[0];
	R2 = x[1];
	R3 = x[2];
	R4 = x[3];
	Ce = x[4];
	Cp = x[5];
	Ci = x[6];
	Cc = x[7];

	//std::cout << "result values: R1=" << R1 << ", R2=" << R2 << ", R3=" << R3 << ", R4=" << R4 << ", Ce=" << Ce << ", Cp=" << Cp << ", Ci=" << Ci << ", Cc=" << Cc << std::endl;

	// メモリ解放
	delete[] y;
	delete[] fvec;
	delete[] fjac;
	delete[] wa4;

	return GrayBoxResult(fnorm, R1, R2, R3, R4, Ce, Cp, Ci, Cc);
}

double GrayBox::forward(double R1, double R2, double R3, double R4, double Ce, double Cp, double Ci, double Cc, double Rwin, double q_sol_factor, cv::Mat_<double>& predictedY) {
	predictedY = cv::Mat_<double>(2, U.cols);

	// Xの初期化
	cv::Mat_<double> x(4, 1);
	for (int i = 0; i < 4; ++i) {
		x(i, 0) = C2K(20);	// X(i, 0);
	}
	
	// build matrix A and B
	cv::Mat_<double> A(4, 4, 0.0);
	cv::Mat_<double> B(4, 3, 0.0);
	buildMatrices(R1, R2, R3, R4, Ce, Cp, Ci, Cc, W, Ap, Ac, H, P, Rwin, A, B);

	/*
	std::cout << "A: " << A << std::endl;
	std::cout << "B: " << B << std::endl;
	std::cout << "C: " << C << std::endl;
	std::cout << "D: " << D << std::endl;
	*/

	// Yを予測する
	double fnorm = 0.0;
	for (int t = 0; t < U.cols; ++t) {
		//std::cout << "x: " << x << std::endl;
		//std::cout << "u: " << U.col(t) << std::endl;

		U(0, t) *= q_sol_factor;


		// dx/dt = Ax + Bu
		cv::Mat_<double> dx = A * x + B * U.col(t);
		x += dx * 3600; // 1 hour = 3600 sec

		//std::cout << "dx: " << dx << std::endl;

		//std::cout << x(0, 0) << "," << x(1, 0) << "," << x(2, 0) << "," << x(3, 0) << std::endl;

		// y = Cx + Du
		predictedY.col(t) = C * x + D * U.col(t);

		for (int k = 0; k < Y.rows; ++k) {
			fnorm += SQR(Y(k, t) - predictedY(k, t));
		}
	}

	return sqrt(fnorm / U.cols);
}
