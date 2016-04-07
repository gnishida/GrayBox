/**
 * This is a sample implementation of the Gray Box model described in
 * "An Inverse Gray-Box Model for Transient Building Load Prediction"
 * by Braun and Chaturvedi.
 *
 * @author Gen Nishida
 * @date 3/7/2016
 */

#include <iostream>
#include "GrayBox.h"
#include <fstream>

int main(int argc, char* argv[]) {
	GrayBox gb;
	gb.loadTrainingData("../data/simulation_60ft.csv", "../data/range_60ft.csv");

	// forward test
	cv::Mat_<double> predictedY;
	double fnorm = gb.forward(
		0.002,		
		0.004,
		0.0008,
		0.004,
		22727800,
		3973900,
		54363900,
		2991300,
		predictedY);
	std::cout << "Fnorm: " << fnorm << std::endl;
	/*
	gb.forward((gb.ranges["R1"].minimum + gb.ranges["R1"].maximum) * 0.5,
		(gb.ranges["R2"].minimum + gb.ranges["R2"].maximum) * 0.5, 
		(gb.ranges["R3"].minimum + gb.ranges["R3"].maximum) * 0.5,
		(gb.ranges["R4"].minimum + gb.ranges["R4"].maximum) * 0.5,
		(gb.ranges["Ce"].minimum + gb.ranges["Ce"].maximum) * 0.5,
		(gb.ranges["Cp"].minimum + gb.ranges["Cp"].maximum) * 0.5,
		(gb.ranges["Ci"].minimum + gb.ranges["Ci"].maximum) * 0.5,
		(gb.ranges["Cc"].minimum + gb.ranges["Cc"].maximum) * 0.5,
		predictedY);
	*/

	return 0;

	// GrayBox
	/*
	{
		double R1 = (gb.ranges["R1"].minimum + gb.ranges["R1"].maximum) * 0.5;
		double R2 = (gb.ranges["R2"].minimum + gb.ranges["R2"].maximum) * 0.5;
		double R3 = (gb.ranges["R3"].minimum + gb.ranges["R3"].maximum) * 0.5;
		double R4 = (gb.ranges["R4"].minimum + gb.ranges["R4"].maximum) * 0.5;
		double Ce = (gb.ranges["Ce"].minimum + gb.ranges["Ce"].maximum) * 0.5;
		double Cp = (gb.ranges["Cp"].minimum + gb.ranges["Cp"].maximum) * 0.5;
		double Ci = (gb.ranges["Ci"].minimum + gb.ranges["Ci"].maximum) * 0.5;
		double Cc = (gb.ranges["Cc"].minimum + gb.ranges["Cc"].maximum) * 0.5;
		gb.inverse(R1, R2, R3, R4, Ce, Cp, Ci, Cc);
	}
	*/


	double min_fnorm = std::numeric_limits<double>::max();
	double min_R1, min_R2, min_R3, min_R4, min_Ce, min_Cp, min_Ci, min_Cc;

	int progress_count = 0;
	for (double R1 = gb.ranges["R1"].minimum; R1 < gb.ranges["R1"].maximum + gb.ranges["R1"].step * 0.5; R1 += gb.ranges["R1"].step) {
		for (double R2 = gb.ranges["R2"].minimum; R2 < gb.ranges["R2"].maximum + gb.ranges["R2"].step * 0.5; R2 += gb.ranges["R2"].step) {
			for (double R3 = gb.ranges["R3"].minimum; R3 < gb.ranges["R3"].maximum + gb.ranges["R3"].step * 0.5; R3 += gb.ranges["R3"].step) {
				for (double R4 = gb.ranges["R4"].minimum; R4 < gb.ranges["R4"].maximum + gb.ranges["R4"].step * 0.5; R4 += gb.ranges["R4"].step) {
					std::cout << progress_count + 1 << "/256..." << std::endl;

					for (double Ce = gb.ranges["Ce"].minimum; Ce < gb.ranges["Ce"].maximum + gb.ranges["Ce"].step * 0.5; Ce += gb.ranges["Ce"].step) {
						for (double Cp = gb.ranges["Cp"].minimum; Cp < gb.ranges["Cp"].maximum + gb.ranges["Cp"].step * 0.5; Cp += gb.ranges["Cp"].step) {
							for (double Ci = gb.ranges["Ci"].minimum; Ci < gb.ranges["Ci"].maximum + gb.ranges["Ci"].step * 0.5; Ci += gb.ranges["Ci"].step) {
								for (double Cc = gb.ranges["Cc"].minimum; Cc < gb.ranges["Cc"].maximum + gb.ranges["Cc"].step * 0.5; Cc += gb.ranges["Cc"].step) {
									double norm = gb.forward(R1, R2, R3, R4, Ce, Cp, Ci, Cc, predictedY);
									if (norm < min_fnorm) {
										std::cout << "min_fnorm: " << norm << ", R1=" << R1 << ", R2=" << R2 << ", R3=" << R3 << ", R4=" << R4 << ", Ce=" << Ce << ", Cp=" << Cp << ", Ci=" << Ci << ", Cc=" << Cc << std::endl;
										min_fnorm = norm;
										min_R1 = R1;
										min_R2 = R2;
										min_R3 = R3;
										min_R4 = R4;
										min_Ce = Ce;
										min_Cp = Cp;
										min_Ci = Ci;
										min_Cc = Cc;
									}

									/*
									if (rand() % 20 == 0) {
										GrayBoxResult result = gb.inverse(R1, R2, R3, R4, Ce, Cp, Ci, Cc);
										if (result.fnorm < min_fnorm) {
											std::cout << "min_fnorm: " << result.fnorm << std::endl;
											min_fnorm = result.fnorm;
											min_R1 = result.R1;
											min_R2 = result.R2;
											min_R3 = result.R3;
											min_R4 = result.R4;
											min_Ce = result.Ce;
											min_Cp = result.Cp;
											min_Ci = result.Ci;
											min_Cc = result.Cc;
										}
									}
									*/
								}
							}
						}
					}

					progress_count++;
				}
			}
		}
	}

	std::cout << "Best fnorm: " << min_fnorm << std::endl;
	std::cout << "result values: R1=" << min_R1 << ", R2=" << min_R2 << ", R3=" << min_R3 << ", R4=" << min_R4 << ", Ce=" << min_Ce << ", Cp=" << min_Cp << ", Ci=" << min_Ci << ", Cc=" << min_Cc << std::endl;

	GrayBoxResult result = gb.inverse(min_R1, min_R2, min_R3, min_R4, min_Ce, min_Cp, min_Ci, min_Cc);
	std::cout << "After the optimization: " << result.fnorm << std::endl;
	std::cout << "result values: R1=" << result.R1 << ", R2=" << result.R2 << ", R3=" << result.R3 << ", R4=" << result.R4 << ", Ce=" << result.Ce << ", Cp=" << result.Cp << ", Ci=" << result.Ci << ", Cc=" << result.Cc << std::endl;


	{
		double norm = 0.0;
		gb.forward(min_R1, min_R2, min_R3, min_R4, min_Ce, min_Cp, min_Ci, min_Cc, predictedY);
		for (int t = 0; t < predictedY.cols; ++t) {
			for (int k = 0; k < predictedY.rows; ++k) {
				if (k > 0) std::cout << ",";
				std::cout << predictedY(k, t);
				norm += (gb.Y(k, t) - predictedY(k, t)) * (gb.Y(k, t) - predictedY(k, t));
			}
			std::cout << std::endl;
		}
		std::cout << "norm: " << sqrt(norm / predictedY.cols) << std::endl;
	}

	return 0;
}