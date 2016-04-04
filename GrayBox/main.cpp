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
	gb.forward((gb.ranges["R1"].minimum + gb.ranges["R1"].maximum) * 0.5,
		(gb.ranges["R2"].minimum + gb.ranges["R2"].maximum) * 0.5, 
		(gb.ranges["R3"].minimum + gb.ranges["R3"].maximum) * 0.5,
		(gb.ranges["R4"].minimum + gb.ranges["R4"].maximum) * 0.5,
		(gb.ranges["Ce"].minimum + gb.ranges["Ce"].maximum) * 0.5,
		(gb.ranges["Cp"].minimum + gb.ranges["Cp"].maximum) * 0.5,
		(gb.ranges["Ci"].minimum + gb.ranges["Ci"].maximum) * 0.5,
		(gb.ranges["Cc"].minimum + gb.ranges["Cc"].maximum) * 0.5,
		predictedY);


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

	for (double R1 = gb.ranges["R1"].minimum + gb.ranges["R1"].step; R1 < gb.ranges["R1"].maximum; R1 += gb.ranges["R1"].step) {
		for (double R2 = gb.ranges["R2"].minimum + gb.ranges["R2"].step; R2 < gb.ranges["R2"].maximum; R2 += gb.ranges["R2"].step) {
			for (double R3 = gb.ranges["R3"].minimum + gb.ranges["R3"].step; R3 < gb.ranges["R3"].maximum; R3 += gb.ranges["R3"].step) {
				for (double R4 = gb.ranges["R4"].minimum + gb.ranges["R4"].step; R4 < gb.ranges["R4"].maximum; R4 += gb.ranges["R4"].step) {
					for (double Ce = gb.ranges["Ce"].minimum + gb.ranges["Ce"].step; Ce < gb.ranges["Ce"].maximum; Ce += gb.ranges["Ce"].step) {
						for (double Cp = gb.ranges["Cp"].minimum + gb.ranges["Cp"].step; Cp < gb.ranges["Cp"].maximum; Cp += gb.ranges["Cp"].step) {
							for (double Ci = gb.ranges["Ci"].minimum + gb.ranges["Ci"].step; Ci < gb.ranges["Ci"].maximum; Ci += gb.ranges["Ci"].step) {
								for (double Cc = gb.ranges["Cc"].minimum + gb.ranges["Cc"].step; Cc < gb.ranges["Cc"].maximum; Cc += gb.ranges["Cc"].step) {
									GrayBoxResult result = gb.inverse(R1, R2, R3, R4, Ce, Cp, Ci, Cc);
									if (result.fnorm < min_fnorm) {
										std::cout << "!!!!!!!!!!!!!! minimum updated !!!!!!!!!!!!!" << std::endl;
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
							}
						}
					}
				}
			}
		}
	}

	std::cout << "Best fnorm: " << min_fnorm << std::endl;
	std::cout << "result values: R1=" << min_R1 << ", R2=" << min_R2 << ", R3=" << min_R3 << ", R4=" << min_R4 << ", Ce=" << min_Ce << ", Cp=" << min_Cp << ", Ci=" << min_Ci << ", Cc=" << min_Cc << std::endl;





	return 0;
}