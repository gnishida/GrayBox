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
	//gb.forward(predictedY);


	// GrayBox
	{
		double R1 = (gb.ranges["R1"].minimum + gb.ranges["R1"].maximum) * 0.5;
		double R2 = (gb.ranges["R2"].minimum + gb.ranges["R2"].maximum) * 0.5;
		double R3 = (gb.ranges["R3"].minimum + gb.ranges["R3"].maximum) * 0.5;
		double R4 = (gb.ranges["R4"].minimum + gb.ranges["R4"].maximum) * 0.5;
		double Rwin = (gb.ranges["Rwin"].minimum + gb.ranges["Rwin"].maximum) * 0.5;
		double Ce = (gb.ranges["Ce"].minimum + gb.ranges["Ce"].maximum) * 0.5;
		double Cp = (gb.ranges["Cp"].minimum + gb.ranges["Cp"].maximum) * 0.5;
		double Ci = (gb.ranges["Ci"].minimum + gb.ranges["Ci"].maximum) * 0.5;
		double Cc = (gb.ranges["Cc"].minimum + gb.ranges["Cc"].maximum) * 0.5;
		gb.inverse(R1, R2, R3, R4, Rwin, Ce, Cp, Ci, Cc);
	}



	for (double R1 = gb.ranges["R1"].minimum; R1 < gb.ranges["R1"].maximum + gb.ranges["R1"].step; R1 += gb.ranges["R1"].step) {
		for (double R2 = gb.ranges["R2"].minimum; R2 < gb.ranges["R2"].maximum + gb.ranges["R2"].step; R2 += gb.ranges["R2"].step) {
			for (double R3 = gb.ranges["R3"].minimum; R3 < gb.ranges["R3"].maximum + gb.ranges["R3"].step; R3 += gb.ranges["R3"].step) {
				for (double R4 = gb.ranges["R4"].minimum; R4 < gb.ranges["R4"].maximum + gb.ranges["R4"].step; R4 += gb.ranges["R4"].step) {
					for (double Rwin = gb.ranges["Rwin"].minimum; Rwin < gb.ranges["Rwin"].maximum + gb.ranges["Rwin"].step; Rwin += gb.ranges["Rwin"].step) {
						for (double Ce = gb.ranges["Ce"].minimum; Ce < gb.ranges["Ce"].maximum + gb.ranges["Ce"].step; Ce += gb.ranges["Ce"].step) {
							for (double Cp = gb.ranges["Cp"].minimum; Cp < gb.ranges["Cp"].maximum + gb.ranges["Cp"].step; Cp += gb.ranges["Cp"].step) {
								for (double Ci = gb.ranges["Ci"].minimum; Ci < gb.ranges["Ci"].maximum + gb.ranges["Ci"].step; Ci += gb.ranges["Ci"].step) {
									for (double Cc = gb.ranges["Cc"].minimum; Cc < gb.ranges["Cc"].maximum + gb.ranges["Cc"].step; Cc += gb.ranges["Cc"].step) {
										gb.inverse(R1, R2, R3, R4, Rwin, Ce, Cp, Ci, Cc);
									}
								}
							}
						}
					}
				}
			}
		}
	}

	return 0;
}