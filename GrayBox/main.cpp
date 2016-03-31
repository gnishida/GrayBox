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
	if (argc < 3) {
		std::cout << "Usage: " << argv[0] << " <simulation file name> <range file name>" << std::endl;
		return -1;
	}

	GrayBox gb;
	gb.loadTrainingData(argv[1], argv[2]);


	// forward test
	cv::Mat_<double> predictedY;
	gb.forward(predictedY);


	// GrayBox
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