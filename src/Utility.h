/*
 * Utility.h
 *
 *  Created on: Nov 15, 2017
 *      Author: Kole Nunley
 */
#ifndef UTILITY_H
#define UTILITY_H

#include <iostream>
#include <fstream>
#include <random>
#include <vector>
#include <math.h>
#include "json.hpp"

class Utility {
public:

	static float gaussRandom();
	static float randomFloat(float min, float max);
	static int randomInt(int min, int max);
	static float randomN(float mu, float std);
    static std::vector<float>* zeros(int n);
    static void writeJSON(nlohmann::json* j, std::string filepath, bool del = true);
    static nlohmann::json* readJSON(std::string filepath);
    static float sig(float x);
private:
	//caching for better random generation
	static bool return_v;
	static float v_val;
};

#endif /* UTILITY_H_ */
