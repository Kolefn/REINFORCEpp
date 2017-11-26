/*
 * Utility.cpp
 *
 *  Created on: Nov 15, 2017
 *      Author: kole
 */

#include "Utility.h"

bool Utility::return_v = false;
float Utility::v_val = 0;

//gauss random technique
float Utility::gaussRandom(){
	if(return_v){
		return_v = false;
		return v_val;
	}

	float u = randomFloat(-1,1);
	float v = randomFloat(-1,1);
	float r = u*u + v*v;
	if(r == 0 || r > 1) return gaussRandom();
	float c = std::sqrt(-2*std::log(r)/r);
	v_val = v*c; //cached
	return_v = true;
	return u*c;
}

//create array of length n and fill with zeros
//vector similar to dynamic array in other languages like Java
std::vector<float>* Utility::zeros(int n){
	if(n > 0){
		std::vector<float> * array  = new std::vector<float>(n,0);

		return array;
	}else{
		return new std::vector<float>();
	}



}

//@TODO VERIFY THIS
float Utility::randomFloat(float min, float max){
		std::random_device rd;  //Will be used to obtain a seed for the random number engine
	    std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
	    std::uniform_real_distribution<> dis(min, max);
	    return dis(gen);
}
//one less than max
int Utility::randomInt(int min, int max){
	std::mt19937 rng;
	rng.seed(std::random_device()());
	std::uniform_int_distribution<std::mt19937::result_type> dist6(min, max-1);
	return dist6(rng);
}

//generate number based on matrix value and standard deviation
//and gaussian random.
float Utility::randomN(float mu, float std){
	return mu+gaussRandom()*std;
}

void Utility::writeJSON(nlohmann::json* j, std::string filepath, bool del){
    std::ofstream outputfile;
    outputfile.open(filepath);
	outputfile << std::setw(4) << (*j) << std::endl;
	outputfile.close();
	if(del){
		delete j;
	}
}


nlohmann::json* Utility::readJSON(std::string filepath){
    
	std::ifstream inputfile;
	inputfile.open(filepath);//open json file

	nlohmann::json* j = new nlohmann::json; //pointer -> new json
	inputfile >> (*j); //parse file

	return j;

}

//sigmoid helper function
//one divided by one plus e to the negative x
float Utility::sig(float x){
    float alpha = 0.01;
    return 0.5 * (x * alpha / (1 + std::abs(x*alpha)) + 0.5);
}

