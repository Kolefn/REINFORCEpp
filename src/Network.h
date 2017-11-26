/*
 * Network.h
 *
 *  Created on: Nov 18, 2017
 *      Author: kole
 *
 *      A network abstracts a mapping of Matrix objects
 */
#ifndef NETWORK_H
#define NETWORK_H

#include <string>
#include <map>
#include "Matrix.h"

using json = nlohmann::json;
class Network {
public:
	Network();
	Network(const Network& n);
	Network& operator=(const Network &n);
	virtual ~Network();

	void add(std::string key, Matrix* m);
	Matrix* getMatrix(std::string);
	void setMatrix(std::string key, Matrix* m);

	void update(float alpha);
	void zeroGrads();
	Matrix* flattenGrads();


    void fromJSON(json j);
	Network* copy();
	json* toJSON();

private:
	std::map<std::string, Matrix*> map;
};

#endif
