/*
 * Network.cpp
 *
 *  Created on: Nov 18, 2017
 *      Author: kole
 */

#include "Network.h"

Network::Network() {

}

//network from json
void Network::fromJSON(json j) {

	for(json::iterator it = j.begin(); it != j.end(); ++it){ //parse json for matrix keys
		Matrix* m = new Matrix(it.value()); //create matrix from matrix json
		this->add(it.key(), m);
	}

}
Network::Network(const Network &n){
	map = n.map;
}
Network& Network::operator=(const Network &n){
	map = n.map;
	return *this;
}



void Network::add(std::string key, Matrix* m ){
	if(map.count(key) == 1){
		std::cout << "Network::add key already exists" << std::endl;
		throw 0;
	}else {
		map[key] = m;
	}
}

Matrix* Network::getMatrix(std::string key){
	if(map.count(key) == 1){
		return map[key];
	}else{
		std::cout << "Network::getMatrix no key exists!" << std::endl;
		throw 0;
	}
}

void Network::setMatrix(std::string key, Matrix* m){
	if(map.count(key) == 1){
		map[key] = m;
	}else{
		std::cout << "Network::setMatrix no key exists!" << std::endl;
		throw 0;
	}
}



//call update on all matrices in network
void Network::update(float alpha){
	for(auto& pair : map){
			pair.second->update(alpha);
	}
}


//call gradFillConst with 0 on all matrices
void Network::zeroGrads(){
	for(auto& pair : map){
		pair.second->gradFillConst(0);
	}
}

Matrix* Network::flattenGrads(){
	int netWeightCount = 0;
	for(auto& pair : map){
		netWeightCount += pair.second->length();
	}
	//create matrix with as many rows as weights in network
	//and only 1 column. 'flat'
	Matrix* g = new Matrix(netWeightCount,1);
	int ix = 0;
	for(auto& pair : map){
		Matrix* m = pair.second;
		int weights = m->length();
		for(int i = 0; i < weights; i++){
			g->setWeight(ix, m->getDWeight(i));

			ix++;
		}
	}

	return g;
}


Network* Network::copy(){
	Network* net = new Network();
	for(const auto& pair : map){
		net->add(pair.first,pair.second);
	}

	return net;
}
json* Network::toJSON(){
	json* j = new json;
	for(auto& pair : map){ //parse map keys
		json* temp = pair.second->toJSON();
		(*j)[pair.first] = (*temp); //set key-matrix pairs in json
		delete temp; //never used again so delete
	}

	return j;
}



Network::~Network() {
	//carefully destroy all matrices in the network
	for(std::map<std::string, Matrix*>::iterator itr = map.begin(); itr != map.end(); itr++)
	{
		delete (itr->second);
	}
	map.clear();
}

