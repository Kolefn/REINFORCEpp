/*
 * DQNAgent.h
 *
 *  Created on: Nov 19, 2017
 *      Author: kole
 */
#ifndef DQNAGENT_H
#define DQNAGENT_H

#include <vector>
#include "json.hpp"
#include "Matrix.h"
#include "Network.h"
#include "Graph.h"
struct agent_options {
	 int NUM_STATE_FEATURES = 0;
	 int MAX_NUM_ACTIONS = 0;
	 int NUM_HIDDEN_UNITS=100;
	 float gamma=0.75;
	 float epsilon=0.1;
	 float alpha=0.01;
	 int experience_add_every=25;
	 int EXPERIENCE_SIZE=5000;
	 int learning_steps_per_iteration=10;
	 float tderror_clamp=1.0;
//     agent_options(int _NUM_STATE_FEATURES, int _MAX_NUM_ACTIONS, int _NUM_HIDDEN_UNITS,
//             float _gamma, float _epsilon, float _alpha, int _experience_add_every,
//             int _EXPERIENCE_SIZE, int _learning_steps_per_iteration, float _tderror_clamp) {
//         NUM_STATE_FEATURES = _NUM_STATE_FEATURES;
//         MAX_NUM_ACTIONS = _MAX_NUM_ACTIONS;
//         NUM_HIDDEN_UNITS = _NUM_HIDDEN_UNITS;
//         gamma = _gamma;
//         epsilon = _epsilon;
//         alpha = _alpha;
//         experience_add_every = _experience_add_every;
//         EXPERIENCE_SIZE = _EXPERIENCE_SIZE;
//         learning_steps_per_iteration = _learning_steps_per_iteration;
//         tderror_clamp = _tderror_clamp;
//     };
//    agent_options();
};

//@TODO should s0, s1 be pointers?
//@TODO manage memory of agents SARSA memory
struct SARSA {
	Matrix s0;
	int a0;
	float r0;
	Matrix s1;
	int a1;
	SARSA() : s0(Matrix(0,0)), a0(0), r0(std::numeric_limits<float>::infinity()), s1(Matrix(0,0)),a1(0){}
	SARSA(const SARSA& s) : s0(s.s0), a0(s.a0), r0(s.r0), s1(s.s1), a1(s.a1){}
	SARSA& operator=(const SARSA& s){
		s0 = s.s0;
		a0 = s.a0;
		r0 = s.r0;
		s1 = s.s1;
		a1 = s.a1;
		return *this;
	}
};
class DQNAgent {
public:
    DQNAgent();
    void set(int _NUM_STATE_FEATURES, int _MAX_NUM_ACTIONS, int _NUM_HIDDEN_UNITS,
             float _gamma, float _epsilon, float _alpha, int _experience_add_every,
        int _EXPERIENCE_SIZE, int _learning_steps_per_iteration, float _tderror_clamp);
	DQNAgent(agent_options options);
	DQNAgent(json j);
	virtual ~DQNAgent();

	int act(std::vector<float> slist);
	void learn(float r1);


	json* toJSON();
	void fromJSON();
    
    void adjustAlpha(float x);
    
    void adjustEpsilon(float x);
    
    void adjustGamma(float x);

    int getStateFeatures();

private:

	Matrix* forwardQ(Matrix* s, bool needs_backprop);
	float learnFromTuple(SARSA _sarsa);

	int experience_add_every;
	//@TODO can this be dynamic?
	int EXPERIENCE_SIZE;
	int learning_steps_per_iteration;
	float tderror_clamp;

	float gamma; //future reward discount factor. The agent 'greediness'.
	float epsilon; //Epsilon-greedy policy, The 'risk' or 'randomness' of the agent.
	float alpha; //Value function learning rate. How much the weights are adjusted.


	int nh; //number of hidden units in the network
	int ns; //number of state features
	int na; //number of possible actions

	Network net;

	std::vector<SARSA> exp; //experience of previous states,actions,rewards
	int expi; //index of exp array
	SARSA sarsa;

	long long t; //elapsed time

	float tderror;


	Graph* lastG;
};

#endif
