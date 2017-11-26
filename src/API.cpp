/*
//  API.cpp
//  RPP
//
//  Created by Kole Nunley on 11/22/17.
//  Copyright © 2017 Kole Nunley. All rights reserved.
 
    The api exposes several functions so other programs can create agent with the RPP bundle.
*/
#include "API.hpp"

DQNAgent* addAgent(int _NUM_STATE_FEATURES, int _MAX_NUM_ACTIONS, int _NUM_HIDDEN_UNITS,
                   float _gamma, float _epsilon, float _alpha, int _experience_add_every,
                   int _EXPERIENCE_SIZE, int _learning_steps_per_iteration, float _tderror_clamp){
    
    agent_options options;
    options.NUM_STATE_FEATURES = _NUM_STATE_FEATURES;
    options.MAX_NUM_ACTIONS = _MAX_NUM_ACTIONS;
    options.NUM_HIDDEN_UNITS = _NUM_HIDDEN_UNITS;
    options.gamma = _gamma;
    options.epsilon = _epsilon;
    options.alpha = _alpha;
    options.experience_add_every = _experience_add_every;
    options.EXPERIENCE_SIZE = _EXPERIENCE_SIZE;
    options.learning_steps_per_iteration = _learning_steps_per_iteration;
    options.tderror_clamp = _tderror_clamp;
    return new DQNAgent(options);
}
void deleteAgent(DQNAgent* agent){
    agent->~DQNAgent();
}

void rewardAgent(DQNAgent* agent, float reward, int dAlpha, int dEpsilon){
    agent->learn(reward);
    agent->adjustAlpha(dAlpha);
    agent->adjustEpsilon(dEpsilon);
}
int  getActionFromAgent(DQNAgent* agent, float* state){
    //state array to state vector
    int n =  agent->getStateFeatures();
    std::vector<float> v;
    for(int i = 0; i < n; i++){
        v.push_back(*(state+i));
    }
    
    return agent->act(v);
}
