import numpy as np


def majority_voting(row_proximity, row_angle, row_separation):
	prox_action = np.argmax(row_proximity)
	angle_action = np.argmax(row_angle)
	separation_action = np.argmax(row_separation)
	prox_vote = np.zeros(5)
	prox_vote[prox_action] = 1
	angle_vote = np.zeros(5)
	angle_vote[angle_action] = 1
	separation_vote = np.zeros(5)
	separation_vote[separation_action] = 1
	vote_array = prox_vote + angle_vote + separation_vote
	action = np.argmax(vote_array)
	return action

def row_to_ranking(prob_row):
	num_actions = len(prob_row)
	ranking = np.zeros(num_actions)
	i = num_actions
	for _ in range(num_actions):
		best_action = np.argmax(prob_row)
		prob_row[best_action] = -1000
		ranking[best_action] = i
		i = i - 1
	return ranking

def ranking_voting(row_proximity, row_angle, row_separation):
	ranking_proximity = row_to_ranking(row_proximity)
	ranking_angle = row_to_ranking(row_angle)
	ranking_separation = row_to_ranking(row_separation)
	ranking = ranking_proximity + ranking_angle + ranking_separation
	action = np.argmax(ranking)
	return action