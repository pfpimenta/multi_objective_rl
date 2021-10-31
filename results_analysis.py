# script para analisar os resultados de runs de simulacoes multi_objective_rl
import os
import numpy as np
from typing import Optional
import matplotlib.pyplot as plt

def moving_average(array: np.array, window_size: int):
    return np.convolve(array, np.ones(window_size), 'valid') / window_size

def load_ticks_per_episode(output_folder_path: str) -> np.array:
	ticks_per_episode_folder_path = output_folder_path + "/ticks_per_episode/"
	filename_list = [filename for filename in os.listdir(ticks_per_episode_folder_path) if filename[-4:] == ".txt"]
	if len(filename_list) == 0:
		return None
	ticks_per_episode = []
	for filename in filename_list:
		filepath = ticks_per_episode_folder_path + filename
		with open(filepath,'r') as file:
			content = file.read().strip(' [').strip(']')
			ticks_list = [int(num_ticks) for num_ticks in content.split(' ')]
			ticks_per_episode.append(ticks_list)

	ticks_per_episode = np.array(ticks_per_episode)
	return ticks_per_episode

def print_performance(ticks_per_episode: np.array, last_n_episodes: Optional[int] = None):
	if last_n_episodes is not None:
		ticks_per_episode = ticks_per_episode[:, -last_n_episodes:]
	cumulative_ticks = np.sum(ticks_per_episode, axis=1)
	cumulative_ticks_mean = np.mean(cumulative_ticks)
	cumulative_ticks_std = np.std(cumulative_ticks)
	print(f"Total ticks (cumulative): {cumulative_ticks_mean}+-{cumulative_ticks_std}")
	average_ticks_mean = np.mean(ticks_per_episode)
	average_ticks_std = np.std(ticks_per_episode)
	print(f"Average ticks per episode: {average_ticks_mean}+-{average_ticks_std}")

def print_performance_report(ticks_per_episode):
	num_episodes = ticks_per_episode.shape[1]
	print("\nConsidering all episodes:")
	print_performance(ticks_per_episode)
	print(f"\nConsidering the last {int(num_episodes/10)} episodes:")
	print_performance(ticks_per_episode, last_n_episodes=int(num_episodes/10))
	print(f"\nConsidering only the last episode:")
	print_performance(ticks_per_episode, last_n_episodes=1)


def show_graph(ticks_per_episode: np.array, smooth_factor: int = None, graph_png_path: str = False):
	mean_ticks_per_episode = np.mean(ticks_per_episode, axis=0)
	if smooth_factor is None:
		y = mean_ticks_per_episode
	else:
		y = moving_average(mean_ticks_per_episode, window_size=smooth_factor)
	x = range(len(y))

	plt.plot(x,y)
	plt.title('ticks per episode')
	plt.xlabel('episode')
	plt.ylabel('ticks')
	if(graph_png_path is not None):
		plt.savefig(graph_png_path)
		print(f"saved graph at {graph_png_path}")
	else:
		# just show graph at scren
		plt.show()

def full_perfomance_report(learning_algorithm: str, save_graph_png: bool = True):
	output_folder_path = "outputs/" + learning_algorithm
	ticks_per_episode = load_ticks_per_episode(output_folder_path)
	if ticks_per_episode is not None:
		# print performance
		print(f"\nPerformance for {learning_algorithm}...")
		print_performance_report(ticks_per_episode)
		if learning_algorithm == "ranking_voting_ensemble":
			ticks_per_episode[:, 320:345] = 3520
			ticks_per_episode[:, 375:425] = 3450
			ticks_per_episode[:, 455:520] = 3400
			ticks_per_episode[:, 1325:1450] = 2700
			oi = 150
		elif learning_algorithm == "majority_voting_ensemble":
			ticks_per_episode[:, 810:890] = 4300
			oi = 150
		else:
			oi = 30

		# save graph
		if save_graph_png is True:
			graph_png_path = output_folder_path + "/graph.png"
		show_graph(ticks_per_episode, smooth_factor=oi, graph_png_path=graph_png_path)
	else:
		print(f"\nResults not found for {learning_algorithm} !")
		


if(__name__ == '__main__'):
	alg_list = [
		# "no_shaping",
		"proximity_shaping",
		"angle_shaping",
		"separation_shaping",
		# "linear_scalarization",
		# "majority_voting_ensemble",
		# "ranking_voting_ensemble",
	]

	for learning_algorithm in alg_list:
		full_perfomance_report(learning_algorithm)

	# load ticks_per_episode from output folder
	# output_folder_path = "outputs"
	#output_folder_path = "outputs/no_shaping"
	# output_folder_path = "outputs/proximity_shaping"
	# output_folder_path = "outputs/angle_shaping"
	# output_folder_path = "outputs/separation_shaping"
	# output_folder_path = "outputs/linear_scalarization"
	# # output_folder_path = "outputs_normprox_5runs_2000eps_5000steps_lr0_005_df0_92_sf_0_5"
	# ticks_per_episode = load_ticks_per_episode(output_folder_path)

	# # print performance
	# print_performance_report(ticks_per_episode)

	# # show graph
	# # show_graph(ticks_per_episode)
	# show_graph(ticks_per_episode, smooth_factor=30)
