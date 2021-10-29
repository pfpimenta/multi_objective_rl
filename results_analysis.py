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
	print(f"Performance: {cumulative_ticks_mean}+-{cumulative_ticks_std}")

def show_graph(ticks_per_episode: np.array, smooth_factor: int = None):
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
	plt.show()


if(__name__ == '__main__'):
	# load ticks_per_episode from output folder
	# output_folder_path = "outputs"
	output_folder_path = "outputs_normprox_5runs_2000eps_5000steps_lr0_005_df0_92_sf_0_5"
	ticks_per_episode = load_ticks_per_episode(output_folder_path)

	# print performance
	num_episodes = ticks_per_episode.shape[1]
	print_performance(ticks_per_episode)
	print_performance(ticks_per_episode, last_n_episodes=int(num_episodes/10))
	print_performance(ticks_per_episode, last_n_episodes=1)

	# show graph
	avg_ticks_per_episode = np.mean(ticks_per_episode, axis=0)
	# show_graph(ticks_per_episode)
	show_graph(ticks_per_episode, smooth_factor=30)
