import pandas as pd
import seaborn as sns

#MADQN
from matplotlib import pyplot as plt

MADQN_results_modified = pd.read_csv('../results/Apr_20_23_52_45/episodes_rewards_10000.csv')

MADQN_results_modified = MADQN_results_modified.drop(['nb_episodes','episode_rewards'],axis=1)

MADQN_results_modified.rename(columns={'avg_rewards': 'MADQN_avg_rewards_modified', 'std_rewards': 'MADQN_std_rewards_modified'}, inplace=True)


MADQN_results_unmodified = pd.read_csv('../results/Apr_19_17_18_39/episodes_rewards_10000.csv')

MADQN_results_unmodified = MADQN_results_unmodified.drop(['nb_episodes','episode_rewards'],axis=1)

MADQN_results_unmodified.rename(columns={'avg_rewards': 'MADQN_avg_rewards_unmodified', 'std_rewards': 'MADQN_std_rewards_unmodified'}, inplace=True)


rewards_merged_df = pd.merge(MADQN_results_modified, MADQN_results_unmodified, on='eval_episodes', how='inner')


colors = [0, 5, 2, 6, 1, 3]
color_cycle = sns.color_palette()

alpha = 0.3
legend_size = 15
line_size_others = 2
line_size_ours = 2
tick_size = 18
label_size = 18

save_location = "../Result_Graphs/Comparison/"


# Combined
fig, ax = plt.subplots(figsize=(19.2, 10.8))
ax.set_title('Comparison in Medium Mode - MADQN', size=label_size)

# Plot MAPPO
ax.plot(rewards_merged_df['eval_episodes'], rewards_merged_df['MADQN_avg_rewards_modified'], lw=line_size_others, label='MADQN_Modified', linestyle=':', color=color_cycle[colors[0]])
ax.fill_between(rewards_merged_df['eval_episodes'], rewards_merged_df['MADQN_avg_rewards_modified'] - rewards_merged_df['MADQN_std_rewards_modified'], rewards_merged_df['MADQN_avg_rewards_modified'] + rewards_merged_df['MADQN_std_rewards_modified'], facecolor=color_cycle[colors[0]], alpha=alpha)

# Plot MADQN
ax.plot(rewards_merged_df['eval_episodes'], rewards_merged_df['MADQN_avg_rewards_unmodified'], lw=line_size_others, label='MAACKTR_Unmodified', linestyle=':', color=color_cycle[colors[1]])
ax.fill_between(rewards_merged_df['eval_episodes'], rewards_merged_df['MADQN_avg_rewards_unmodified'] - rewards_merged_df['MADQN_std_rewards_unmodified'], rewards_merged_df['MADQN_avg_rewards_unmodified'] + rewards_merged_df['MADQN_std_rewards_unmodified'], facecolor=color_cycle[colors[1]], alpha=alpha)

ax.legend(fontsize=legend_size, loc='upper left', ncol=1)
ax.set_xlabel('Evaluation Epochs', fontsize=label_size)
ax.set_ylabel('Evaluation Reward', fontsize=label_size)
ax.tick_params(axis='x', labelsize=tick_size)
ax.tick_params(axis='y', labelsize=tick_size)
ax.grid(True)

plt.tight_layout()

plt.savefig(save_location+"Comparison-MADQN")