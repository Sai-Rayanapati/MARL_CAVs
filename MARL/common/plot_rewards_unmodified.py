import pandas as pd
import seaborn as sns

#MADQN
from matplotlib import pyplot as plt

MADQN_results = pd.read_csv('../results/Apr_20_23_52_05/episodes_rewards_10000.csv')

MADQN_results = MADQN_results.drop(['nb_episodes','episode_rewards'],axis=1)

MADQN_results.rename(columns={'avg_rewards': 'MADQN_avg_rewards', 'std_rewards': 'MADQN_std_rewards'}, inplace=True)

#MAPPO
MAPPO_results = pd.read_csv('../results/Apr_20_23_52_31/episodes_rewards_10000.csv')

MAPPO_results = MAPPO_results.drop(['nb_episodes','episode_rewards'],axis=1)

MAPPO_results.rename(columns={'avg_rewards': 'MAPPO_avg_rewards', 'std_rewards': 'MAPPO_std_rewards'}, inplace=True)

#MAACKTR
MAACKTR_results = pd.read_csv('../results/Apr_19_17_18_39/episodes_rewards_10000.csv')

MAACKTR_results = MAACKTR_results.drop(['nb_episodes','episode_rewards'],axis=1)

MAACKTR_results.rename(columns={'avg_rewards': 'MAACKTR_avg_rewards', 'std_rewards': 'MAACKTR_std_rewards'}, inplace=True)

# Merge the df
merged_df = pd.merge(MADQN_results, MAPPO_results, on='eval_episodes', how='inner')

rewards_merged_df = pd.merge(merged_df, MAACKTR_results, on='eval_episodes', how='inner')


colors = [0, 5, 2, 6, 1, 3]
color_cycle = sns.color_palette()

alpha = 0.3
legend_size = 15
line_size_others = 2
line_size_ours = 2
tick_size = 18
label_size = 18

save_location = "../Result_Graphs/Unmodified/"

# MAPPO
fig, ax0 = plt.subplots(figsize=(19.2, 10.8))
ax0.set_title('MAPPO - Medium Mode', size=label_size)
ax0.plot(rewards_merged_df['eval_episodes'], rewards_merged_df['MAPPO_avg_rewards'], lw=line_size_others, label='MAPPO', linestyle=':',
         color=color_cycle[colors[0]])
ax0.fill_between(rewards_merged_df['eval_episodes'], rewards_merged_df['MAPPO_avg_rewards'] - rewards_merged_df['MAPPO_std_rewards'], rewards_merged_df['MAPPO_avg_rewards'] + rewards_merged_df['MAPPO_std_rewards'],
                 facecolor=color_cycle[colors[0]], edgecolor='none', alpha=alpha)

leg1 = ax0.legend(fontsize=legend_size, loc='lower right', ncol=2)

ax0.tick_params(axis='x', labelsize=tick_size)
ax0.tick_params(axis='y', labelsize=tick_size)
ax0.set_xlabel('Evaluation epochs', fontsize=label_size)
ax0.set_ylabel('Evaluation reward', fontsize=label_size)
ax0.ticklabel_format(axis="x")

ax0.grid()

plt.tight_layout()
plt.savefig(save_location+"MAPPO")
# plt.show()

# MADQN
fig, ax1 = plt.subplots(figsize=(19.2, 10.8))
ax1.set_title('MADQN - Medium Mode', size=label_size)
ax1.plot(rewards_merged_df['eval_episodes'], rewards_merged_df['MADQN_avg_rewards'], lw=line_size_others, label='MADQN', linestyle=':',
         color=color_cycle[colors[0]])
ax1.fill_between(rewards_merged_df['eval_episodes'], rewards_merged_df['MADQN_avg_rewards'] - rewards_merged_df['MADQN_std_rewards'], rewards_merged_df['MADQN_avg_rewards'] + rewards_merged_df['MADQN_std_rewards'],
                 facecolor=color_cycle[colors[0]], edgecolor='none', alpha=alpha)

leg1 = ax1.legend(fontsize=legend_size, loc='lower right', ncol=2)

ax1.tick_params(axis='x', labelsize=tick_size)
ax1.tick_params(axis='y', labelsize=tick_size)
ax1.set_xlabel('Evaluation epochs', fontsize=label_size)
ax1.set_ylabel('Evaluation reward', fontsize=label_size)
ax1.ticklabel_format(axis="x")

ax1.grid()

plt.tight_layout()
plt.savefig(save_location+"MADQN")
# plt.show()

# MAACKTR
fig, ax2 = plt.subplots(figsize=(19.2, 10.8))
ax2.set_title('MAACKTR - Medium Mode', size=label_size)
ax2.plot(rewards_merged_df['eval_episodes'], rewards_merged_df['MAACKTR_avg_rewards'], lw=line_size_others, label='MAACKTR', linestyle=':',
         color=color_cycle[colors[0]])
ax2.fill_between(rewards_merged_df['eval_episodes'], rewards_merged_df['MAACKTR_avg_rewards'] - rewards_merged_df['MAACKTR_std_rewards'], rewards_merged_df['MAACKTR_avg_rewards'] + rewards_merged_df['MAACKTR_std_rewards'],
                 facecolor=color_cycle[colors[0]], edgecolor='none', alpha=alpha)

leg1 = ax2.legend(fontsize=legend_size, loc='lower right', ncol=2)

ax2.tick_params(axis='x', labelsize=tick_size)
ax2.tick_params(axis='y', labelsize=tick_size)
ax2.set_xlabel('Evaluation epochs', fontsize=label_size)
ax2.set_ylabel('Evaluation reward', fontsize=label_size)
ax2.ticklabel_format(axis="x")

ax2.grid()

plt.tight_layout()
plt.savefig(save_location+"MAACKTR")
# plt.show()

# Combined
fig, ax = plt.subplots(figsize=(19.2, 10.8))
ax.set_title('Comparison in Medium Mode', size=label_size)

# Plot MAPPO
ax.plot(rewards_merged_df['eval_episodes'], rewards_merged_df['MAPPO_avg_rewards'], lw=line_size_others, label='MAPPO', linestyle=':', color=color_cycle[colors[0]])
ax.fill_between(rewards_merged_df['eval_episodes'], rewards_merged_df['MAPPO_avg_rewards'] - rewards_merged_df['MAPPO_std_rewards'], rewards_merged_df['MAPPO_avg_rewards'] + rewards_merged_df['MAPPO_std_rewards'], facecolor=color_cycle[colors[0]], alpha=alpha)

# Plot MADQN
ax.plot(rewards_merged_df['eval_episodes'], rewards_merged_df['MADQN_avg_rewards'], lw=line_size_others, label='MADQN', linestyle=':', color=color_cycle[colors[1]])
ax.fill_between(rewards_merged_df['eval_episodes'], rewards_merged_df['MADQN_avg_rewards'] - rewards_merged_df['MADQN_std_rewards'], rewards_merged_df['MADQN_avg_rewards'] + rewards_merged_df['MADQN_std_rewards'], facecolor=color_cycle[colors[1]], alpha=alpha)

# Plot MAACKTR
ax.plot(rewards_merged_df['eval_episodes'], rewards_merged_df['MAACKTR_avg_rewards'], lw=line_size_others, label='MAACKTR', linestyle=':', color=color_cycle[colors[2]])
ax.fill_between(rewards_merged_df['eval_episodes'], rewards_merged_df['MAACKTR_avg_rewards'] - rewards_merged_df['MAACKTR_std_rewards'], rewards_merged_df['MAACKTR_avg_rewards'] + rewards_merged_df['MAACKTR_std_rewards'], facecolor=color_cycle[colors[2]], alpha=alpha)

ax.legend(fontsize=legend_size, loc='upper left', ncol=1)
ax.set_xlabel('Evaluation Epochs', fontsize=label_size)
ax.set_ylabel('Evaluation Reward', fontsize=label_size)
ax.tick_params(axis='x', labelsize=tick_size)
ax.tick_params(axis='y', labelsize=tick_size)
ax.grid(True)

plt.tight_layout()

plt.savefig(save_location+"Comparison")