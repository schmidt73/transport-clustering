import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

plt.style.use("~/plotting/paper.mplstyle")
sns.set_palette("deep")
SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=12)    # fontsize of the tick labels
plt.rc('ytick', labelsize=12)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('legend', title_fontsize=BIGGER_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

df = pd.read_csv("results_summary_gaussians.csv")

relabeling_map = {
    "mr": "MC",
    "lot": "LOT",
    "frlc": "FRLC"
}

palette = sns.color_palette("deep")

algorithm_colors = {
    "MC": palette[0],    # first color in deep palette
    "LOT": palette[1],   # second color in deep palette
    "FRLC": palette[2]   # third color in deep palette
}

hue_order = ["MC", "LOT", "FRLC"]

# Set the palette for the algorithms
#df = df[df['instance'] == 'n5000_k250_sigma0.3_perturb0.1']
#df = df[df['algorithm'] != 'frlc']

df['algorithm'] = df['algorithm'].map(relabeling_map)
df['rank'] = df['rank'].str[1:].astype(int)

monge_values = df[df['algorithm'] == 'MC'].pivot_table(
    index=['rank', 'seed', 'instance'], values='objective_cost'
)

df_with_monge = df.merge(
    monge_values, 
    left_on=['rank', 'seed', 'instance'], 
    right_index=True, 
    suffixes=('', '_monge')
)

df_with_monge['cost_ratio'] = df_with_monge['objective_cost'] / df_with_monge['objective_cost_monge']

instances = [ 
    (0.1, 'n5000_k250_sigma0.1_perturb0.1'),
    (0.2, 'n5000_k250_sigma0.2_perturb0.1'),
    (0.3, 'n5000_k250_sigma0.3_perturb0.1')
]

fig, axes = plt.subplots(figsize=(9, 3.5), nrows=1, ncols=3)
for i, (sigma, instance) in enumerate(instances):
    sns.boxplot(data=df_with_monge[(df['instance'] == instance)], x='rank', y='cost_ratio', hue='algorithm', ax=axes[i],
                palette=algorithm_colors, hue_order=hue_order)

    if i == 0:
        axes[i].set_ylabel(r'Cost Ratio: $\langle C, P^* \rangle_F / \langle C, P_{MC}^* \rangle_F$')
    else:
        axes[i].set_ylabel(None)
    axes[i].set_xlabel('Rank')
    axes[i].legend(title='Algorithm', loc='upper left')
    if i != 0:
        axes[i].get_legend().remove()
    axes[i].set_title(f'Shifted Gaussians ($\\sigma^2 = {sigma}$)')
    axes[i].axhline(y=1.0, linestyle='--', color='gray', alpha=0.7)  # Reference line at ratio=1

fig.savefig("figures/shifted_gaussians_cost_ratio.pdf")
plt.tight_layout()

fig, axes = plt.subplots(figsize=(9, 3.5), nrows=1, ncols=3)

for i, (sigma, instance) in enumerate(instances):
    rank_df = df[(df['rank'] == 250) & (df['instance'] == instance)]
    
    metrics_df = pd.melt(
        rank_df, 
        id_vars=['algorithm', 'seed', 'instance'], 
        value_vars=['X_ari', 'X_ami', 'Y_ari', 'Y_ami'],
        var_name='metric',
        value_name='score'
    )
    
    metrics_df['metric'] = metrics_df['metric'].map({
        'X_ari': 'ARI ($\\mathsf{X}$)',
        'X_ami': 'AMI ($\\mathsf{X}$)',
        'Y_ari': 'ARI ($\\mathsf{Y}$)',
        'Y_ami': 'AMI ($\\mathsf{Y}$)'
    })
    
    sns.boxplot(data=metrics_df, x='metric', y='score', hue='algorithm', ax=axes[i],
                palette=algorithm_colors, hue_order=hue_order)
    
    axes[i].set_ylabel('Value' if i == 0 else '')
    axes[i].set_xlabel('Metric')
    axes[i].legend(title='Algorithm', loc='lower left')
    if i != 0:
        axes[i].get_legend().remove()
    axes[i].set_title(f'Shifted Gaussians ($\\sigma^2 = {sigma}$)')

fig.savefig("figures/shifted_gaussians_clustering.pdf")
plt.tight_layout()
#plt.show()

df = pd.read_csv("results_summary_cliques.csv")

df['algorithm'] = df['algorithm'].map(relabeling_map)
df['rank'] = df['rank'].str[1:].astype(int)

monge_values = df[df['algorithm'] == 'MC'].pivot_table(
    index=['rank', 'seed', 'instance'], values='objective_cost'
)

df_with_monge = df.merge(
    monge_values, 
    left_on=['rank', 'seed', 'instance'], 
    right_index=True, 
    suffixes=('', '_monge')
)

df_with_monge['cost_ratio'] = df_with_monge['objective_cost'] / df_with_monge['objective_cost_monge']
print(df_with_monge)

fig, ax = plt.subplots(figsize=(4, 4))
sns.boxplot(data=df_with_monge, x='rank', y='cost_ratio', hue='algorithm', ax=ax, palette=algorithm_colors, hue_order=hue_order)
ax.set_ylabel(r'Cost Ratio: $\langle C, P^* \rangle_F / \langle C, P_{MC}^* \rangle_F$')
ax.set_xlabel('Rank')
ax.legend(title='Algorithm', loc='upper left')
ax.set_title(f'Stochastic Block Model')
ax.axhline(y=1.0, linestyle='--', color='gray', alpha=0.7)  # Reference line at ratio=1
plt.tight_layout()
fig.savefig("figures/sbm_cost_ratio.pdf")

fig, ax = plt.subplots(figsize=(4, 4))

metrics_df = pd.melt(
    df, 
    id_vars=['algorithm', 'seed', 'instance', 'rank'], 
    value_vars=['X_ari', 'X_ami', 'Y_ari', 'Y_ami'],
    var_name='metric',
    value_name='score'
)

metrics_df['metric'] = metrics_df['metric'].map({
    'X_ari': 'ARI ($\\mathsf{X}$)',
    'X_ami': 'AMI ($\\mathsf{X}$)',
    'Y_ari': 'ARI ($\\mathsf{Y}$)',
    'Y_ami': 'AMI ($\\mathsf{Y}$)'
})

# Filter for a specific rank if needed
# metrics_df = metrics_df[metrics_df['rank'] == 250]

sns.boxplot(data=metrics_df, x='metric', y='score', hue='algorithm', ax=ax, palette=algorithm_colors, hue_order=hue_order)
ax.set_ylabel('Value')
ax.set_xlabel('Metric')
ax.legend(title='Algorithm', loc='upper left')
ax.set_title('Stochastic Block Model')

plt.tight_layout()
fig.savefig("figures/sbm_clustering.pdf")
#plt.show()

df = pd.read_csv("results_summary_moons.csv")
instances = [(0.1, 'n5000_s1_noise0.1'), (0.25, 'n5000_s1_noise0.25'), (0.5, 'n5000_s1_noise0.5')]

df['algorithm'] = df['algorithm'].map(relabeling_map)
df['rank'] = df['rank'].str[1:].astype(int)

monge_values = df[df['algorithm'] == 'MC'].pivot_table(
    index=['rank', 'seed', 'instance'], values='objective_cost'
)

df_with_monge = df.merge(
    monge_values, 
    left_on=['rank', 'seed', 'instance'], 
    right_index=True, 
    suffixes=('', '_monge')
)

df_with_monge['cost_ratio'] = df_with_monge['objective_cost'] / df_with_monge['objective_cost_monge']

fig, axes = plt.subplots(figsize=(9, 3.5), nrows=1, ncols=3)
for i, (sigma, instance) in enumerate(instances):
    sns.boxplot(data=df_with_monge[(df['instance'] == instance)], x='rank', y='cost_ratio', hue='algorithm', ax=axes[i],
                palette=algorithm_colors, hue_order=hue_order)

    if i == 0:
        axes[i].set_ylabel(r'Cost Ratio: $\langle C, P^* \rangle_F / \langle C, P_{MC}^* \rangle_F$')
    else:
        axes[i].set_ylabel(None)
    axes[i].set_xlabel('Rank')
    axes[i].legend(title='Algorithm', loc='upper left')
    if i != 0:
        axes[i].get_legend().remove()
    axes[i].set_title(f'2 Moons, 8 Gaussians ($\\sigma^2 = {sigma}$)')
    axes[i].axhline(y=1.0, linestyle='--', color='gray', alpha=0.7)  # Reference line at ratio=1

plt.tight_layout()
fig.savefig("figures/two_moons_cost_ratio.pdf")
plt.show()