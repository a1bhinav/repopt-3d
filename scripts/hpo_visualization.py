#%%
import pandas as pd
import matplotlib.pyplot as plt

#%%
# df = pd.read_csv('../third_party/openscene/out/hpo_fusion_seed3407/ce/hpo_trials.csv')
df = pd.read_csv('../third_party/openscene/out/hpo_fusion_seed3407/ce_h1/hpo_trials.csv')

# Identify the latest appended run block: from the last trial_idx == 0 to the end.
reset_points = df.index[df['trial_idx'] == 0]
latest_run_start = reset_points[-1] if len(reset_points) > 0 else 0
is_new_run = df.index >= latest_run_start

#%%
df.head()

#%%
df.columns
# %%
hyperparameters_to_vis = ['base_lr', 'batch_size', 'weight_decay', 'epochs',
'h1_lambda', 'h1_k', 'h1_sigma'
]

# %%
for param in hyperparameters_to_vis:
    plt.figure(figsize=(10, 5))
    plt.scatter(
        df.loc[~is_new_run, param],
        df.loc[~is_new_run, 'val_miou'],
        color='C0',
        label='old runs'
    )
    plt.scatter(
        df.loc[is_new_run, param],
        df.loc[is_new_run, 'val_miou'],
        color='orange',
        label='new run'
    )
    plt.title(f'{param} vs. val_miou')
    plt.xlabel(param)
    plt.ylabel('val_miou')
    plt.legend()
    plt.show()
# %%