import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

df = pd.read_csv('./evaluation/results.csv')

ati_cols = ['ati_1', 'ati_2', 'ati_3', 'ati_4', 'ati_5', 'ati_6', 'ati_7', 'ati_8', 'ati_9']
ati_scores = df[ati_cols].sum(axis=1) / len(ati_cols)
df['ati_score'] = ati_scores

sipa_pre_cols = ['sipa_pre_1', 'sipa_pre_2', 'sipa_pre_3', 'sipa_pre_4', 'sipa_pre_5', 'sipa_pre_6']
sipa_pre_scores = df[sipa_pre_cols].sum(axis=1) / len(sipa_pre_cols)
df['sipa_pre_score'] = sipa_pre_scores

fost_pre_cols = ['fost_pre_1', 'fost_pre_2', 'fost_pre_3', 'fost_pre_4', 'fost_pre_5']
fost_pre_scores = df[fost_pre_cols].sum(axis=1) / len(fost_pre_cols)
df['fost_pre_score'] = fost_pre_scores

sipa_post_cols = ['sipa_post_1', 'sipa_post_2', 'sipa_post_3', 'sipa_post_4', 'sipa_post_5', 'sipa_post_6']
sipa_post_scores = df[sipa_post_cols].sum(axis=1) / len(sipa_post_cols)
df['sipa_post_score'] = sipa_post_scores

fost_post_cols = ['fost_post_1', 'fost_post_2', 'fost_post_3', 'fost_post_4', 'fost_post_5']
fost_post_scores = df[fost_post_cols].sum(axis=1) / len(fost_post_cols)
df['fost_post_score'] = fost_post_scores

ess_cols = ['ess_1', 'ess_2', 'ess_3', 'ess_4', 'ess_5', 'ess_6', 'ess_7', 'ess_8']
ess_scores = df[ess_cols].sum(axis=1) / len(ess_cols)
df['ess_score'] = ess_scores

nasa_tlx_cols = ['nasa-tlx_1', 'nasa-tlx_2', 'nasa-tlx_3', 'nasa-tlx_4', 'nasa-tlx_5']
nasa_tlx_scores = df[nasa_tlx_cols].sum(axis=1) / len(nasa_tlx_cols)
df['nasa_tlx_score'] = nasa_tlx_scores

sipa_diff_scores = df['sipa_pre_score'] - df['sipa_post_score']
df['sipa_diff_score'] = sipa_diff_scores

df.to_csv('./evaluation/results_with_means.csv')

df_small = df[['age',
               'semester',
               'guidance',
               'ati_score',
               'sipa_pre_score',
               'fost_pre_score',
               'sipa_post_score',
               'fost_post_score',
               'ess_score',
               'nasa_tlx_score']]
print('All participants:')
print(df_small.describe().to_string())

g_df = df_small[df_small['guidance'] == True]
print('Guided participants:')
print(g_df.describe().to_string())

u_df = df_small[df_small['guidance'] == False]
print('Unguided participants:')
print(u_df.describe().to_string())

ati_df = pd.concat([df_small['ati_score'], g_df['ati_score'], u_df['ati_score']], axis=1,
                   keys=['all', 'guided', 'unguided'])
sipa_pre_df = pd.concat([df_small['sipa_pre_score'], g_df['sipa_pre_score'], u_df['sipa_pre_score']], axis=1,
                        keys=['all', 'guided', 'unguided'])
fost_pre_df = pd.concat([df_small['fost_pre_score'], g_df['fost_pre_score'], u_df['fost_pre_score']], axis=1,
                        keys=['all', 'guided', 'unguided'])
sipa_post_df = pd.concat([df_small['sipa_post_score'], g_df['sipa_post_score'], u_df['sipa_post_score']], axis=1,
                         keys=['all', 'guided', 'unguided'])
fost_post_df = pd.concat([df_small['fost_post_score'], g_df['fost_post_score'], u_df['fost_post_score']], axis=1,
                         keys=['all', 'guided', 'unguided'])
ess_df = pd.concat([df_small['ess_score'], g_df['ess_score'], u_df['ess_score']], axis=1,
                   keys=['all', 'guided', 'unguided'])
nasa_tlx_df = pd.concat([df_small['nasa_tlx_score'], g_df['nasa_tlx_score'], u_df['nasa_tlx_score']], axis=1,
                        keys=['all', 'guided', 'unguided'])

sns.set(style='whitegrid', rc={'figure.figsize': (10, 20)})
fig, axes = plt.subplots(5, 2, sharex=True, sharey='row')

axes[0, 0].set_title('SIPA Pre')
sns.boxplot(ax=axes[0, 0], data=sipa_pre_df)

axes[0, 1].set_title('SIPA Post')
sns.boxplot(ax=axes[0, 1], data=sipa_post_df)

axes[1, 0].set_title('FOST Pre')
sns.boxplot(ax=axes[1, 0], data=fost_pre_df)

axes[1, 1].set_title('FOST Post')
sns.boxplot(ax=axes[1, 1], data=fost_post_df)

axes[2, 1].set_title('ESS Post')
sns.boxplot(ax=axes[2, 1], data=ess_df)

axes[3, 1].set_title('NASA-TLX Post')
sns.boxplot(ax=axes[3, 1], data=nasa_tlx_df)

axes[4, 0].set_title('ATI Pre')
sns.boxplot(ax=axes[4, 0], data=ati_df)

# sns.boxplot(data=sipa_post_df)
plt.savefig('./evaluation/boxplots.png')
