import pandas as pd

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

print('All participants:')
print(df[['age',
          'semester',
          'ati_score',
          'sipa_pre_score',
          'fost_pre_score',
          'sipa_post_score',
          'fost_post_score',
          'ess_score',
          'nasa_tlx_score']].describe().to_string())

g_df = df[df['guidance'] == True]
print('Guided participants:')
print(g_df[['age',
            'semester',
            'ati_score',
            'sipa_pre_score',
            'fost_pre_score',
            'sipa_post_score',
            'fost_post_score',
            'ess_score',
            'nasa_tlx_score']].describe().to_string())

u_df = df[df['guidance'] == False]
print('Unguided participants:')
print(u_df[['age',
            'semester',
            'ati_score',
            'sipa_pre_score',
            'fost_pre_score',
            'sipa_post_score',
            'fost_post_score',
            'ess_score',
            'nasa_tlx_score']].describe().to_string())
