# %%
import pandas as pd
# %%
candy = pd.DataFrame(index=['1994', '1996'])
candy['prior'] = 1/2, 1/2
candy['likelihood'] = 0.2 * 0.2, 0.14 * 0.1
candy['unnorm'] = candy['prior'] * candy['likelihood']
prob_norm = candy.unnorm.sum()
candy['posterior'] = candy['unnorm'] / prob_norm
candy.head()
# %%
