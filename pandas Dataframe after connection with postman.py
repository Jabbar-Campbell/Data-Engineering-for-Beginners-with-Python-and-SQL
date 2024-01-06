# %%
pd.DataFrame(x)

# %%
df = pd.DataFrame(x)

# %%
df.head()

# %%


# iterating the columns
for col in df.columns:
    print(col)

# %%
df['docs'].str.split(',')

# %%
df['docs']

# %%
df['docs'].str.split(':')

# %%
df['docs'][1]

# %%
df['docs'][2] 

# %%
df['docs'][3] 

# %%
df['docs'][1] 

# %%
df.iloc[1: , :]

# %%
df

# %%
df['docs']

# %%
type(df['docs'][1])

# %%
df['docs'].apply(pd.Series)

# %%
df=df['docs'].apply(pd.Series)

# %%
df.drop('type', axis=1)

# %%
for col in df.columns:
    print(col)

# %%
response.text


