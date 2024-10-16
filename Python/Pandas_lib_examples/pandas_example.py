import pandas as pd

df_titanic = pd.read_csv('train.csv')

def create_df():
    test_df = pd.DataFrame.from_dict({'id':[0,1], 'product':["tomato", "cucmber"]})
    print(test_df)

def select_first_last():
    print(df_titanic.head(5), df_titanic.tail(3))

def select_series():
    print(df_titanic['Name'])

def select_multiple():
    print(df_titanic[['Name', 'Age']].tail(10))

def select_by_iloc():
    print(df_titanic.iloc[1:3, :5])

def select_with_if():
    print(df_titanic[df_titanic['Age'] > 20 | df_titanic['Age'].isin([5,10])])
    #print(df_titanic.loc[df_titanic['Age'].notna(), 'Name'])

def sort():
    print(df_titanic.sort_values('Pclass').head(10))

def concat():
    df2 = df_titanic.copy()
    dfc = pd.concat([df_titanic, df2])
    print(dfc.shape)

def stats():
    print(df_titanic['Age'].describe())

def stats_groupedby():
    #print(df_titanic.groupby('Sex')['Age'].mean())
    print(df_titanic['Sex'].value_counts())
    print(df_titanic.groupby(['Sex', 'Survived'])['Age'].mean())

def _plot():
    df_titanic['Age'].plot(kind='kde')
    #lib required

_plot()