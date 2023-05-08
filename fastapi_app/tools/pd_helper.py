
def pd_first_row_element(df, col):
    if len(df.index) == 1:
        return df.iloc[0][col]
    else:
        raise TypeError("df should have only one row, not {}".format(len(df.index)))