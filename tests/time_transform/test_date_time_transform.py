import pandas as pd
from lazyauto.time_transform.date_time_transform import datetime_ohe, datetime_simple

start_date = pd.to_datetime("2023-03-08")
end_date = pd.to_datetime("2023-03-15")

dates = pd.date_range(start_date, end_date)
df = pd.DataFrame(dates, columns=["Date"])

def test_datetime_ohe():
    a= datetime_ohe(df['Date'])
    pd.isnull(a).sum()
    assert any(pd.isnull(a).sum() == 0)
    
def test_datetime_simple():
    a= datetime_simple(df['Date'])
    pd.isnull(a).sum()
    assert any(pd.isnull(a).sum() == 0)
    