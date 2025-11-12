from typing import  Literal

import numpy
import pandas

def corr(dataframe : pandas.DataFrame ,
         method : Literal['pearson' , 'spearman' , 'kendall'] ,
         min_periods: int = 1) -> pandas.DataFrame:

    """
    Tự implement hàm correlation tương tự pandas corr()

    Parameters:
    - dataframe: DataFrame cần tính correlation
    - method: Phương pháp tính correlation
    - min_periods: Số lượng quan sát tối thiểu

    """

    """
    Lấy các cột của numeric_dataframe bằng cách lọc 1 dataframe ban đầu sao cho
    chỉ lấy những cột chứa số 
    
    Ví dụ :
    
        'A': [1, 2, 3],         # int
        'B': [1.5, 2.5, 3.5],   # float
        'C': ['x', 'y', 'z'],   # object
        'D': [True, False, True] # bool
        
        Sau khi chạy numeric_dataframe = dataframe.select_dtypes(include = [numpy.number])
        xong thì biến numeric_dataframe này chỉ còn 
    
           A    B
        0  1  1.5
        1  2  2.5
        2  3  3.5
        
    """
    numeric_dataframe = dataframe.select_dtypes(include = [numpy.number])
    columns = numeric_dataframe.columns
    n = len(columns)

