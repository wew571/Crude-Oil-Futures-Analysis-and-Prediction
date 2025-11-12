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

    # -----Giải thích đoạn numeric_dataframe -----
    # Lấy các cột của numeric_dataframe bằng cách lọc 1 dataframe ban đầu sao cho
    # chỉ lấy những cột chứa số
    #
    # Ví dụ :
    #
    #     'A': [1, 2, 3],         # int
    #     'B': [1.5, 2.5, 3.5],   # float
    #     'C': ['x', 'y', 'z'],   # object
    #     'D': [True, False, True] # bool
    #
    #     Sau khi chạy numeric_dataframe = dataframe.select_dtypes(include = [numpy.number])
    #     xong thì biến numeric_dataframe này chỉ còn
    #
    #        A    B
    #     0  1  1.5
    #     1  2  2.5
    #     2  3  3.5

    numeric_dataframe = dataframe.select_dtypes(include = [numpy.number])
    columns = numeric_dataframe.columns
    n = len(columns)

    #Tạo ma trận kết quả có đường chéo chính bằng 1
    corr_matrix = numpy.eye(n)
    corr_matrix = pandas.DataFrame(corr_matrix , index=columns , columns=columns)

    for i in range(n):
        for j in range(i + 1 , n):
            columns1 = numeric_dataframe.iloc[:,i]
            columns2 = numeric_dataframe.iloc[:,j]

            # -----Giải thích biến mask -----
            # Sử dụng mask để lọc thông tin . Nếu k có thì trong lúc tính
            # ma trận tương quan ( Correlation Matrix ) mà có 1 giá trị là NaN
            # thì nó sẽ tính cả cái ma trận đấy toàn NaN
            #
            # Ví dụ :
            #     Dataframe gốc :
            #         'A': [1, 2, np.nan],
            #         'B': [2, np.nan, 6],
            #         'C': [3, 4, 9]
            #
            #     Khi không sử dụng mask :
            #             A   B   C
            #         A NaN NaN NaN
            #         B NaN NaN NaN
            #         C NaN NaN NaN
            #
            #     khi sử dụng mask :
            #             A    B    C
            #         A  1.0  1.0  1.0
            #         B  1.0  1.0  1.0
            #         C  1.0  1.0  1.0
            #
            # Diễn giải chi tiết cách hoạt động :
            #     Giả sử có 1 dataframe :
            #
            #         'columns1': [1, 2, np.nan],
            #         'columns2': [2, np.nan, 6],
            #
            #     columns1.isna() sẽ có : False , False , True
            #     columns2.isna() sẽ có : False , True , False
            #
            #     Kết hợp điều kiện vào :
            #     (columns1.isna() | columns2.isna()) sẽ là : False , True , True
            #
            #     ~ là dấu của toán tử NOT
            #
            #     Cuối cùng thì mask sẽ có giá trị : True , False , False

            mask = ~(columns1.isna() | columns2.isna())
            x = columns1[mask]
            y = columns2[mask]

            if len(x) < min_periods:
                value = numpy.nan
            else:
                if method == 'pearson':
                    value = calculate_pearson(x , y)
                elif method == 'spearman':
                    value = calculate_spearman(x, y)
                elif method == 'kendall':
                    value = calculate_kendall(x , y)
                else:
                    value = numpy.nan

            corr_matrix.iloc[i , j] = value
            corr_matrix.iloc[j , i] = value

    return corr_matrix

def calculate_pearson( x : pandas.Series , y : pandas.Series)->float:
    """
    Tính toán Pearson

    Parameters:
        x : cột x dùng để tính
        y : cột y dùng để tính

    Returns:
        value : float

    Công thức toán học :

                Σ[(xᵢ - x̄)(yᵢ - ȳ)]
        r = --------------------------
            √[Σ(xᵢ - x̄)² * Σ(yᵢ - ȳ)²]

            Σ trong này là từ i = 1 tới n
            x̄ : giá trị trung bình của x
            ȳ : giá trị trung bình của y
    """

    x_mean = x.mean()
    y_mean = y.mean()

    numerator = ((x - x_mean) * (y - y_mean)).sum()
    denominator = numpy.sqrt( ((x - x_mean)**2).sum() * ((y - y_mean)**2).sum() )

    if denominator == 0:
        return numpy.nan

    value = numerator / denominator
    return value

def calculate_spearman( x : pandas.Series , y : pandas.Series)->float:
    """
    Tính Spearman correlation (dựa trên rank)

    Parameters:
        x : cột x dùng để tính
        y : cột y dùng để tính

    Returns:
        cacalculate_pearson ( x_rank, y_rank ) : float
    """
    x_rank = x.rank()
    y_rank = y.rank()

    return calculate_pearson(x_rank, y_rank)

def calculate_kendall( x : pandas.Series , y : pandas.Series)->float:
    """
    Tính Kendall's tau correlation

    Parameters:
        x : cột x dùng để tính
        y : cột y dùng để tính

    Returns:
        value : float

    Công thức toán học :

                C - D
        r = ------------
                C + D

        C = số concordant pairs (cặp đồng thuận, cùng tăng hoặc cùng giảm)
        D = số discordant pairs (cặp nghịch thuận, một tăng một giảm)
        r là giá trị nằm trong khoảng [-1 , 1 ]
    """

    n = len(x)
    if n < 2:
        return numpy.nan

    concordant = 0
    discordant = 0

    for i in range(n):
        for j in range(i+1 , n):
            x_difference = x.iloc[i] - x.iloc[j]
            y_difference = y.iloc[i] - y.iloc[j]

            if x_difference * y_difference > 0:
                concordant += 1
            elif x_difference * y_difference < 0:
                discordant += 1

        # Nếu C + D = 0 thì sẽ tạo ra 1 số k xác định
        # Như công thức sẽ là
        #
        #           C - D
        #   r = ------------
        #            0
    if concordant + discordant == 0 :
        return numpy.nan

    value = ( concordant - discordant ) / (concordant + discordant)
    return value
