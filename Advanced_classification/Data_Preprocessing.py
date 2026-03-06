# -*- coding : utf-8 -*-
import numpy as np

def Standarize(X):
    """標準化

    Args:
        X (ベクトル): 入力データ
        
    Returns:
        X (ベクトル): 標準化後の入力データ
    """
    return (X - np.mean(X)) / np.std(X)

def L2Regularization(w, a):
    """L2正則化<br>
    実用を考えるならば、モデルの中に組み込む必要がある.<br>
    今回は、関数として書いてみただけ.
    
    Args:
        X (ベクトル): 重みベクトル
        a (float): 正則化係数
        
    Returns:
        w (ベクトル): 正規化後の重みベクトル
    """
    return (a / len(w)) * sum(w ** 2)