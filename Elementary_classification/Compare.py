# -*- coding:utf-8 -*-

import numpy as np
import Perceptron
import ADALINE

if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from pandas import DataFrame
    
    # Irisデータセットの読み込み
    X, y = load_iris(return_X_y=True)
    
    # データの確認
    df = DataFrame(X, columns=load_iris().feature_names)
    df["target"] = y
    print(df.head())
    print(df.tail())
    
    # データの構造を変更する
    X = X[:100, [0, 2]]    # 入力データXを、sepalとpetalの長さとする(targetは3つあり、1つあたり50行ずつあるため、150行のうち前100行のみを選択)
    y = y[:100]            # 正解データyを、setosaとversicolorの2クラスにする(0と1の2値分類)
    
    # 外れ値として一行を追加（行方向=axis=0）
    X = np.append(X, [[5.5, -1]], axis=0)
    print(X)
    
    # データ変更後のデータ確認
    # 注意: outlier を追加したので y の長さも合わせて延長する
    df = DataFrame(X, columns=["sepal length", "petal length"])
    # X に新しい行が加わったため y にダミーのラベルを足す（ここでは -1 を使用）
    y = np.append(y, 1)
    df["target"] = y
    print(df.head())
    print(df.tail())
    
    # パーセプトロンモデルを学習
    for i in range(3):
        model = Perceptron.Perceptron(X, y, alpha=0.01, n_iters=999+i)
        acc = model.fit(get_accuracy=True)
        print(f"Accuracy: {acc}")
        model.plot_dicision_boundary(resolution=0.01)
    
    # ADALINEモデルを学習
    for i in range(3):
        model = ADALINE.ADALINE(X, y, alpha=0.01, n_iters=999+i)
        acc = model.fit(get_accuracy=True)
        print(f"Accuracy: {acc}")
        model.plot_dicision_boundary(resolution=0.01)