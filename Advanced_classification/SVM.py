# -*- coding : utf-8 -*-
import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris

# Irisデータセットの読み込み
X, y = load_iris(return_X_y=True)

# データの確認
df = pd.DataFrame(X, columns=load_iris().feature_names)
df["target"] = y
print(df.head())
print(df.tail())

# データの構造を変更する
X = X[:100, [0, 2]]    # 入力データXを、sepalとpetalの長さとする(targetは3つあり、1つあたり50行ずつあるため、150行のうち前100行のみを選択)
y = y[:100]            # 正解データyを、setosaとversicolorの2クラスにする(0と1の2値分類)

# データ変更後のデータ確認
df = pd.DataFrame(X, columns=["sepal length", "petal length"])
df["target"] = y
print(df.head())
print(df.tail())

# SVMモデルを学習
model = SVC(kernel="linear", C=1.0)
model.fit(X, y)

# 決定境界のプロット
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 200),
    np.linspace(y_min, y_max, 200)
)
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.coolwarm)
plt.xlabel("sepal length")
plt.ylabel("petal length")
plt.title("SVMの決定境界", fontdict={'family':'MS Gothic'})
plt.show()