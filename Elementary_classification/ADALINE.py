# -*- coding: utf-8 -*-
import numpy as np

class ADALINE:
    """
    # パラメータ
    **X**       :入力データ<br>
    **y**       :正解データ<br>
    **alpha**   :学習率<br>
    **n_iters** :行列の重み更新の繰り返し回数<br>
    **weights** :重みベクトル(必須ではない)<br>
    **bias**    :バイアス(必須ではない)
    """
    def __init__(self, X, y, alpha=0.01, n_iters=1000, weights=None, bias=None):
        self.X_         = X
        self.y_         = y
        self.alpha_     = alpha
        self.n_iters_   = n_iters
        self.weights_   = weights
        self.bias_      = bias
        
    def fit(self, get_accuracy=False):
        """
        ADALINEモデルを入力データに適合させる
        """
        row, col = self.X_.shape        # row:(データ数=サンプル数)、col:列数(特徴量の数=次元数)
        
        # パラメータ(重みとバイアス)の初期化
        if self.weights_ is None:
            self.weights_ = np.zeros(col)   # 注意：数式では重みベクトルは1行col列として考えているが、プログラム内部ではcol行1列となっている
        if self.bias_    is None:
            self.bias_    = 0
                    
        # 重みとバイアスを求める
        for _ in range(self.n_iters_):  # 行列全体の計算をn_iters回繰り返す
            
            z = self.net_input(self.X_)                                     # 総入力を計算しておく
            nabla_wL, nabla_bL = self.nabla_objective(z)  # 目的関数(損失関数)の偏微分値を計算
            
            delta_weights = - self.alpha_ * nabla_wL                        # 重みの更新量
            delta_bias    = - self.alpha_ * nabla_bL                        # バイアスの更新量
            self.weights_ += delta_weights                                  # 重みを更新する
            self.bias_    += delta_bias                                     # バイアスを更新する
                        
        if get_accuracy:
            return self.accuracy(self.y_, self.predict(self.X_))
        
    def predict(self, X):
        """総入力zを活性化関数に通して、決定関数を用いてクラスラベル(0か1)を予測する
        """
            
        z = self.net_input(X)
        activated_z = self.activation(z)
        y_pred = self.threshold(activated_z)
        
        return y_pred
    
    def net_input(self, X):
        """総入力zを計算する
        """
        return np.dot(X, self.weights_) + self.bias_
    
    def threshold(self, z):
        """決定関数<br>
        与えられた総入力zを基に、0か1を返す
        """
        #return 1 if z >= 0 else 0      # accuracy実行時に行列が全て代入されるため、その場合はzは1次元の配列になり、このコードでは対応できない.
        return np.where(z >= 0.5, 1, 0)   # 行列にも対応したコード
    
    def activation(self, z):
        """活性化関数<br>
        与えられた総入力zを基に、値を返す
        """
        return z
    
    def nabla_objective(self, z):
        """目的関数の偏微分<br>
        ADALINEの目的関数は損失関数<br>
        損失関数の偏微分値を返す
        数式上は重みベクトルとバイアスを代入するが、このプログラムでは総入力を入れる
        """
        error = self.y_ - self.activation(z)                    # i行1列
        nabla_wL = -(2 / len(self.y_)) * np.dot(error.T, self.X_) # 1行i列 × i行j列 = 1行j列
        nabla_bL = -(2 / len(self.y_)) * sum(error)
        
        return nabla_wL, nabla_bL
        
    
    def accuracy(self, y_true, y_pred):
        """正解率を計算する
        """
        return sum(y_true == y_pred) / len(y_true)
    
    def plot_dicision_boundary(self, resolution=0.01):
        """決定境界をプロットする<br>
        resolution:決定境界をプロットするためのグリッドの解像度(グリッドの点と点の距離)
        """
        import matplotlib.pyplot as plt
        
        # 決定境界をプロットするためのグリッドを作成
        x_min, x_max = self.X_[:, 0].min() - 1, self.X_[:, 0].max() + 1
        y_min, y_max = self.X_[:, 1].min() - 1, self.X_[:, 1].max() + 1
        
        # 表示範囲を設定
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        
        # x_minとx_maxの地点で、予測値が変わるyの値を求める(2分探索)
        y_border_upper_x_min = y_border_upper_x_max = y_max
        y_border_lower_x_min = y_border_lower_x_max = y_min
        y_middle_x_min = y_middle_x_max = (y_max + y_min) / 2
        for _ in range(int(1/resolution)):
            y_pred_x_min = self.predict([x_min, y_middle_x_min])
            y_pred_x_max = self.predict([x_max, y_middle_x_max])
            
            if y_pred_x_min == 0:
                y_border_lower_x_min = y_middle_x_min
            else:
                y_border_upper_x_min = y_middle_x_min
                
            if y_pred_x_max == 0:
                y_border_lower_x_max = y_middle_x_max
            else:
                y_border_upper_x_max = y_middle_x_max
                
            y_middle_x_min = (y_border_upper_x_min + y_border_lower_x_min) / 2
            y_middle_x_max = (y_border_upper_x_max + y_border_lower_x_max) / 2
            
        # x_minとx_maxの地点で、予測値が変わるyの値を結ぶ直線を決定境界とする
        slope = (y_border_upper_x_max - y_border_upper_x_min) / (x_max - x_min)
        intercept = y_border_upper_x_min - slope * x_min
        x = np.array([x_min-1, x_max+1])
        y = slope * x + intercept
    
        # 決定境界近似曲線をプロット、境界で色分け
        plt.plot(x, y, color='black', linestyle='-', label=f'決定境界近似')
        plt.fill_between(x, y, y_min-1, color='blue', alpha=0.2, label='Class 0 領域')
        plt.fill_between(x, y, y_max+1, color='red' , alpha=0.2, label='Class 1 領域')
        
        # 正解値が0と1のデータ点を異なるマーカー、予測値が0と1のデータ点を異なる色でプロット
        cls0_collect = self.X_[np.logical_and(self.predict(self.X_) == 0, self.y_ == 0)]
        cls1_collect = self.X_[np.logical_and(self.predict(self.X_) == 1, self.y_ == 1)]
        cls1_error   = self.X_[np.logical_and(self.predict(self.X_) == 1, self.y_ == 0)]
        cls0_error   = self.X_[np.logical_and(self.predict(self.X_) == 0, self.y_ == 1)]
        plt.scatter(cls0_collect[:, 0], cls0_collect[:, 1], c='blue', edgecolors='k', marker='o', label='Class 0 (予測成功)')
        plt.scatter(cls1_collect[:, 0], cls1_collect[:, 1], c='red' , edgecolors='k', marker='s', label='Class 1 (予測成功)')
        plt.scatter(cls1_error[:, 0]  , cls1_error[:, 1]  , c='cyan' , edgecolors='k', marker='o', label='Class 0 (予測失敗)')
        plt.scatter(cls0_error[:, 0]  , cls0_error[:, 1]  , c='orange', edgecolors='k', marker='s', label='Class 1 (予測失敗)')
                
        # 軸ラベル、タイトル、凡例の設定
        plt.xlabel(xlabel='sepal length (cm)', fontdict={'family':'MS Gothic'})
        plt.ylabel(ylabel='petal length (cm)', fontdict={'family':'MS Gothic'})
        plt.title(label='ADALINEの決定領域と予測結果' , fontdict={'family':'MS Gothic'})
        plt.legend(prop={'family':'MS Gothic', 'size':7}, loc='lower right', )
        
        plt.show()
        
        
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
    
    # ADALINEモデルを適用できるようにするために、データの構造を変更する
    X = X[:100, [0, 2]]    # 入力データXを、sepalとpetalの長さとする(targetは3つあり、1つあたり50行ずつあるため、150行のうち前100行のみを選択)
    y = y[:100]            # 正解データyを、setosaとversicolorの2クラスにする(0と1の2値分類)
    
    # データ変更後のデータ確認
    df = DataFrame(X, columns=["sepal length", "petal length"])
    df["target"] = y
    print(df.head())
    print(df.tail())
    
    # ADALINEモデルを学習
    model = ADALINE(X, y, alpha=0.01, n_iters=1000)
    acc = model.fit(get_accuracy=True)
    print(f"Accuracy: {acc}")
    
    # 決定境界のプロット
    model.plot_dicision_boundary(resolution=0.01)
    
    print(model.weights_)