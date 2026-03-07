# -*- coding: utf-8 -*-
import numpy as np

class LogisticRegression:
    """
    ADALINEをコピペして活性化関数と目的関数を変更
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
        ロジスティック回帰モデルを入力データに適合させる
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
            nabla_wL, nabla_bL = self.nabla_objective(z)    # 目的関数(損失関数)の偏微分値を計算
            
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
        今回はシグモイド関数
        """
        return 1 / (1 + np.exp(-z))
    
    def nabla_objective(self, z):
        """目的関数の偏微分<br>
        ロジスティック回帰の目的関数は損失関数(2値交差エントロピー)<br>
        損失関数の偏微分値を返す
        数式上は活性化済みの総入力を代入するが、このプログラムでは総入力を入れる
        """
        error = self.activation(z) - self.y_
        nabla_wL = np.dot(error, self.X_)
        nabla_bL = np.sum(error)
        
        return nabla_wL, nabla_bL
        
    
    def accuracy(self, y_true, y_pred):
        """正解率を計算する
        """
        return sum(y_true == y_pred) / len(y_true)
    
    def plot_decision_boundary(model, X, y):# バグがあったため、変更
        import matplotlib.pyplot as plt
        # グリッド作成
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200)
        )
        # 予測
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        # 図示
        plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
        plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.coolwarm)
        plt.xlabel("sepal length")
        plt.ylabel("petal length")
        plt.title("自作ロジスティック回帰の決定境界", fontdict={'family':'MS Gothic'})
        plt.show()
        
        
if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from pandas import DataFrame
    import Data_Adjustment
    
    # Irisデータセットの読み込み
    X, y = load_iris(return_X_y=True)
    
    # データの確認
    df = DataFrame(X, columns=load_iris().feature_names)
    df["target"] = y
    print(df.head())
    print(df.tail())
    
    # ロジスティック回帰モデルを適用できるようにするために、データの構造を変更する
    X = X[:-50, [0, 2]]                     # 入力データXを、sepalとpetalの長さとする(targetは3つあり、1つあたり50行ずつあるため、150行のうち前100行のみを選択)
    X = Data_Adjustment.Standardization(X)  # 標準化

    y = y[:-50]                             # 正解データyを、setosaとversicolorの2クラスにする(0と1の2値分類)
    
    # データ変更後のデータ確認
    df = DataFrame(X)
    df["target"] = y
    print(df.head())
    print(df.tail())
    
    # ロジスティック回帰モデルを学習
    model = LogisticRegression(X, y, alpha=0.01, n_iters=1000)
    acc = model.fit(get_accuracy=True)
    print(f"Accuracy: {acc}")
    
    # 決定境界のプロット
    model.plot_decision_boundary(X, y)
    
    print(model.weights_)