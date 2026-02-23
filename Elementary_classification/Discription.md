# 初めに
本ノート作成のきっかけは、アルゴリズムの解説や実装をできる限り順を追って詳細に行うことで、アルゴリズムの理解コストをできる限り下げられないかと考えたことです。
機械学習やディープラーニングについての概要をある程度理解した前提で書いたため、用語の理解を深めるような説明はあまり行われていません。
参考は、以下の2つの参考書や気づいたら知っていた情報、大学の講義で得た知識らへんです。
[Python機械学習プログラミング［PyTorch＆scikit-learn編］ (impress top gear) | Sebastian Raschka, uxi (Hayden) Liu, Vahid Mirjalili, 株式会社クイープ, 福島真太朗 |本 | 通販 | Amazon](https://www.amazon.co.jp/Python%E6%A9%9F%E6%A2%B0%E5%AD%A6%E7%BF%92%E3%83%97%E3%83%AD%E3%82%B0%E3%83%A9%E3%83%9F%E3%83%B3%E3%82%B0%EF%BC%BBPyTorch%EF%BC%86scikit-learn%E7%B7%A8%EF%BC%BD-impress-gear-Sebastian-Raschka/dp/429501558X/ref=sr_1_6?__mk_ja_JP=%E3%82%AB%E3%82%BF%E3%82%AB%E3%83%8A&crid=I02ZDITXLGJF&dib=eyJ2IjoiMSJ9.fHzRiDTrbzW9G-jKqcR9kYAVdvedptfy5MmCZYIJaEsf0hzY3jTXWbBEnXNljrRr60cxstwDUo2kxYpAoALAOn2FjXqyR1y24T_smnKbm28pBYPtBsLbGeCwuGsdcuqqGpjFB2b4MhaISphyn6uubK5a1YTiQdowNFALKNQgPWmKP2Mq0PDzBHYVo3zF1yseMxrD7FWRCwcmDgEvoTC1z_Ff-evffgjlCva5Y1hHEwe8QzUb2CKtzCG_wVPQf7Y-jZNsOTndeHvUF3Djw_9il6UVxl7H3vuYPdE_wlBE4Js.vmOZBEGK5nR2OwtTj1el5p3rFLXiNFSBlqbnF-7flh8&dib_tag=se&keywords=Python%E6%A9%9F%E6%A2%B0%E5%AD%A6%E7%BF%92%E3%83%97%E3%83%AD%E3%82%B0%E3%83%A9%E3%83%9F%E3%83%B3%E3%82%B0&qid=1771732900&sprefix=python%E6%A9%9F%E6%A2%B0%E5%AD%A6%E7%BF%92%E3%83%97%E3%83%AD%E3%82%B0%E3%83%A9%E3%83%9F%E3%83%B3%E3%82%B0%2Caps%2C206&sr=8-6)
[深層学習教科書 ディープラーニング G検定（ジェネラリスト）公式テキスト 第3版 (EXAMPRESS) | 一般社団法人日本ディープラーニング協会, 山下 隆義, 猪狩 宇司, 今井 翔太, 巣籠 悠輔, 瀬谷 啓介, 徳田 有美子, 中澤 敏明, 藤本 敬介, 古川 直裕, 松尾 豊, 松嶋 達也 |本 | 通販 | Amazon](https://www.amazon.co.jp/%E6%B7%B1%E5%B1%A4%E5%AD%A6%E7%BF%92%E6%95%99%E7%A7%91%E6%9B%B8-%E3%83%87%E3%82%A3%E3%83%BC%E3%83%97%E3%83%A9%E3%83%BC%E3%83%8B%E3%83%B3%E3%82%B0-G%E6%A4%9C%E5%AE%9A%EF%BC%88%E3%82%B8%E3%82%A7%E3%83%8D%E3%83%A9%E3%83%AA%E3%82%B9%E3%83%88%EF%BC%89%E5%85%AC%E5%BC%8F%E3%83%86%E3%82%AD%E3%82%B9%E3%83%88-%E7%AC%AC3%E7%89%88-%E4%B8%80%E8%88%AC%E7%A4%BE%E5%9B%A3%E6%B3%95%E4%BA%BA%E6%97%A5%E6%9C%AC%E3%83%87%E3%82%A3%E3%83%BC%E3%83%97%E3%83%A9%E3%83%BC%E3%83%8B%E3%83%B3%E3%82%B0%E5%8D%94%E4%BC%9A/dp/4798184810/ref=sr_1_2_sspa?__mk_ja_JP=%E3%82%AB%E3%82%BF%E3%82%AB%E3%83%8A&crid=2KC18QUM9I00Q&dib=eyJ2IjoiMSJ9.sSDxeNwvPb5QgFKLo7IQA--zjE7zEcRV5Bf6OkzrzUqpLQ75Yoi3joz0LQk7IeU6K-Isbrj3U3dp7mNKDVN8rvD2mACJEpvgJK_MSg0moXqaWQDpo3Wo7ytsyaZvBtJJagu4PbRlXvUpa8mY2O4BENLmG2HbXBBfdHtxVPFzZFY8UMHDksAGQKUfLzObTK0yF9od-fNEj_CzK6otzjAAFVwwifeqPkVb_UUbfE5FUJ2JAyn9Zj6skocQZ8XI6ezGxSdnWtEbreAH0-Ixnxt0KoXgFUrywPpSPc8EeOU8DHQ.davdCLdw_DEgfz9tutB354MLe2OKa6e56etPxki2bf8&dib_tag=se&keywords=G%E6%A4%9C%E5%AE%9A&qid=1771732944&sprefix=g%E6%A4%9C%E5%AE%9A%2Caps%2C187&sr=8-2-spons&sp_csd=d2lkZ2V0TmFtZT1zcF9hdGY&psc=1)

# 人工ニューロン(パーセプトロン)
パーセプトロンは、脳の神経細胞を模したモデル。
### パーセプトロンの使用用途
入力に基づいた出力が0か1になる**二値分類**に用いることができる。
二値分類とは、**入力をクラス0とクラス1の2クラスに分類**すること。
### パーセプトロンの概要
入力を1つ与えると、出力が1つ得られる。
出力の種類は、0か1の2種類あり、入力の信号(実数)がニューロンに与えられると、0か1が出力される。
ニューロンには閾値が存在し、入力値が閾値以上ならば1、閾値未満なら0となる。
### パーセプトロンの原理
**入力値ベクトル**$x$が行列として入力されると考える。
**入力値**は入力値ベクトルの各1行にあたり、神経細胞に与えられる外部からの1刺激と考えられる。
1刺激は任意の個数(列数)の値が合わさったものと考えられる。
入力値の列は**特徴量**と呼ばれ、1つ(1行)の入力値がn列の場合、n個の特徴量があるということになり、**n次元**のデータとなる。
パーセプトロンの学習では、**重みベクトル**にある各**重み**(各特徴量に対する感度)は最適値を求めたい**パラメータ**となる。

$$
\begin{gather}
入力値ベクトル(m行n列)\ \ x= 
\left[
\begin{array}{}
  x_{11}&x_{12}&\cdots&x_{1n}\\
  x_{21}&x_{22}&\cdots&x_{2n}\\
  \vdots&\vdots&\ddots&\vdots\\
  x_{m1}&x_{m2}&\cdots&x_{mn}
\end{array}
\right]
,\ \
重みベクトル(1行n列)\ \ w=\left[
\begin{array}{}
w_1,\ w_2,\ \cdots,\ w_n
\end{array}
\right]
\end{gather}
$$
1行の入力値と重みベクトルのドット積(内積)を**総入力**$z$として、これを最終的な入力とする。
$$
m行目の総入力(ドット積)\ \ z_m= w\cdot x_m = w_1x_{m1} + w_2x_{m2} + \cdots + w_nx_{mn}
$$


**決定関数**$σ(z)$は、総入力$z$と**閾値**$θ$を基に0か1を出力する。
閾値は、神経細胞の発火しやすさを表していて、0は発火していない状態、1は発火した状態と言い換えられる。
パーセプトロンの学習では、決定関数の閾値が、重みベクトルと同様、最適値を求めたいパラメータとなる。
$$
決定関数 \ \ σ(z_m) =
\begin{cases}
1,\ \ (z \ge θ) \\
0,\ \ (z \lt θ)
\end{cases}
$$

### パーセプトロンの学習
重みベクトルにある各重み(各特徴量に対する感度)と閾値を調整することで、望ましい出力(神経細胞の反応)が得られるようにする。
望ましい出力とは、入力データに付属している**正解データ**と同じ出力のこと。
入力データから得られた出力(0か1)に応じて、パラメータである重みベクトル$w$と閾値$θ$を更新する。
**バイアスユニット**を$b=-θ$として、閾値を総入力に組み込むことで、全てのパラメータが1つの式で更新できるようになる。
総入力をバイアスユニットで調整する感じ。
$$
\begin{gather}
閾値の式変形(b=-θ)\\
z \ge θ \iff z-θ \ge0 \iff z+b\ge0\\
z < θ \iff z-θ<0 \iff z+b<0\\\\
総入力zを再定義\\
z_m = w\cdot x_m + b\\\\
決定関数はこうなる\\
 \ \ σ(z_m) =
\begin{cases}
1,\ \ (z \ge 0) \\
0,\ \ (z \lt 0)
\end{cases}
\end{gather}
$$
$i$行$j$列の入力データと、行ごとの正解データが与えられるとする。
重みベクトル$w$とバイアス$b$を学習で更新したい。
$$

\begin{gather}
事前に与えられるデータ \\
入力データ\ 
x= 
\left[
\begin{array}{}
  x_{11}&x_{12}&\cdots&x_{1j}\\
  x_{21}&x_{22}&\cdots&x_{2j}\\
  \vdots&\vdots&\ddots&\vdots\\
  x_{i1}&x_{i2}&\cdots&x_{ij}
\end{array}
\right],\ \
正解データ\ y=\left[
\begin{array}{}
y_1\\
y_2\\
\vdots\\
y_i
\end{array}
\right]\\\\
学習で求めたい値\\
重みベクトル\ \ w=\left[
\begin{array}{}
w_1,\ w_2,\ \cdots,\ w_j
\end{array}
\right],\ \ 
バイアス\ b\ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \
\end{gather}
$$
各入力データから得られる出力と正解データができる限り一致するように、重みベクトル$w$とバイアス$b$を更新することを考える。
ここで、$\hat y_i$を入力$x_i$と重みベクトル$w$から得られた**予測値**(出力)とすると、**予測値ベクトル**は$\hat y$となり、正解データ$y$と同じ形状になる。
$$
\begin{gather}
予測値ベクトル\\
\hat y=\left[
\begin{array}{}
\hat y_1\\
\hat y_2\\
\vdots\\
\hat y_i
\end{array}
\right]
\\ \\
重みベクトルとバイアスの更新式\\
w\leftarrow w+\Delta w\\
b\leftarrow b+\Delta b\ \\\\
更新値の計算式(\alpha は任意の値)\\
\Delta w=\alpha \sum_i (y-\hat y)\cdot x_{i}\\
\Delta b=\alpha \sum_i (y-\hat y)
\end{gather}
$$
今回は0か1が出力のため、予測値$\hat y_i$が正解値$y_i$と同じ(=正しく予測できた)場合は$(y_i-\hat y_i)$が0となり、更新値も0になるため、更新は事実上行われない。
予測値が正解値と違う(=正しく予測できなかった)場合は$(y_i-\hat y_i)$が非0となり、更新が行われる。
$\alpha$は**学習率=learning rate**で、更新値をどれだけ反映させるかを決める。(プログラムでは、$\alpha$以外に$eta$と書かれることもある)
学習率の注意点として、学習アルゴリズム内で更新されないパラメータ(**ハイパーパラメータ**)であるため、人間側で決める必要がある。
学習率が小さすぎると重みの更新がなかなか進まず、大きすぎると重みが収束から遠ざかりつづけてしまうことがあるため、良い感じの値を探索して見つける必要がある。
### プログラムに適用する
プログラムに適用しやすくするために、式の活用手順を明確にする。
1. 初めに、重みベクトル$w$とバイアスユニット$b$を初期化する(0か小さい乱数値)
2. 入力データ$x_{i}$(1行)ごとに総入力$z$を計算して、決定関数$σ$から出力値(予測値)$\hat y_i$を得る
3. 予測値に基づいて重みとバイアスを更新する
4. 2と3の演算を全ての行で行う
注意点1：出力は1行の入力ごとに得るため、重みとバイアスの更新も1行の出力が得られた時に、1行分行う
注意点2：バイアスは行列ではないため、行ごとのバイアスの更新値は全て一つのバイアス$b$に足される。
注意点3：1つの重みを何度も更新するため、全ての行列の演算は繰り返し行われる(例えば、1000回とか)。
