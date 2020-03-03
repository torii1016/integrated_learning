from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

print(mnist.train.images.shape)
"""
# モデルの作成
# 画像データをxとする
x = tf.placeholder(tf.float32, [None, 784])
# モデルの重みをwと設定する
W = tf.Variable(tf.zeros([784,10]))
# モデルのバイアス
b = tf.Variable(tf.zeros([10]))
# トレーニングデータ
y = tf.nn.softmax(tf.matmul(x, W) + b)
# 正解のデータ
y_ = tf.placeholder(tf.float32, [None, 10])
# 損失関数をクロスエントロピーとする
#
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
# 学習係数を指定して勾配降下アルゴリズムを用いてクロスエントロピーを最小化する
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# 変数の初期化
init = tf.initialize_all_variables()
# セッションの作成
sess = tf.Session()
# セッションの開始及び初期化
sess.run(init)

#学習
for i in range(1000):
    #トレーニングデータからランダムに100個抽出する
    batch_xs, batch_ys = mnist.train.next_batch(100)
    #確率的勾配降下法によりクロスエントロピーを最小化するような重みを更新する
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    #予測値と正解値を比較してbool値にする
    #argmax(y,1)は予測値の各行で最大となるインデックスをひとつ返す
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    #boole値を0もしくは1に変換して平均値をとる、これを正解率とする
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
"""