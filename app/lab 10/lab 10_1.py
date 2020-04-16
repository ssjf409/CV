import tensorflow as tf

#x_train=[1, 2, 3]
#y_train=[1, 2, 3]

X_data=[[1,2,3],[2,4,6],[3,6,9]]
y_data=[[1], [2], [3]];

X=tf.placeholder(tf.float32, [None,3])
Y=tf.placeholder(tf.float32, [None,1])

W=tf.Variable(tf.random_normal([3, 1]), name='weight')
b=tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.matmul(X,W)+b

cost=tf.reduce_mean(tf.square(hypothesis-Y))

optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.01)
train=optimizer.minimize(cost)

sess=tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001):
	h_, W_, b_, _ = sess.run([hypothesis, W, b, train], feed_dict={X:X_data, Y:y_data})
	if step%20 == 0:
		print(step, h_, W_, b_)

h_=sess.run(hypothesis, feed_dict={X:[[1.5, 3, 4.5]]})
print(h_)

