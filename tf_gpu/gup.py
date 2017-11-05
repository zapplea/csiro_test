import tensorflow as tf

g=tf.Graph()
with g.device('/gpu:0'):
    with g.as_default():
        a=tf.constant([1,2,3])
       	b=tf.constant([1,2,3])
       	c=tf.add(a,b)
    with tf.Session(graph=g,config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        print(sess.run(c))