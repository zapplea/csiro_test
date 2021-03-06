import tensorflow as tf
import pickle
import numpy as np

def gen():
    f=open('/datastore/che313/yibing_data/nosqldb/table.pkl','rb')
    dic=pickle.load(f)
    table=pickle.load(f)
    g=tf.Graph()
    with g.device('/gpu:0'):
        with g.as_default():
            print('create graph')
            t=tf.Variable(np.array(table),name='table')
            saver=tf.train.Saver()
        with tf.Session(graph=g,config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            print('print t:')
            sess.run(tf.global_variables_initializer())
            print('run table')
            print(sess.run(t)[:10])
            saver.save(sess,'model/lookup_model.ckpt')


def restore():
    g=tf.Graph()
    with g.device('/gpu:0'):
        with g.as_default():
            saver=tf.train.import_meta_graph('model/lookup_model.ckpt.meta')
            t = g.get_tensor_by_name('table:0')
        with tf.Session(graph=g,config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            saver.restore(sess,'model/lookup_model.ckpt')
            print(sess.run(t)[:10])


if __name__ == '__main__':
    print('=========generate=========')
    gen()
    print('=========restore=========')
    restore()
