#coding:utf-8
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def get_files(file_dir):
    image = []
    label = []
    if not os.path.exists(file_dir):
        print("{} is not exits.".format(file_dir))
        exit()
    file = open(file_dir,'r')
    root = 'C:/Users/zanghao/Desktop/公司/yi/In-shop_Clothes_Retrieval_Benchmark/img/'
    for line in file:
        temp = line.strip('\n').split()
        image.append(root+temp[0])
        label.append(temp[1])
    temp = np.array([image,label])
    temp = temp.transpose()
    np.random.shuffle(temp)
    image_list = list(temp[:,0])
    label_list = list(temp[:,1])
    #label_list = [int(i.split('0')[]) for i in label_list]
    return image_list,label_list

def get_batch(image,label,image_W,image_H,batch_size,capacity):

    image = tf.cast(image,tf.string)
    label = tf.cast(label,tf.string)
    #generate input queue
    input_queue = tf.train.slice_input_producer([image,label])
    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])
    image = tf.image.decode_jpeg(image_contents,channels=3)
    image = tf.image.resize_image_with_crop_or_pad(image,image_W,image_H)
    image = tf.image.per_image_standardization(image)
    image_batch,label_batch = tf.train.batch([image,label],
                                             batch_size=batch_size,
                                             num_threads=32,
                                             capacity=capacity)
    label_batch = tf.reshape(label_batch,[batch_size])
    image_batch = tf.cast(image_batch,tf.float32)

    return image_batch,label_batch

def build_input():
    BATCH_SIZE = 1
    CAPACITY = 256
    IMG_W = 256
    IMG_H = 256
    file_dir = 'C:/Users/zanghao/Desktop/公司/yi/In-shop_Clothes_Retrieval_Benchmark/Eval/train.txt'
    image_list, label_list = get_files(file_dir)
    image_batch, label_batch = get_batch(image_list,label_list,IMG_W,IMG_H,BATCH_SIZE,CAPACITY)
    label_batch = tf.zeros([BATCH_SIZE, 1])
    return image_batch,label_batch


def main():
    BATCH_SIZE = 2
    CAPACITY = 256
    IMG_W = 256
    IMG_H = 256
    file_dir = 'C:/Users/zanghao/Desktop/公司/yi/In-shop_Clothes_Retrieval_Benchmark/Eval/train.txt'
    image_list, label_list = get_files(file_dir)
    image_batch, label_batch = get_batch(image_list,label_list,IMG_W,IMG_H,BATCH_SIZE,CAPACITY)
    with tf.Session() as sess:
        i = 0
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        try:
            while not coord.should_stop() and i<2:
                img,label = sess.run([image_batch,label_batch])
                for j in np.arange(BATCH_SIZE):
                    print('label: %s' %label[i])
                    plt.imshow(img[j,:,:,:])
                    plt.show()
                i+=1
        except tf.errors.OutOfRangeError:
            print ('done')
        finally:
            coord.request_stop()
        coord.join(threads)

if __name__ == "__main__":
    main()



