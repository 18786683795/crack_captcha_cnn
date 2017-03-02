import numpy as np
import matplotlib.image as mpimg
import tensorflow as tf
IMAGE_HEIGHT = 50
IMAGE_WIDTH = 160
MAX_CAPTCHA = 4
PIC_COUNT = 128
number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
char_set = number+alphabet+ALPHABET
CHAR_SET_LEN = len(char_set)
def all_text():
    a_text = []
    try:
        file = open('e:/captcha.txt','r')
        line = file.readline()  # 调用文件的 readline()方法
        while line:
            line = line.replace('\n','')
            a_text.append(line)
            line = file.readline()
    finally:
        if file:
            file.close()
    return a_text
def get_img(index):
    filename = "e:/img/"+str(index)+'.jpg'
    captcha_image = mpimg.imread(filename)
    captcha_image = np.array(captcha_image)
    return captcha_image
#转为灰度图像
def convert2gray(img):
    if len(img.shape) > 2:
        gray = np.mean(img, -1)
        # 上面的转法较快，正规转法如下
        # r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
        # gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray
    else:
        return img
# 文本转向量
def text2vec(text):
    text_len = len(text)
    if text_len>MAX_CAPTCHA:
        raise ValueError('验证码最长4个字符')
    vector = np.zeros(MAX_CAPTCHA*CHAR_SET_LEN)
    def char2pos(c):
        if c == '_':
            k = 62
            return k
        k = ord(c)-48
        if k>9:
            k=ord(c)-55
            if k>35:
                k = ord(c)-61
                if k>61:
                    raise ValueError('No map')
        return k
    for i,c in enumerate(text):
        idx = i*CHAR_SET_LEN + char2pos(c)
        vector[idx] = 1
    return vector
# 向量转回文本
def vec2text(vec):
    char_pos = vec.nonzero()[0]
    text=[]
    for i,c in enumerate(char_pos):
        char_at_pos = i
        char_idx = c % CHAR_SET_LEN
        if char_idx < 10:
            char_code = char_idx + ord('0')
        elif char_idx < 36:
            char_code = char_idx - 10 + ord('A')
        elif char_idx < 62:
            char_code = char_idx - 36 + ord('a')
        else:
            raise ValueError('error')
        text.append(chr(char_code))
    return "".join(text)
def get_next_batch(a_text,start,batch_size=128):
    batch_x = np.zeros([batch_size, IMAGE_HEIGHT*IMAGE_WIDTH])
    batch_y = np.zeros([batch_size, MAX_CAPTCHA*CHAR_SET_LEN])
    for i in range(batch_size):
        text = a_text[start+i]
        print(text+"index:"+str(start+i))
        image = get_img(index=start+i)
        image = convert2gray(image)
        batch_x[i, :] = image.flatten() / 255
        batch_y[i, :] = text2vec(text)
    return batch_x, batch_y


X = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT * IMAGE_WIDTH])
Y = tf.placeholder(tf.float32, [None, MAX_CAPTCHA * CHAR_SET_LEN])
keep_prob = tf.placeholder(tf.float32)  # dropout

def crack_captcha_cnn(w_alpha=0.01, b_alpha=0.1):
    x = tf.reshape(X, shape=[-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])

    w_c1 = tf.Variable(w_alpha * tf.random_normal([3, 3, 1, 32]))
    b_c1 = tf.Variable(b_alpha * tf.random_normal([32]))
    conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, w_c1, strides=[1, 1, 1, 1], padding='SAME'), b_c1))
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv1 = tf.nn.dropout(conv1, keep_prob)

    w_c2 = tf.Variable(w_alpha * tf.random_normal([3, 3, 32, 64]))
    b_c2 = tf.Variable(b_alpha * tf.random_normal([64]))
    conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv1, w_c2, strides=[1, 1, 1, 1], padding='SAME'), b_c2))
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv2 = tf.nn.dropout(conv2, keep_prob)

    w_c3 = tf.Variable(w_alpha * tf.random_normal([3, 3, 64, 64]))
    b_c3 = tf.Variable(b_alpha * tf.random_normal([64]))
    conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv2, w_c3, strides=[1, 1, 1, 1], padding='SAME'), b_c3))
    conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    conv3 = tf.nn.dropout(conv3, keep_prob)

    # Fully connected layer
    w_d = tf.Variable(w_alpha * tf.random_normal([7 * 32 * 40, 1024]))
    b_d = tf.Variable(b_alpha * tf.random_normal([1024]))
    dense = tf.reshape(conv3, [-1, w_d.get_shape().as_list()[0]])
    dense = tf.nn.relu(tf.add(tf.matmul(dense, w_d), b_d))
    dense = tf.nn.dropout(dense, keep_prob)

    w_out = tf.Variable(w_alpha * tf.random_normal([1024, MAX_CAPTCHA * CHAR_SET_LEN]))
    b_out = tf.Variable(b_alpha * tf.random_normal([MAX_CAPTCHA * CHAR_SET_LEN]))
    out = tf.add(tf.matmul(dense, w_out), b_out)
    # out = tf.nn.softmax(out)
    return out
a_text = all_text()
# 训练
def train_crack_captcha_cnn():
    output = crack_captcha_cnn()
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=output, logits=Y))
    # 最后一层用来分类的softmax和sigmoid有什么不同？
    # optimizer 为了加快训练 learning_rate应该开始大，然后慢慢衰
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

    predict = tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN])
    max_idx_p = tf.argmax(predict, 2)
    max_idx_l = tf.argmax(tf.reshape(Y, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)
    correct_pred = tf.equal(max_idx_p, max_idx_l)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        step = 0
        start_index = 0
        my_batch_size = 64
        while True:
            batch_x, batch_y = get_next_batch(a_text, start=start_index, batch_size=my_batch_size)
            start_index = (start_index+my_batch_size) % PIC_COUNT;
            _, loss_ = sess.run([optimizer, loss], feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.75})
            # print(step, loss_)
            # 每100 step计算一次准确率
            if step % 100 == 0:
                batch_x_test, batch_y_test = get_next_batch(a_text, start=start_index, batch_size=my_batch_size)
                start_index = (start_index + my_batch_size) % PIC_COUNT;
                acc = sess.run(accuracy, feed_dict={X: batch_x_test, Y: batch_y_test, keep_prob: 1.})
                print('当前准确率为：'+ acc)
                # 如果准确率大于50%,保存模型,完成训练
                if step >= PIC_COUNT:
                    saver.save(sess, "e:/crack_capcha.model", global_step=step)
                    print('完成并保存')
                    break
            step += 1
train_crack_captcha_cnn()
