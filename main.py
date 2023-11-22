## 网络构建部分: refer: https://github.com/dragen1860/MAML-TensorFlow

#################################################
# 任务描述：5-ways，1-shot图像分类任务，图像统一处理成 84 * 84 * 3 = 21168的尺寸。分类5个，每一个分类中一个图像
# support set：5 * 1  训练集
# query set：5 * 15  测试集
# 训练取1个batch的任务：batch size：4
# 对训练任务进行训练时，更新5次：K = 5
#################################################

print(support_x)  # (4, 5, 21168)
print(query_x)  # (4, 75, 21168)
print(support_y)  # (4, 5, 5)
print(query_y)  # (4, 75, 5)
print(meta_batchsz)  # 4
print(K)  # 5

model = MAML()
model.build(support_x, support_y, query_x, query_y, K, meta_batchsz, mode='train')


class MAML:
    def __init__(self):
        pass

    def build(self, support_xb, support_yb, query_xb, query_yb, K, meta_batchsz, mode='train'):
        """
        :param support_xb: [4, 5, 84*84*3]
        :param support_yb: [4, 5, n-way]
        :param query_xb:  [4, 75, 84*84*3]
        :param query_yb: [4, 75, n-way]
        :param K:  训练任务的网络更新步数
        :param meta_batchsz: 任务数，4
        """

        self.weights = self.conv_weights()  # 创建或者复用网络参数；训练任务对应的网络复用meta网络的参数
        training = True if mode is 'train' else False

        def meta_task(input):
            """
            :param support_x:   [setsz, 84*84*3] (5, 21168)
            :param support_y:   [setsz, n-way] (5, 5)
            :param query_x:     [querysz, 84*84*3] (75, 21168)
            :param query_y:     [querysz, n-way] (75, 5)
            :param training:    training or not, for batch_norm
            :return:
            """

            support_x, support_y, query_x, query_y = input
            query_preds, query_losses, query_accs = [], [], []  # 子网络更新K次，记录每一次queryset的结果

            ## 第0次对网络进行更新
            support_pred = self.forward(support_x, self.weights, training)  # 前向计算support set
            support_loss = tf.nn.softmax_cross_entropy_with_logits(logits=support_pred, labels=support_y)  # support set loss
            # 计算交叉熵损失
            support_acc = tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(support_pred, dim=1), axis=1),
                                                      tf.argmax(support_y, axis=1))
            # 计算准确率
            grads = tf.gradients(support_loss, list(self.weights.values()))  # 计算support set的梯度
            gvs = dict(zip(self.weights.keys(), grads)) # 权重 - 梯度
            # 使用support set的梯度计算的梯度更新参数，theta_pi = theta - alpha * grads
            fast_weights = dict(zip(self.weights.keys(),
                                    [self.weights[key] - self.train_lr * gvs[key] for key in self.weights.keys()]))
            # 更新模型参数

            # 使用梯度更新后的参数对quert set进行前向计算
            query_pred = self.forward(query_x, fast_weights, training)
            query_loss = tf.nn.softmax_cross_entropy_with_logits(logits=query_pred, labels=query_y)
            query_preds.append(query_pred)
            query_losses.append(query_loss)

            # 第1到 K-1次对网络进行更新
            for _ in range(1, K):
                loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.forward(support_x, fast_weights, training),
                                                               labels=support_y)
                grads = tf.gradients(loss, list(fast_weights.values()))
                gvs = dict(zip(fast_weights.keys(), grads))
                fast_weights = dict(zip(fast_weights.keys(), [fast_weights[key] - self.train_lr * gvs[key]
                                                              for key in fast_weights.keys()]))
                query_pred = self.forward(query_x, fast_weights, training)
                query_loss = tf.nn.softmax_cross_entropy_with_logits(logits=query_pred, labels=query_y)
                # 子网络更新K次，记录每一次queryset的结果
                query_preds.append(query_pred)
                query_losses.append(query_loss)

            for i in range(K):
                query_accs.append(tf.contrib.metrics.accuracy(tf.argmax(tf.nn.softmax(query_preds[i], dim=1), axis=1),
                                                              tf.argmax(query_y, axis=1)))
            result = [support_pred, support_loss, support_acc, query_preds, query_losses, query_accs]
            return result

        # return: [support_pred, support_loss, support_acc, query_preds, query_losses, query_accs]
        out_dtype = [tf.float32, tf.float32, tf.float32, [tf.float32] * K, [tf.float32] * K, [tf.float32] * K]
        result = tf.map_fn(meta_task, elems=(support_xb, support_yb, query_xb, query_yb),
                           dtype=out_dtype, parallel_iterations=meta_batchsz, name='map_fn')
        support_pred_tasks, support_loss_tasks, support_acc_tasks, \
            query_preds_tasks, query_losses_tasks, query_accs_tasks = result

        if mode is 'train':
            self.support_loss = support_loss = tf.reduce_sum(support_loss_tasks) / meta_batchsz
            self.query_losses = query_losses = [tf.reduce_sum(query_losses_tasks[j]) / meta_batchsz
                                                for j in range(K)]
            self.support_acc = support_acc = tf.reduce_sum(support_acc_tasks) / meta_batchsz
            self.query_accs = query_accs = [tf.reduce_sum(query_accs_tasks[j]) / meta_batchsz
                                            for j in range(K)]

            # 更新meta网络，只使用了第 K步的query loss。这里应该是个超参，更新几步可以调调
            optimizer = tf.train.AdamOptimizer(self.meta_lr, name='meta_optim')
            gvs = optimizer.compute_gradients(self.query_losses[-1])
# def ********