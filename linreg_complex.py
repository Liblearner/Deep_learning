import random
import torch

# 生成数据集
# w = [2,-3.4]和b = 4.2和噪声c生成数据集(原始的X值)和标签（真实y值）
# 标准差为0.01


def synthetic_data(w, b, num_examples):
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b  # 乘法
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))


# 读取数据集
# batch_size：每个批量中的个数；features:特征矩阵；labels：标签值
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    # 这些样本是随机读取的，没有特定的顺序
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
            indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]
        # 重要，这个yield在for循环的内部，否则每次利用的数据不对


# 定义模型,这里就是线性模型的计算式
def linreg(X, w, b):
    return torch.matmul(X, w) + b


# 定义平方损失函数
def squared_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


# 定义优化算法，这里使用小批量随机梯度下降
# params:随机的待计算参数，lr：学习率，batch_size：小批量的个数，取平均
def sgd(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()


# 定义真实参数
true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)

# 每个小批的数量为10
batch_size = 10
# X,y用于遍历，data_iter中的yield用于生成迭代器，这样的话返回值就可以生成可遍历对象
# batch_size = 10；feature1000个1*2，labels1000个1*1；
# 可以看到每次都返回了1*20*2的矩阵；1*20*1的矩阵
for X, y in data_iter(batch_size, features, labels):
    print(X, '\n', y)
    break
# 初始化参数模型：
w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

'''
训练过程:
每次迭代中，我们读取一小批量训练样本，并通过我们的模型来获得一组预测。
计算损失之后，我们开始反向传播，存储每个参数的梯度，最后，调用优化算法sgd来更新模型参数
'''
lr = 0.03  # 超参数学习率
num_epochs = 3  # 迭代周期，同样是超参数
net = linreg
loss = squared_loss

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y)
        l.sum().backward()
        sgd([w, b], lr, batch_size)
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1},loss{float(train_l.mean()):f}')

print(f'w的估计误差: {true_w - w.reshape(true_w.shape)}')
print(f'b的估计误差: {true_b - b}')
