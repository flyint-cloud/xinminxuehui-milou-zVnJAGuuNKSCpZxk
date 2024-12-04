[合集 \- 机器学习(5\)](https://github.com)[1\.机器学习：线性回归（上）11\-19](https://github.com/SXWisON/p/18554744)[2\.机器学习：线性回归（下）11\-26](https://github.com/SXWisON/p/18569342)[3\.机器学习：逻辑回归12\-02](https://github.com/SXWisON/p/18581725):[豆荚加速器官网](https://baitenghuo.com)[4\.机器学习：神经网络构建（上）12\-03](https://github.com/SXWisON/p/18582519)5\.机器学习：神经网络构建（下）12\-04收起
#### 简介


在上一篇文章《[机器学习：神经网络构建（上）](https://github.com)》中讨论了线性层、激活函数以及损失函数层的构建方式，本节中将进一步讨论网络构建方式，并完整的搭建一个简单的分类器网络。


#### 目录


1. 网络Network
2. 数据集管理器 DatasetManager
3. 优化器 Optimizer
4. 代码测试


## 网络Network


### 网络定义




---


在设计神经网络时，其基本结构是由一层层的神经元组成的，这些层可以是输入层、隐藏层和输出层。为了实现这一结构，通常会使用向量（vector）容器来存储这些层，因为层的数量是可变的，可能根据具体任务的需求而变化。


即使在网络已经进行了预训练并具有一定的参数的情况下，对于特定的任务，通常还是需要进行模型微调。这是因为不同的任务可能有不同的数据分布和要求，因此训练是构建高性能神经网络模型的重要步骤。


在训练过程中，有三个关键组件：


1. **损失函数**：神经网络的学习目标，通过最小化损失函数来优化模型参数。选择合适的损失函数对于确保模型能够学习到有效的特征表示至关重要。
2. **优化器**：优化器负责调整模型的参数以最小化损失函数。除了基本的参数更新功能外，优化器还可以提供更高级的功能，如学习率调整和参数冻结，这些功能有助于提高训练效率和模型性能。
3. **数据集管理器**：负责在训练过程中有效地管理和提供数据，包括数据的加载、预处理和批处理，以确保数据被充分利用。


对于网络的外部接口（公有方法），主要有以下几类：


1. **网络设置**：添加网络层、设置损失函数、优化器和数据集等操作，用于配置网络的结构和训练参数。
2. **网络推理**：前向传播和反向传播方法，用于在训练和测试过程中进行预测和参数更新。
3. **网络训练**：使用配置好的数据集和训练方法，执行指定次数的训练迭代，以优化网络参数。


以下是代码示例：



```


|  | class Network { |
| --- | --- |
|  | private: |
|  | vector> layers; |
|  |  |
|  | shared_ptr lossFunction; |
|  | shared_ptr optimizer; |
|  | shared_ptr datasetManager; |
|  |  |
|  | public: |
|  | void addLayer(shared_ptr layer); |
|  |  |
|  | void setLossFunction(shared_ptr lossFunc); |
|  | void setOptimizer(shared_ptr opt); |
|  | void setDatasetManager(shared_ptr manager); |
|  |  |
|  | MatrixXd forward(const MatrixXd& input); |
|  | void backward(const MatrixXd& outputGrad); |
|  |  |
|  | double train(size_t epochs, size_t batchSize); |
|  | }; |


```

使用shared\_ptr的好处：
存储方式vector\>和vector相比，如果直接存储 Layer 对象，需要手动管理内存，包括分配和释放内存，这不仅容易出错，还可能导致内存泄漏或悬挂指针的问题。而使用 std::shared\_ptr 可以大大简化内存管理，提高代码的健壮性和可维护性。


### 网络训练




---


网络的训练函数通常包含两个输入参数，训练的集数和批尺寸：


* **集数`epochs`**：指训练集被完整的迭代的次数。在每一个epoch中，网络会使用训练集中的所有样本进行参数更新。
* **批尺寸`batchSize`**：指在一次迭代中用于更新模型参数的样本数量。在每次迭代中，模型会计算这些样本的总梯度，并据此调整模型的参数。


因此，网络的训练函数由两层循环结构组成，外层循环结构表示完整迭代的次数，直至完成所有迭代时停止。内层循环表示训练集中样本被网络调取的进度，直至训练集中的所有数据被调用时停止。


网络的训练过程是由多次的参数迭代（更新）完成的。而参数的的迭代是以批（Batch）为单位的。具体来说，一次迭代包含如下步骤：


1. **获取数据**：从数据集管理器中获取一批的数据（包含输入和输出）
2. **前向传播**：采用网络对数据进行推理，得到预测结果，依据预测结果评估损失。
3. **反向传播**：计算损失函数关于各层参数的梯度。
4. **参数更新**：依据损失、梯度等信息，更新各层梯度。
5. **日志更新**：计算并输出每个epoch的累积误差。


代码设计如下：



```


|  | double Network::train(size_t epochs, size_t batchSize) { |
| --- | --- |
|  | double totalLoss = 0.0; |
|  | size_t sampleCount = datasetManager->getTrainSampleCount(); |
|  |  |
|  | for (size_t epoch = 0; epoch < epochs; ++epoch) { |
|  | datasetManager->shuffleTrainSet(); |
|  | totalLoss = 0.0; |
|  | for (size_t i = 0; i < sampleCount; i += batchSize) { |
|  | // 获取一个小批量样本 |
|  | auto batch = datasetManager->getTrainBatch(batchSize, i / batchSize); |
|  | MatrixXd batchInput = batch.first; |
|  | MatrixXd batchLabel = batch.second; |
|  |  |
|  | // 前向传播 |
|  | MatrixXd predicted = forward(batchInput); |
|  | double loss = lossFunction->computeLoss(predicted, batchLabel); |
|  |  |
|  | // 反向传播 |
|  | MatrixXd outputGrad = lossFunction->computeGradient(predicted, batchLabel); |
|  | backward(outputGrad); |
|  |  |
|  | // 参数更新 |
|  | optimizer->update(layers); |
|  |  |
|  | // 累计损失 |
|  | totalLoss += loss; |
|  | } |
|  | totalLoss /= datasetManager->getTrainSampleCount(); |
|  | // 输出每个epoch的损失等信息 |
|  | std::cout << "Epoch " << epoch << ", totalLoss = " << totalLoss << "\n"; |
|  | } |
|  | return totalLoss / (epochs * (sampleCount / batchSize)); // 返回平均损失（简化示例） |
|  | } |


```

### 网络的其它公有方法




---


下面的代码给出了网络的其它公有方法的代码实现：



```


|  | void Network::addLayer(std::shared_ptr layer) { |
| --- | --- |
|  | layers.push_back(layer); |
|  | } |
|  |  |
|  | void Network::setLossFunction(std::shared_ptr lossFunc) { |
|  | lossFunction = lossFunc; |
|  | } |
|  |  |
|  | void Network::setOptimizer(std::shared_ptr opt) { |
|  | optimizer = opt; |
|  | } |
|  |  |
|  | void Network::setDatasetManager(std::shared_ptr manager) { |
|  | datasetManager = manager; |
|  | } |
|  |  |
|  | MatrixXd Network::forward(const MatrixXd& input) { |
|  | MatrixXd currentInput = input; |
|  | for (const auto& layer : layers) { |
|  | currentInput = layer->forward(currentInput); |
|  | } |
|  | return currentInput; |
|  | } |
|  |  |
|  | void Network::backward(const MatrixXd& outputGrad) { |
|  | MatrixXd currentGrad = outputGrad; |
|  | for (auto it = layers.rbegin(); it != layers.rend(); ++it) { |
|  | currentGrad = (*it)->backward(currentGrad); |
|  | } |
|  | } |


```


> `forward`方法除了作为训练时的步骤之一，还经常用于网络推理（预测），因此声明为公有方法



> `backward`方法只在训练时使用，在正常的使用用途中，不会被外部调用，因此，其可以声明为私有方法。


## 数据集管理器 DatasetManager




---


数据集管理器本质目的是提高网络对数据的利用率，其主要职能有：


1. 保存数据：提供更为安全可靠的数据管理。
2. 数据打乱：以避免顺序偏差，同时提升模型的泛化能力。
3. 数据集划分：讲数据划分为训练集、验证集和测试集。
4. 数据接口：使得外部可以轻松的获取批量数据。



```


|  | class DatasetManager { |
| --- | --- |
|  | private: |
|  | MatrixXd input; |
|  | MatrixXd label; |
|  | std::vector<int> trainIndices; |
|  | std::vector<int> valIndices; |
|  | std::vector<int> testIndices; |
|  |  |
|  | public: |
|  | // 设置数据集的方法 |
|  | void setDataset(const MatrixXd& inputData, const MatrixXd& labelData); |
|  |  |
|  | // 划分数据集为训练集、验证集和测试集 |
|  | void splitDataset(double trainRatio = 0.8, double valRatio = 0.1, double testRatio = 0.1); |
|  |  |
|  | // 获取训练集、验证集和测试集的小批量数据 |
|  | std::pair getBatch(std::vector<int>& indices, size_t batchSize, size_t offset = 0); |
|  |  |
|  | // 随机打乱训练集 |
|  | void shuffleTrainSet(); |
|  |  |
|  | // 获取批量数据 |
|  | std::pair getTrainBatch(size_t batchSize, size_t offset = 0); |
|  | std::pair getValidationBatch(size_t batchSize, size_t offset = 0); |
|  | std::pair getTestBatch(size_t batchSize, size_t offset = 0); |
|  |  |
|  | // 获取样本数量的方法 |
|  | size_t getSampleCount() const; |
|  | size_t getTrainSampleCount() const; |
|  | size_t getValidationSampleCount() const; |
|  | size_t getTestSampleCount() const; |
|  | }; |


```

### 数据集初始化




---


数据集初始化分为三步：数据集设置、数据集划分、数据集打乱。



```


|  | // 设置数据集 |
| --- | --- |
|  | void  ML::DatasetManager::setDataset(const MatrixXd& inputData, const MatrixXd& labelData) { |
|  | input = inputData; |
|  | label = labelData; |
|  |  |
|  | trainIndices.resize(input.rows()); |
|  | std::iota(trainIndices.begin(), trainIndices.end(), 0); |
|  | valIndices.clear(); |
|  | testIndices.clear(); |
|  | } |
|  |  |
|  | // 打乱训练集 |
|  | void ML::DatasetManager::shuffleTrainSet() { |
|  | std::shuffle(trainIndices.begin(), trainIndices.end(), std::mt19937{ std::random_device{}() }); |
|  | } |
|  |  |
|  | // 划分数据集为训练集、验证集和测试集 |
|  | void ML::DatasetManager::splitDataset(double trainRatio, double valRatio, double testRatio) { |
|  | size_t totalSamples = input.rows(); |
|  | size_t trainSize = static_cast<size_t>(totalSamples * trainRatio); |
|  | size_t valSize = static_cast<size_t>(totalSamples * valRatio); |
|  | size_t testSize = totalSamples - trainSize - valSize; |
|  |  |
|  | shuffleTrainSet(); |
|  |  |
|  | valIndices.assign(trainIndices.begin() + trainSize, trainIndices.begin() + trainSize + valSize); |
|  | testIndices.assign(trainIndices.begin() + trainSize + valSize, trainIndices.end()); |
|  | trainIndices.resize(trainSize); |
|  | } |


```


> 对于打乱操作较频繁的场景，打乱索引是更为高效的操作；而对于不经常打乱的场景，直接在数据集上打乱更为高效。本例中仅给出打乱索引的代码示例。


### 数据获取




---


在获取数据时，首先明确所需数据集的类型（训练集或验证集）。然后，根据预设的批次大小（Batchsize），从索引列表中提取相应数量的索引，并将这些索引对应的数据存储到临时矩阵中。最后，导出数据，完成读取操作。



```


|  | // 获取训练集、验证集和测试集的小批量数据 |
| --- | --- |
|  | std::pair ML::DatasetManager::getBatch(std::vector<int>& indices, size_t batchSize, size_t offset) { |
|  | size_t start = offset * batchSize; |
|  | size_t end = std::min(start + batchSize, indices.size()); |
|  | MatrixXd batchInput = MatrixXd::Zero(end - start, input.cols()); |
|  | MatrixXd batchLabel = MatrixXd::Zero(end - start, label.cols()); |
|  |  |
|  | for (size_t i = start; i < end; ++i) { |
|  | batchInput.row(i - start) = input.row(indices[i]); |
|  | batchLabel.row(i - start) = label.row(indices[i]); |
|  | } |
|  |  |
|  | return std::make_pair(batchInput, batchLabel); |
|  | } |
|  |  |
|  | // 获取训练集的批量数据 |
|  | std::pair ML::DatasetManager::getTrainBatch(size_t batchSize, size_t offset) { |
|  | return getBatch(trainIndices, batchSize, offset); |
|  | } |
|  |  |
|  | // 获取验证集的批量数据 |
|  | std::pair ML::DatasetManager::getValidationBatch(size_t batchSize, size_t offset) { |
|  | return getBatch(valIndices, batchSize, offset); |
|  | } |
|  |  |
|  | // 获取测试集的批量数据 |
|  | std::pair ML::DatasetManager::getTestBatch(size_t batchSize, size_t offset) { |
|  | return getBatch(testIndices, batchSize, offset); |
|  | } |


```

### 数据集尺寸的外部接口




---


为便于代码开发，需要为数据集管理器设计外部接口，以便于外部可以获取各个数据集的尺寸。



```


|  | size_t ML::DatasetManager::getSampleCount() const { |
| --- | --- |
|  | return input.rows(); |
|  | } |
|  |  |
|  | size_t ML::DatasetManager::getTrainSampleCount() const { |
|  | return trainIndices.size(); |
|  | } |
|  |  |
|  | size_t ML::DatasetManager::getValidationSampleCount() const { |
|  | return valIndices.size(); |
|  | } |
|  |  |
|  | size_t ML::DatasetManager::getTestSampleCount() const { |
|  | return testIndices.size(); |
|  | } |


```

## 优化器 Optimizer




---


随机梯度下降是一种优化算法，用于最小化损失函数以训练模型参数。与批量梯度下降（Batch Gradient Descent）不同，SGD在每次更新参数时只使用一个样本（或一个小批量的样本），而不是整个训练集。这使得SGD在计算上更高效，且能够更快地收敛，尤其是在处理大规模数据时。以下为随机梯度下降的代码示例：



```


|  | class Optimizer { |
| --- | --- |
|  | public: |
|  | virtual void update(std::vector>& layers) = 0; |
|  | virtual ~Optimizer() {} |
|  | }; |
|  |  |
|  | class SGDOptimizer : public Optimizer { |
|  | private: |
|  | double learningRate; |
|  | public: |
|  | SGDOptimizer(double learningRate) : learningRate(learningRate) {} |
|  | void update(std::vector>& layers) override; |
|  | }; |
|  |  |
|  | void SGDOptimizer::update(std::vector>& layers) { |
|  | for (auto& layer : layers) { |
|  | layer->update(learningRate); |
|  | } |
|  | } |


```

## 代码测试




---


如果你希望测试这些代码，首先可以从本篇文章，以及[上一篇文章](https://github.com)中复制代码，并参考下述图片构建你的解决方案。
![description](https://img2024.cnblogs.com/blog/3320410/202412/3320410-20241204010619712-987798472.png)
如果你有遇到问题，欢迎联系作者！


### 示例1：线性回归




---


下述代码为线性回归的测试样例：



```


|  | namespace LNR{ |
| --- | --- |
|  | // linear_regression |
|  | void gen(MatrixXd& X, MatrixXd& y); |
|  | void test(); |
|  | } |
|  |  |
|  | void LNR::gen(MatrixXd& X, MatrixXd& y) { |
|  | MatrixXd w(X.cols(), 1); |
|  |  |
|  | X.setRandom(); |
|  | w.setRandom(); |
|  |  |
|  | X.rowwise() -= X.colwise().mean(); |
|  | X.array().rowwise() /= X.array().colwise().norm(); |
|  |  |
|  | y = X * w; |
|  | } |
|  |  |
|  | void LNR::test() { |
|  | std::cout << std::fixed << std::setprecision(2); |
|  |  |
|  | size_t input_dim = 10; |
|  | size_t sample_num = 2000; |
|  |  |
|  | MatrixXd X(sample_num, input_dim); |
|  | MatrixXd y(sample_num, 1); |
|  |  |
|  | gen(X, y); |
|  |  |
|  | ML::DatasetManager dataset; |
|  | dataset.setDataset(X, y); |
|  |  |
|  | ML::Network net; |
|  |  |
|  | net.addLayer(std::make_shared(input_dim, 1)); |
|  |  |
|  | net.setLossFunction(std::make_shared()); |
|  | net.setOptimizer(std::make_shared(0.25)); |
|  | net.setDatasetManager(std::make_shared(dataset)); |
|  |  |
|  | size_t epochs = 600; |
|  | size_t batch_size = 50; |
|  | net.train(epochs, batch_size); |
|  |  |
|  | MatrixXd error(sample_num, 1); |
|  |  |
|  | error = net.forward(X) - y; |
|  |  |
|  | std::cout << "error=\n" << error << "\n"; |
|  | } |


```

**详细解释**


1. `gen`函数：用以生成测试数据。
2. 网络结构：本例的网络结构中只包含一个线性层，其中输入尺寸为特征维度，输出尺寸为1。
3. 损失函数：采用MSE均方根误差作为损失函数。


**输出展示**


完成训练后，网络预测值与真实值的误差如下图；容易发现，网络具有较好的预测精度。


![description](https://img2024.cnblogs.com/blog/3320410/202412/3320410-20241204012153994-980525517.png)
### 示例2：逻辑回归




---


下述代码为逻辑回归的测试样例：



```


|  | namespace LC { |
| --- | --- |
|  | // Linear classification |
|  | void gen(MatrixXd& X, MatrixXd& y); |
|  | void test(); |
|  | } |
|  |  |
|  | void LC::gen(MatrixXd& X, MatrixXd& y) { |
|  | MatrixXd w(X.cols(), 1); |
|  |  |
|  | X.setRandom(); |
|  | w.setRandom(); |
|  |  |
|  | X.rowwise() -= X.colwise().mean(); |
|  | X.array().rowwise() /= X.array().colwise().norm(); |
|  |  |
|  | y = X * w; |
|  |  |
|  | y = y.unaryExpr([](double x) { return x > 0.0 ? 1.0 : 0.0; }); |
|  | } |
|  |  |
|  | void LC::test() { |
|  | std::cout << std::fixed << std::setprecision(3); |
|  |  |
|  | size_t input_dim = 10; |
|  | size_t sample_num = 2000; |
|  |  |
|  | MatrixXd X(sample_num, input_dim); |
|  | MatrixXd y(sample_num, 1); |
|  |  |
|  | gen(X, y); |
|  |  |
|  | ML::DatasetManager dataset; |
|  | dataset.setDataset(X, y); |
|  |  |
|  | ML::Network net; |
|  |  |
|  | net.addLayer(std::make_shared(input_dim, 1)); |
|  | net.addLayer(std::make_shared()); |
|  |  |
|  | net.setLossFunction(std::make_shared()); |
|  | net.setOptimizer(std::make_shared(0.05)); |
|  | net.setDatasetManager(std::make_shared(dataset)); |
|  |  |
|  | size_t epochs = 200; |
|  | size_t batch_size = 25; |
|  | net.train(epochs, batch_size); |
|  |  |
|  | MatrixXd predict(sample_num, 1); |
|  |  |
|  | predict = net.forward(X); |
|  |  |
|  | predict = predict.unaryExpr([](double x) { return x > 0.5 ? 1.0 : 0.0; }); |
|  |  |
|  | MatrixXd error(sample_num, 1); |
|  |  |
|  | error = y - predict; |
|  |  |
|  | error = error.unaryExpr([](double x) {return (x < 0.01 && x>-0.01) ? 1.0 : 0.0; }); |
|  |  |
|  | std::cout << "正确率=\n" << error.sum() / sample_num << "\n"; |
|  | } |


```

**详细解释**


1. `gen`函数：用以生成测试数据。
2. 网络结构：本例的网络结构中包含一个线性层及一个激活函数层，其中：线性层输入尺寸为特征维度，输出尺寸为1。
3. 损失函数：采用对数误差作为损失函数。


**输出展示**
下图反映了网络预测过程中的损失变化，可以看到损失逐渐下降的趋势。
![description](https://img2024.cnblogs.com/blog/3320410/202412/3320410-20241204012842326-1526136944.png)


完成训练后，输出网络的预测结果的正确率。可以发现，网络具有较好的预测精度。
![description](https://img2024.cnblogs.com/blog/3320410/202412/3320410-20241204012901418-1513160188.png)


