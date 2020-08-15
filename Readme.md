# 基于知识度量的图像多粒度识别

> author：丁磊
>
> time:2020.7.15-2020.7.27
>
> reference: *See Better Before Looking Closer: Weakly Supervised Data Augmentation Network for Fine-Grained Visual Classiﬁcation*

## 一、运行环境

1. Anaconda3-64bit

2. Python 3.7

3. TensorFlow 2.0 GPU版

4. Numpy 1.16.4

5. opencv-python 3.4.7

6. tf-slim 1.2.0

7. matplotlib 3.1

8. IDE：Pycharm

9. PC配置：i7-6700GQ 2.6GHz      GTX 1060

   ![image-20200727160532211](D:\Programming\WS_DAN\image\image-20200727160532211.png)

## 二、run the code

### 2.1 数据集准备及预处理

![image-20200727202457211](D:\Programming\WS_DAN\image\image-20200727202457211.png)

上图为在本机上的原始数据集，因数据集太大，不方便存入压缩包，因此压缩包文件中不含数据集，请见谅。

因CUB-200-2011官方数据集缺失（官网上没找到），本项目采用[Stanford-Cars](https://ai.stanford.edu/~jkrause/cars/car_dataset.html)数据集。数据集下载后，打开`convert_data.py`文件。

```
if __name__ == '__main__':
    datasets_dir = "D:/Programming/WS_DAN/datasets"
    run(datasets_dir)
```

在`convert_data.py`文件底部，找到上述代码块部分，修改`datasets_dir`为本机数据集文件夹，然后run`convert_data.py`。`convert_data.py`文件会自动匹配Car_train,Car_test,devkit文件夹，并在`datasets_dir`文件夹目录下生成`tfrecords`文件夹，然后在该文件夹下生成对应的`*.tfrecord`文件。

运行过程如下图所示：

![image-20200726142128410](D:\Programming\WS_DAN\image\image-20200726142128410.png)

![image-20200726142618501](D:\Programming\WS_DAN\image\image-20200726142618501.png)

当出现如下提示，则转换成功。

![image-20200726143111835](D:\Programming\WS_DAN\image\image-20200726143111835.png)

从下图中可以看到，该文件夹下生成了对应了`*.tfrecord`文件。

![image-20200726143207621](D:\Programming\WS_DAN\image\image-20200726143207621.png)

### 2.2 训练模型

同样的，在`train_model.py`底部找到下面的代码块

```python
if __name__ == '__main__':
    model_root = "D:/Programming/WS_DAN/models"
    datasets_dir = "D:/Programming/WS_DAN/datasets"

    main(model_root= model_root,datasets_dir= datasets_dir)
```

修改`model_root`及`datasets_dir`然后run `train_model.py`文件，进行模型的训练，本项目鉴于本机的性能原因，采用较为简单但有效的VGG16模型进行分类。需要注意的是，在`Model_deploy.py`文件中有相关设备的配置，如下所示，在迁移训练时可能需要进行修改。（该部分，我是直接采用的`TensorFlow`官方给定的代码，没有改动）

```python
_deployment_params = {'num_clones': 1,
                      'clone_on_cpu': False,
                      'replica_id': 0,
                      'num_replicas': 1,
                      'num_ps_tasks': 0,
                      'worker_job_name': 'localhost',
                      'ps_job_name': 'ps'}
```

如果没有问题，run之后应如下图所示：

![img](D:\Programming\WS_DAN\image\YGKVUR5D8IP8C_J9QH0ZD7F.png)

然后等待训练模型，模型会每隔一定时间自动保存，间隔时间设定在`train_model.py`最上方，如下所示：

```python
save_summaries_secs = 60*2
save_interval_secs = 60*4
```

### 2.3 测试

因[Stanford-Cars](https://ai.stanford.edu/~jkrause/cars/car_dataset.html)数据集中，其测试数据集没有label标注，详见其数据集devikit中的readme.txt文件，有如下描述：

> cars_train_annos.mat:
>   Contains the variable 'annotations', which is a struct array of length
>   num_images and where each element has the fields:
>     bbox_x1: Min x-value of the bounding box, in pixels
>     bbox_x2: Max x-value of the bounding box, in pixels
>     bbox_y1: Min y-value of the bounding box, in pixels
>     bbox_y2: Max y-value of the bounding box, in pixels
>     class: Integral id of the class the image belongs to.
>     fname: Filename of the image within the folder of images.
>
> -cars_test_annos.mat:
>   Same format as 'cars_train_annos.mat', except the class is not provided.

因此本项目中打算，以实际读入图片输出结果为测试结果，但事与愿违，虽然可以从训练集中读取label，但，对于mat文件来说，其属性名无法读取。因此项目缺少一个label到class name的一一对应，这一点，待后续解决。

可能的解决办法如下：找到一个可以直接读取.mat文件属性名的函数，直接通过该函数在`convert_data.py`文件下，generate_datasets（）函数修改为如下代码：(假定可读取.mat文件属性名的函数为get_the_class_name)

```python
def generate_datasets(data_root):
    train_info = sio.loadmat(os.path.join(data_root, 'devkit', 'cars_train_annos.mat'))['annotations'][0]
    test_info = sio.loadmat(os.path.join(data_root, 'devkit', 'cars_test_annos.mat'))['annotations'][0]

    train_dataset = []
    test_dataset = []
    label_to_class = []



    for index in range(len(train_info)):
        images_file = str(train_info['fname'][index][0])
        label = train_info['class'][index][0][0] - 1

        labels_to_classes = {}
        labels_to_classes[label] = get_the_class_name(train_info['class'])

        example = {}
        example['filename'] = os.path.join(data_root, 'cars_train', images_file)
        example['label'] = int(label)
        train_dataset.append(example)
        label_to_class.append(labels_to_classes)

    for index in range(len(test_info)):
        images_file = str(test_info['fname'][index][0])
        label = 1

        example = {}
        example['filename'] = os.path.join(data_root, 'cars_test', images_file)
        example['label'] = int(label)
        test_dataset.append(example)
    
    if( not has_labels(data_root)):
        for i in len(label_to_class):
            write_label_file(label_to_class[i])
    
    return train_dataset, test_dataset
```

另：

Test_model.py中，main()表示对有标签批注的测试集的测试函数，而predict（）函数表示对单张图片进行预测的结果。

![image-20200727202200704](D:\Programming\WS_DAN\image\image-20200727202200704.png)

上图为单张图片的测试结果，因没有class名，因此无法对应。

### 2.4 可视化

利用`tensorboard`打开日志文件即可查看,命令如下：

```base
tensorboard --logdir=/path/to/model_dir --port=8081
```

## 三、其他

### 3.1 关于模型搭建

由于本机基础环境为TensorFlow2.0，并没有安装TensorFlow1.x版本，因此本项目的基础代码以TensorFlow2.0为主，但由于上次项目中，运用Keras.fit过程中遇到了大量与[官方API](https://tensorflow.google.cn/versions/r2.0/api_docs/python/tf)库中描述不符的问题，由于这次时间较短，担心无法debug完或训练有差错，因此本项目中采用tf-slim与tf.compat.v1的兼容性代码搭建模型。

### 3.2 关于基于细粒度的图像识别方法研究

因时间有限，我只读了3篇相关算法文献，并从中选择了一篇作为复现参考，关于3篇文献及该领域的认识与了解，详见文件《文献综述》。

PS:由于文献数量过少，综述格式有些不当，敬请见谅。

### 3.3 关于debug

本项目中遇到的所有问题，均采用阅读源码，官方API，及通过CSDN\Stackflow解决，因此代码可能会有些不标准或冗余，请见谅。

### 3.4 关于设备

原本预想本人电脑VGG16应该能正常训练，但我也不知道为什么运行就是内存超了，因此程序训练及测试是在远程PC上做的。运行日志中设备变换的原因就是如此。

### 3.5 个人收获

在本次项目开发过程中，我首先是第一次了解到了什么是细粒度识别，了解到了关于细粒度识别算法的大概热度方向，也熟悉了一种基于注意力机制的细粒度识别算法。同时，由于使用本就不太熟悉的TF-Slim来架构模型，这使我更加充分认识到了，TF2.0的便捷性，也更清晰地认识到了TF1.x中的各种不便，当时初学时对代码的一知半解也逐渐减弱，通过之前积累的大量源码阅读及本次项目过程中的源码阅读及单步跟踪，我对于Tensorflow的了解又更进了一步，例如，在编写测试代码时，遇到sess.run（）过程中报错，便通过单步跟踪，了解了run的运行流程。

但同样的，目前TensorFlow发展还不够完善，遇到许多问题通过搜索引擎没有办法解决，而通过单步调试，又过于浪费时间，这对于大部分人来说是个棘手的问题，这可能也是很多人选择pytorch的原因之一吧。

### 3.6 关于个人

博客：https://beater.blog.csdn.net/

github: https://github.com/KCSoftDL?tab=repositories

email: 784194906@qq.com

