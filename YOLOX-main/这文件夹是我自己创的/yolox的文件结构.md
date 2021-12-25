### tools/train.py

* `args,exp`解析参数
* 在`main`里创建了一个`Trainer(exp, args)`，并且使用了`trainer.train()`
* 接下来要找`Trainer`的定义

### yolox/core/trainer.py

* 找到`Trainer(33)`的定义
* `train(69)`先执行的是`before_train()`
* `before_train(125)`定义模型是`exp.get_model()`
* `yolox/exp/yolox_base.py`里的`get_model(73)`，里面很明确了使用的model就是YOLOPAFPN和YOLOXHead`self.model=YOLOX(backbone, head)(90)`
* 接下来要找`YOLOX`的定义

### yolox/models/yolox.py

* 找到`YOLOX(11)`类的定义，上面已经说了使用的网络是YOLOPAFPN和YOLOXHead
* 这个文件的`forward(28)`写的非常简洁，就是`fpn_outs = self.backbone(x)`和`outputs = self.head(fpn_outs)`
* 其中PAFPN的部分不需要我们修改，那个就是输出三层结果out3, out4, out5，然后上采样再下采样，搭一个UNet的那玩意
* 我们要改的是后面的head，接下来要找`YOLOXHead`的定义

### yolox/models/yolo_head.py

* 找到`YOLOXHead(18)`的定义
* 它对于每个grid只标一个框，`self.n_anchors = 1(35)`
* 我们需要关注的是它classification的部分
    * `self.cls_convs(39)`, `self.cls_preds(41)`
    * `self.cls_convs.append(57)`, `self.cls_preds.append(97)`
    * `cls_output(158)`
    * `output = torch.cat([..., cls_output, ...])(165,188)`
    * 计算loss：`cls_preds = outputs[:,:,5:](266)`, `loss_cls(393)`
    * `loss = ... + loss_cls + ...(406)`
    * 它的返回值里有这个总loss，也有专门针对每一类的loss
    * 这个返回值在`self.get_losses(195)`里又被作为forward函数的返回值了
    * 至此，可以确定trainer.py文件里的`outputs = self.model(inps, targets)(101)`返回的是一堆loss
* 确定我们是直接改它的文件，还是重新定义一个网络结构自己train，我选择第一种。并且把我改的部分作为分支插入它代码里
* 现在需要找到传参时模型的各参数是什么，以及是如何控制分支的。打个比方，我想给模型加一个新参数`PFOD`，当它为True的时候表示使用我给的PFOD方法，当它为False的时候使用yolox自己的训练方法
* 有个令人欣喜的发现，`get_losses`里的266行，阿冰说它的bounding box, classification, confidence三个东西是拼接在一起的，确实是这样。不过它比想象中的简单，它不是三部分拼成一个长向量，而是竖着按层数堆叠起来的，第0~3层是bbox，第4层是conf，5层以后是20类的分类结果。所以要去掉分类，只需要取前4层即可

### 要加一个开关is_PFOD
* 它控制是否使用PFOD的yolox做训练，这个开关将被从顶层一直传递到yolo head的实现
* `train.py(93)`里加了`parser.add_argument("-pfod",...)`
* `train.py(129)`里加了`exp.merge(args.is_PFOD)`直接暴力往exp里添加这个`is_PFOD`参数
* `yolo_head.py(40)`的初始化函数里加了`self.is_PFOD`
* `yolo_head.py(400)`加了个分支，在PFOD模式下loss_cls是0，别忘了最头上`import numpy`
* `trainer.py(59)`加了个`self.is_PFOD`
* `trainer.py(84)`加了个分支，如果当前到200轮了，就把未标出的unk样本扔了

### yolox/core/trainer.py
* `self.train_loader(151)`这个东西应该是读数据集的，如果你要实现每个stage都重新读数据集，那么每个stage都显式声明这一行


