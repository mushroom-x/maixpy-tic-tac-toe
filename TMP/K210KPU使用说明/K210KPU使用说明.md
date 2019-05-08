

## KPU使用小贴士

* 目前KPU驱动支持的算子在kpu.h中有描述，如果在构建模型中请避免使用在支持算子之外的算子
* 特征图宽度为64倍数时，能够让KPU完全发挥算力（KPU是以64像素宽为单位计算）
* 3×3卷积最能发挥KPU算力，1×1卷积或者depth wise 卷积效率只有3×3卷积的1/8~1/9
* 详细KPU图示见：http://dl.sipeed.com/MAIX/SDK/Document/KPU%20diagram.pdf

## KPU支持的算子列表





## Reference

[TFLite](https://www.tensorflow.org/lite)

[MAIX KPU揭秘：手写你的第一层网络](http://blog.sipeed.com/p/367.html)

[30分钟训练，转换，运行MNIST于MAIX开发板](http://blog.sipeed.com/p/518.html)

