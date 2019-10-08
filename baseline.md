# author：段秋阳，张亦钊

## 目的：建立一套自动化系统，接收人声并判断情绪。（bonus：情感语音生成）

## 方法流程：
1. 梅尔倒频谱（mfcc）提取音频特征，作为模型的输入特征。
2. BiLSTM时间序列，或CNN对spectrogram图像进行分类，从而情感分类。

## 使用工具：
1. pytorch torchaudio tensorflow keras
2. librosa

## 参考资料：
1. [https://towardsdatascience.com/speech-emotion-recognition-with-convolution-neural-network-1e6bb7130ce3](https://towardsdatascience.com/speech-emotion-recognition-with-convolution-neural-network-1e6bb7130ce3)
2. [https://mp.weixin.qq.com/s?__biz=MzI2NjkyNDQ3Mw==&mid=2247491361&idx=1&sn=d17ba628792975e2497124d292f68573&chksm=ea87e7f7ddf06ee19eaf8af843d0ab9889ab0b688a1331c027b7a7229aa0f1cab81234661078&scene=0&xtrack=1&key=6d90834972a32f5a2caf22fc167d13564caaf0905b0892813a7a3756c1d25d52d6446f5030b7cfd2936126361304860c26347aa7c2a5398a8758afefe135f045853c9a26543344382a87a6258e445c41&ascene=14&uin=MjMxNDI3NDQ2&devicetype=Windows+10&version=62060834&lang=zh_CN&pass_ticket=VamF2RkimJKpiIT5chpM2XY%2BUrlCnHj2uj7QHDfQO6Y%3D](https://mp.weixin.qq.com/s?__biz=MzI2NjkyNDQ3Mw==&mid=2247491361&idx=1&sn=d17ba628792975e2497124d292f68573&chksm=ea87e7f7ddf06ee19eaf8af843d0ab9889ab0b688a1331c027b7a7229aa0f1cab81234661078&scene=0&xtrack=1&key=6d90834972a32f5a2caf22fc167d13564caaf0905b0892813a7a3756c1d25d52d6446f5030b7cfd2936126361304860c26347aa7c2a5398a8758afefe135f045853c9a26543344382a87a6258e445c41&ascene=14&uin=MjMxNDI3NDQ2&devicetype=Windows+10&version=62060834&lang=zh_CN&pass_ticket=VamF2RkimJKpiIT5chpM2XY%2BUrlCnHj2uj7QHDfQO6Y%3D)
3. [torchaudio documentation](https://pytorch.org/audio/)
4. [MFCC tutorial](http://practicalcryptography.com/miscellaneous/machine-learning/guide-mel-frequency-cepstral-coefficients-mfccs/)