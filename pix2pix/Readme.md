# pix2pix

*  model 
   其中使用 U-net(resnet-block)，來增加生成的資訊，還有 patch discriminator ， patch discriminator 的作用在於 增加 strides ，作者的意思是在這conv layer的安排下，最後的feature map 的每個像素能對應原圖中的 70 * 70 的大小 ， 而這70 * 70 會有許多重疊 ，參數量也大大的降低。
![](https://github.com/citya1472581234/GAN/blob/master/pix2pix/result/model.jpg?raw=true)

* result 

![](https://github.com/citya1472581234/GAN/blob/master/pix2pix/result/cmp_b0252.jpg?raw=true)
![](https://github.com/citya1472581234/GAN/blob/master/pix2pix/result/cmp_b0252%20(2).jpg?raw=true)
