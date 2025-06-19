# B.S. thesis "An Implementation of GLADNet+CBAM Network for Low-light Image Enhancement"

![image](https://github.com/user-attachments/assets/f33eef14-7afc-4803-9d5d-4d8dcb3b4db1)

## Train
```shell
bash train.sh # for gladnet+cbam
bash train_gladnet.sh # for gladnet
```

## Test
```shell
# plz denote the location of test files in test/test_gladnet.sh
bash test.sh # for gladnet+cbam
bash test_gladnet.sh # for gladnet
```

## Reference
[1] Wang, W., Wei, C., Yang, W., & Liu, J. (2018, May). Gladnet: Low-light enhancement network with global awareness. In 2018 13th IEEE international conference on automatic face & gesture recognition (FG 2018) (pp. 751-755). IEEE. <br>
[2] Woo, S., Park, J., Lee, J. Y., & Kweon, I. S. (2018). Cbam: Convolutional block attention module. In Proceedings of the European conference on computer vision (ECCV) (pp. 3-19). <br>

## Acknowledgement
Thanks for dataset and great code: <br>
[GLADNet](https://github.com/weichen582/GLADNet). <br>
[GLADNet 2.0](https://github.com/abhishek-choudharys/GLADNet). <br>
