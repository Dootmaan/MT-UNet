# MT-UNet

## Update 2022/03/05

The paper has been accepted by ICASSP 2022. The complete code is released today. 

Please note that, if you have requested our code before, that code is depreciated right now and you are encouraged to use the newest version in this repo.

---

**1. Prepare your dataset.**

- Synapse dataset can be found at [the repo of TransUnet](https://github.com/Beckschen/TransUNet). 

- ACDC dataset is a little complicated, since we found that previous works uses different partition but compares with each other directly. To make sure our experiment is more fair, we uses our own partition and rerun all the methods by ourselves. We have uploaded the preprocessed ACDC dataset [here](https://drive.google.com/file/d/13qYHNIWTIBzwyFgScORL2RFd002vrPF2/view?usp=sharing), or you can download the dataset by yourself at [here](https://acdc.creatis.insa-lyon.fr/description/databases.html) 

**2. Clone the code**

- First, clone our code with:
```
git clone git@github.com:Dootmaan/MT-UNet.git
cd MT-UNet
```

- Then, modify "train_mtunet_ACDC.py" and "train_mtunet_Synapse.py" according to your experiment environment. You can search for "path/to/dataset" in these two files and replace them with the real path to the ACDC and Synapse dataset on your machine.

**3. Start training**

- After that, you can start training with:
```
CUDA_VISIBLE_DEVICES=0 nohup python3 -u train_mtunet_ACDC.py >train_mtunet_ACDC.log 2>&1 &
```

The weights will be saved to "./checkpoint/ACDC/mtunet" while the predictions will be saved to "./predictions" by default. You can also load our weights before training with the argparser. 

[ACDC weights](https://drive.google.com/file/d/1eo6d-d_kR0qbHBIHq49TQ1CFpPLypJUT/view?usp=sharing)

[Synapse weights](https://drive.google.com/file/d/1frQAK05UtiAO8rvKG9y5GXABaH70_-Hu/view?usp=sharing)

---

We have tested the code to make sure it works. However, if you still find some bugs, feel free to make a pull request or simply raise an issue.

*You are also encouraged to read the update log below to know more about this repo.*

## Update 2022/01/05

By another round of training based on previous weights, our model also achieved a better performance on ACDC (91.61% DSC). We have changed the weights for ACDC to this newest version and you can check it out for yourself. However, previous versions of weights are still available on Google Drive, and you can access them via previous commits. 

## Update 2022/01/04

We have further trained our MT-UNet and it turns out to have a better result on Synapse with 79.20% DSC. We have changed the public weights of Synapse to this version and will also update the results in our paper.

## Update 2022/01/03

It should be mentioned that we are currently conducting some statistical evaluations on our model and these results will be also made public on this site.

- **[Updated]** Click [here](https://drive.google.com/file/d/1frQAK05UtiAO8rvKG9y5GXABaH70_-Hu/view?usp=sharing) for our weights used on Synapse. 

- **[Updated]** Click [here](https://drive.google.com/file/d/1eo6d-d_kR0qbHBIHq49TQ1CFpPLypJUT/view?usp=sharing) for our weights used on ACDC. The authors of TransUnet did not provide the split of ACDC dataset. Therefore, we conducted all the ACDC experiments based on our own dataset split.

## Update 2021/11/19

- Thank you for your interest in our work. We have uploaded the code of our MTUNet to help peers conduct further research on it. However, rest of the codes (such as the training and testing codes) are currently not so well organized, and we plan to release them upon paper publication. It also should be noted that they are still avaliable right now with a rough appearance. Please contact us for these codes if you are new to this field or having difficulty in applying our model to your own dataset.

---

This is the official implementation for our ICASSP2022 paper *MIXED TRANSFORMER UNET FOR MEDICAL IMAGE SEGMENTATION*

The entire code will be released upon paper publication.