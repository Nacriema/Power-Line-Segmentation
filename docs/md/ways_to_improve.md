# Imrpovements model

Currently, I'm not use the loss function as suggested in the original paper. So let's try with the loss that the author in paper [1] has suggest

## High priority with the idea from [1] at reference [28]
- Reread [1] paper, notice at references:
  - [27 - A new deep learning
  architecture for detection of long linear infrastructure](http://www.mva-org.jp/Proceedings/2017USB/papers/06-05.pdf) 
  - [28 - A novel focal
  phi loss for power line segmentation with auxiliary classifier u-net](https://sci-hub.se/10.3390/s21082803)

## Implement and train the model with Jaccard Loss
- [ ] Research on `IoU loss` and the way people make it become `differentiable` when apply as the loss function.
- [ ] Links to read: 
  - [StackOverflow discussion - Why people does not use the IoU for training model](https://stackoverflow.com/questions/40475246/why-does-one-not-use-iou-for-training)
  - [Paper](https://arxiv.org/pdf/1608.01471.pdf)
  - [Pytorch's Implementation](https://github.com/kevinzakka/pytorch-goodies/blob/master/losses.py)


## Papers
- [[1] PLGAN: Generative Adversarial Networks for Power-Line Segmentation in Aerial Images
](https://arxiv.org/abs/2204.07243)
- [[2] DUFormer : A Novel Architecture for Power Line Segmentation of Aerial Images](https://arxiv.org/pdf/2304.05821.pdf)
- [[3] Automatic High Resolution Wire Segmentation and Removal](https://arxiv.org/pdf/2304.00221.pdf)