## Ưu tiên theo hướng xử lý như trong bài báo đề xuất đã đưa, đặc biệt là sử dụng thử hàm từ [28]
- [ ] Đọc lại bài báo, chú ý references:
  - [27 - A new deep learning
  architecture for detection of long linear infrastructure](http://www.mva-org.jp/Proceedings/2017USB/papers/06-05.pdf) 
  - [28 - A novel focal
  phi loss for power line segmentation with auxiliary classifier u-net](https://sci-hub.se/10.3390/s21082803)

## Model Loại 2 (SecondModel): Cùng một kiến trúc, nhưng training sử dụng Jaccard Loss
- [ ] Tìm hiều về IoU loss, và cách người ta làm cho chúng trở nên `differentiable`. 
- [ ] Link để ngồi đọc: 
  - [StackOverflow discussion](https://stackoverflow.com/questions/40475246/why-does-one-not-use-iou-for-training)
  - [Paper](https://arxiv.org/pdf/1608.01471.pdf)
  - [Pytorch's Implementation](https://github.com/kevinzakka/pytorch-goodies/blob/master/losses.py)

## TASKS
- [ ] Convert the `Readme.md` into English version
- [ ] Implement the loss function that the author has suggest in the `PLGan`, compare the result with the one the author reported
- [ ] These tasks will be completed on this weekend (22-23 April)

## MORE APPROACHES
- [DUFormer : A Novel Architecture for Power Line Segmentation of Aerial Images](https://arxiv.org/pdf/2304.05821.pdf)
- [Automatic High Resolution Wire Segmentation and Removal](https://arxiv.org/pdf/2304.00221.pdf)