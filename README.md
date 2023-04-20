# Mục đích

Cấu trúc thư mục hiện tại mình đang có: 

```text
data_sample/
    im1.png
    im1.json

splitting_datatset_txt/
    test.txt 
    train.txt
    val.txt

scripts/
    xxxs.py
```

Trong đó data_sample sẽ chứa hình ảnh và annotation ở dưới dạng json. 

```text
File: 
    test.txt: 220 records
    train.txt: 905 records
    val.txt: 109 records 

Sum: 1234 records

Folder data_original:
    2484 records: images + json annotations => 1242 records

Redundant: 1242 - 1234 = 8 images
```



- [x] Tìm mối liên hệ giữa hình ảnh và json format. Sau đó tạo ground truth từ chúng xem sao !!! Gợi ý sử dụng file từ scripts á. 

Kết quả đạt được: 

| Original                                        | Sample                                                     | Mask (Power lines)                                        | 
|-------------------------------------------------|------------------------------------------------------------|-----------------------------------------------------------|
| ![](./TTPLA_Processing/data_sample/04_585.jpg)  | ![](./TTPLA_Processing/processing_code/04_585_Sample.png)  | ![](./TTPLA_Processing/processing_code/04_585_Label.png)  | 
| ![](./TTPLA_Processing/data_sample/04_2220.jpg) | ![](./TTPLA_Processing/processing_code/04_2220_Sample.png) | ![](./TTPLA_Processing/processing_code/04_2220_Label.png) |

- [x] Kiểm tra label tạo ra được có đúng chuẩn hay không ? Tức là đọc dữ liệu ra là được bao nhiêu, mình expect là 0, 1. Thực sự thì dữ liệu đọc ra được
là RGB value với line: (255, 255, 255), non-line: (0, 0, 0). Và thông qua phương thức map từ màu sang class trong DataSet của docExtractor thì sẽ chuyển sang
map dạng 0, 1. 

- [x] Tách hình ảnh thành kiểu cấu trúc đơn giản hơn (có thể là tương tự như là đối với DocExtractor vậy, để ta có thể tái sử dụng lại được cái loaddata của nó)


Cấu trúc dữ liệu train của DocExtractor: 

```text
DataSet
    test   
    train 
        seq1_xxxx_Labels.png
        seq1_xxxx.png
    val
        seq16_xxxx_Labels.png
        seq16_xxxx.png
```

Cấu trúc dữ liệu mới được tạo ra từ TTPLA dataset
```text
TTPLA_PreprocesseData
    test 
        04_1234_Labels.png
        04_1234.jpg
    train 
        04_3456_Labels.png
        04_3456.jpg
    val
        04_789_Labels.png
        04_789.jpg
```

- [x] Kiểm tra số lượng hình ảnh tạo ra được, so khớp với số record ở trong mỗi file annotate về json
```text
Test: 440 images -> 220 records
Train: 1810 images -> 905 records
Val: 218 images -> 109 records
```

Vậy là trong dữ liệu dư thừa ra 8 bản ghi không thuộc loại nào, bỏ qua mấy cái đó 

- [x] Xây dựng mô hình cho cái thuật toán PowerLine segmentation. Tìm kiếm model thực hiện tốt điều này. Mình sẽ chọn 
UNET++ (backbone Resnet-34) 

![img.png](docs/images/img.png)

- [x] Xây dựng DataLoader cho segmentation task, kiểm tra chúng.
- [x] Viết script train model.
- [x] Sau khi đẩy dữ liệu full lên trên Drive, thì mình tính toán giá trị trung bình mean và std của toàn bộ dữ liệu trên đó thử:
Kết quả sẽ được note ở đây: 

![img_1.png](docs/images/img_1.png)

> Kết quả: 
> * Mean: tensor([0.4616, 0.4506, 0.4154]) 
> * Std: tensor([0.2368, 0.2339, 0.2415])

Tuy nhiên, ta cũng nên thử mean và std từ ImageNet, do ta tái sử dụng lại Resnet pretrained: 
> * Mean: [0.485, 0.456, 0.406]
> * Std: [0.229, 0.224, 0.225]

- [x] Suy nghĩ cách customize hàm `save_checkpoint(self, val_loss, model)` của thằng `EarlyStopping` sao cho nó sử dụng tất 
cả các trường như thằng `docExtractor` vậy:

- [x] Sử dụng `earlystoping` để có thể save the best model

> Ý tưởng của eary stopping: 
> 
> * B1: Trước khi train model, khởi tạo `early_stopping = EarlyStopping()`
> * B2: Tại mỗi check point mà `val_stat_interval` gây ra, tính toán `validation loss`, sau đó record nó vào `early_stopping` bằng cách `early_stopping(valid_loss, model)`
> * B3: Ngay sau khi thực hiện bước 2, kiểm tra xem thử cái trạng thái của `early_stopping` nó như thế nào `if early_stopping.early_stop:`, nếu 
> nó báo cần kết thúc thì ta kết thúc và break vòng lặp train.

- [x] Kiểm tra mật độ phân phối dữ liệu trên toàn bộ data set, dữ liệu của mình là cực kỳ `imbalance`: 
![img_2.png](docs/images/img_2.png)
```text
RESULT: tensor([820755889,  17900111])
WEIGHT (1./RESULT): tensor([1.2184e-09, 5.5866e-08]) 
```
- [x] Test thử việc truyền weight class vào trong `CrossEntropyLoss`, kết quả có vẻ khả quan đấy !
![img_3.png](docs/images/img_3.png)
- [ ] Hiệu chỉnh code lần cuối cùng rồi đẩy code hoàn chỉnh lên trên đó, sau đó thực hiện train model với full-data
  - [x] Thêm phần resume training 
  - [x] Thêm vào logging, cải thiện format logging sao cho nó dễ đọc nhất có thể (Gợi ý sử dụng kiểu log message của PyImage Search)
    - Loging bao gồm 2 phần: `Terminal log` và `File log` 
    - Đối với `Terminal log`, mình sẽ sử dụng trick để điều chỉnh log (Bằng cách thêm các chỉ màu đặc biệt vào trước mỗi log string)
    - Đối với `File log`, mình sẽ sứ dụng `logging` của python. 
  - [x] Thêm vào trường `data_distribution` để normalize dữ liệu
  - [ ] Thêm vào `Tensorboard logs`: 
    - [x] Xem xét được đường cong huấn luyện `train_loss` và `validation_loss`
    - [ ] Xem xét được performance của model tại điểm `validation` bằng cách xem qua ảnh dự đoán của model trên tensorboard (Cái ý này có vẻ dễ thực hiện hơn ý đầu đấy)
  - [x] Viết `tester.py` script: script này chịu trách nhiệm test performance của model trên tập ảnh test sau khi hoàn tất quá trình train. 
  - [x] Tái tổ chức lại cấu trúc file sao cho hợp lý nhất (Bao gồm cả cấu trúc code, import bla...)
    - [x] Vấn đề lúc `save_state_dict`, mình muốn lưu thêm trường `val_loss_min` để lúc pretrain lại sử dụng nó để khởi 
tạo biến record bên trong `EaryStopping`.

  - [x] Viết thêm `.gitignore` để tránh trường hợp đẩy file có dung lượng lớn lên lên trên Github. 
  - [ ] Tìm hiểu thêm một tính năng nào đó hay ho hỗ trợ code từ github (Ví dụ: Bot, CodeCoverage, ... ) để cái thiện code. (Đã thực hiện với CodeCoverage,...)
    - [x] Đã biết cách sử dụng `Code Coverage` để kiểm tra độ bao phủ của code trong một lần chạy fullflow. 
        > ![img.png](docs/images/img_4.png)
  - [x] Viết thêm `ArgumentParser` vào `trainer.py` script
- [x] Thực hiện một số tính toán đơn giản dựa trên bộ dữ liệu để có thể có được cấu hình ok ban đầu cho file config: 
  > Mình đã note sẵn trong file `first_try.yml` 
- [ ] Vấn đề khó ở đây đó là handle cái vụ về model này, ta thấy rất khó khăn để mô hình học đc, nguyên nhân là do việc 
không cân đối giữa các lớp với nhau. Đề xuất những hướng giải quyết: 
  - [ ] Sau khi train xong với `CrossEntropyLoss`, sử dụng `fine-tune` bằng `LovasLoss`. 
  - [ ] Sử dụng `FocalLoss` với `class weight`, `SoftDiceLoss`, hoặc là `JaccardLoss (IoU Loss)` 
- [ ] Sử dụng `Optima` để tìm được bộ tham số khởi tạo chuẩn khi huấn luyện mô hình.
- [ ] Thực hiện train trên Colab, chú ý những điều sau:
  - [x] Mỗi lần chạy lại train ta sẽ mất hết dữ liệu bên trong `train_metrics.tsv` và `val_metrics.tsv`.
  - [x] **Mình cần ghi lại giá trị val_loss nhỏ nhất trước đó đã lưu được tại best-save, thiết kế lại load check point và thêm việc truyền tham số vào khi khởi tạo EarlyStopping instance**. (Ok, vấn đề đã được giải quyết !)
    > ![img.png](docs/images/img_5.png)
  - [ ] Tiếp tục train trên colab !
  - [ ] Vấn đề hiện tại với model của mình là train hoài mà nó không xuống được nữa, mặc dù nó đang làm khá tốt. Mình nên 
cân nhắc việc `FineTune` với cái `REDUCELRONPLATEAU` của Pytorch để tự động điều chỉnh giá trị learning rate khi mà thấy không ổn. 

- [ ] Làm cho quá trình train trở nên `deterministic`. (Cái này chắc chắn phải để cuối cùng, vì nó cần phải tìm hiểu thêm 
nhiều thứ lắm mới có thể làm được)

## Update kết quả

### Lần 1 + Lần 2 (ngày 24/06/2022)
- [x] Model được tải về máy và lưu ở thư mục `FirsModel_2`

![img.png](docs/images/img_6.png)

- [ ] Chạy Testing ở trên `Colab` đồng thời chạy `Custom dataset` của mình để xem nó thể hiện như thế nào. So sánh ở 2 tiêu chí: IoU ở tập test như thế nào 

- [ ] Thể hiện của nó ở tập test và tập custom như thế nào, để so sánh ra bằng hình ảnh:

`So sánh trên tập Custom Dataset` 

| FirstModel                                                                                        | FirstModel_2                                                                                        |
|---------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------|
| ![](./UNet++/models/FirstModel/test_RealData/blend/im_2.jpg)                                      | ![](./UNet++/models/FirstModel_2/test_RealData/blend/im_2.jpg)                                      |        
| ![](./UNet++/models/FirstModel/test_RealData/blend/camera-kbvision-ptz-2022-02-25175000000Z.jpg)  | ![](./UNet++/models/FirstModel_2/test_RealData/blend/camera-kbvision-ptz-2022-02-25T175000000Z.jpg) |
| ![](./UNet++/models/FirstModel/test_RealData/blend/camera-kbvision-ptz-2022-03-01T100000000Z.jpg) | ![](./UNet++/models/FirstModel_2/test_RealData/blend/camera-kbvision-ptz-2022-03-01T100000000Z.jpg) |
| ![](./UNet++/models/FirstModel/test_RealData/blend/camera-kbvision-ptz-2022-03-01T101000000Z.jpg) | ![](./UNet++/models/FirstModel_2/test_RealData/blend/camera-kbvision-ptz-2022-03-01T101000000Z.jpg) |
| ![](./UNet++/models/FirstModel/test_RealData/blend/camera-kbvision-ptz-2022-03-01T102000000Z.jpg) | ![](./UNet++/models/FirstModel_2/test_RealData/blend/camera-kbvision-ptz-2022-03-01T102000000Z.jpg) |
| ![](./UNet++/models/FirstModel/test_RealData/blend/im_4.jpg)                                      | ![](./UNet++/models/FirstModel_2/test_RealData/blend/im_4.jpg)                                      |
| ![](./UNet++/models/FirstModel/test_RealData/blend/KonTum_QuangNgai.jpg)                          | ![](./UNet++/models/FirstModel_2/test_RealData/blend/KonTum_QuangNgai.jpg)                          |
| ![](./UNet++/models/FirstModel/test_RealData/blend/KonTum_ThanhMy.jpg)                            | ![](./UNet++/models/FirstModel_2/test_RealData/blend/KonTum_ThanhMy.jpg)                            |
| ![](./UNet++/models/FirstModel/test_RealData/blend/ngu_hanh_son_2.jpg)                            | ![](./UNet++/models/FirstModel_2/test_RealData/blend/ngu_hanh_son_2.jpg)                            |
| ![](./UNet++/models/FirstModel/test_RealData/blend/sample6.jpg)                                   | ![](./UNet++/models/FirstModel_2/test_RealData/blend/sample6.jpg)                                   |
| ![](./UNet++/models/FirstModel/test_RealData/blend/sample8.jpg)                                   | ![](./UNet++/models/FirstModel_2/test_RealData/blend/sample8.jpg)                                   |
| ![](./UNet++/models/FirstModel/test_RealData/blend/sample9.jpg)                                   | ![](./UNet++/models/FirstModel_2/test_RealData/blend/sample9.jpg)                                   |
| ![](./UNet++/models/FirstModel/test_RealData/blend/sample11.jpg)                                  | ![](./UNet++/models/FirstModel_2/test_RealData/blend/sample11.jpg)                                  |
| ![](./UNet++/models/FirstModel/test_RealData/blend/SunCoast-powerline-2.jpg)                      | ![](./UNet++/models/FirstModel_2/test_RealData/blend/SunCoast-powerline-2.jpg)                      |

### Lần 3 (27/06/2022)
- [x] Lên Drive tải bản Lần 3 này về, clone thành một bản để backup dữ liệu. 
- [x] Sau đó mới chuyển sang mục Lần 4.
- [ ] Chạy ở trên local này để đánh giá thử model Lần 3 này như thế nào (Ây da, cái này có vẻ không tốt cho lắm đâu, theo 
cảm quan của mình thấy được là như vậy, nhưng để chắc chắn hơn thì phải chạy qua toàn bộ ảnh test để coi metric đạt được
là bao nhiêu cái đã)

* Test metric của lần 2

![img.png](docs/images/img_8.png)

* Test metric của lần 3 

![img.png](docs/images/img_7.png)

**Vậy là lần 2 tốt hơn lần 3 á !!**


### Lần 4

Sau khi sửa xong chỗ update bên trong code. Mình cần phải tải cái lần 3 về, lưu thành một bản copy nữa. Sau đó sửa 2 chỗ:
- [x] Sửa trong file config trên Drive thành 17 
- [x] Ở local mình dùng thằng `test_torchload.py` để sửa cái `milestones` của thằng Scheduler lại thành con 17 luôn.
- [x] Bỏ lại lên trên Drive và tiếp tục train xem thử kết quả nó như thế nào 
- [x] Fail rồi, mô hình nó vẫn không chịu cập nhật learning rate khi train nữa ... 
- [ ] Thử sử dụng `LovasLoss` để tiếp tục train xem sao ! Nếu không được nữa thì phải chuyển sang hàm mục tiêu mới !!!

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

Have a nice day !!!
