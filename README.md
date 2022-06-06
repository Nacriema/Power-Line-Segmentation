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


- [ ] Tách hình ảnh thành kiểu cấu trúc đơn giản hơn (có thể là tương tự như là đối với DocExtractor vậy, để ta có thể tái sử dụng lại được cái loaddata của nó)



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