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



- [ ] Tìm mối liên hệ giữa hình ảnh và json format. Sau đó tạo ground truth từ chúng xem sao !!! Gợi ý sử dụng file từ scripts á. 

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