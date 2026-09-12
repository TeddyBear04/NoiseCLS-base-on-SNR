# Công thức Threshold, BCE Loss và Class Weighting

## 1. Ký hiệu và xác suất đầu ra

Với mẫu thứ $i$ và lớp thứ $c$:

- $x_{i,c}$: logit do classification head của BEATs sinh ra.
- $y_{i,c} \in \{0,1\}$: nhãn ground truth.
- $p_{i,c}$: xác suất dự đoán.
- $N$: số mẫu train, hiện tại $N=18.231$.
- $C=21$: số lớp.
- $n_c$: số mẫu train có nhãn dương của lớp $c$.

Xác suất của mỗi lớp được tính độc lập bằng sigmoid:

$$
p_{i,c}=\sigma(x_{i,c})=\frac{1}{1+e^{-x_{i,c}}}.
$$

Sigmoid cho phép một mẫu nhận đồng thời nhiều nhãn, phù hợp với bài toán
multi-label.

## 2. Binary Cross-Entropy Loss

BCE của một mẫu và một lớp là:

$$
\ell_{i,c}=-\left[y_{i,c}\log(p_{i,c})
+(1-y_{i,c})\log(1-p_{i,c})\right].
$$

Loss trung bình trên batch gồm $B$ mẫu là:

$$
\mathcal{L}_{\mathrm{BCE}}
=\frac{1}{BC}\sum_{i=1}^{B}\sum_{c=1}^{C}\ell_{i,c}.
$$

Mã nguồn sử dụng `BCEWithLogitsLoss`, tính trực tiếp từ logit để ổn định số học
và tránh lỗi `log(0)` khi xác suất quá gần 0 hoặc 1.

## 3. Class Weighting bằng `pos_weight`

Positive weight thô của lớp $c$ được tính từ tập train:

$$
w_c^{\mathrm{raw}}=\frac{N-n_c}{\max(n_c,1)}.
$$

Trọng số được giới hạn trong khoảng $[1,20]$:

$$
w_c=\min\left(20,\max\left(1,w_c^{\mathrm{raw}}\right)\right).
$$

Weighted BCE của mô hình là:

$$
\mathcal{L}
=-\frac{1}{BC}\sum_{i=1}^{B}\sum_{c=1}^{C}
\left[
w_c y_{i,c}\log\sigma(x_{i,c})
+(1-y_{i,c})\log\left(1-\sigma(x_{i,c})\right)
\right].
$$

Viết riêng theo nhãn:

$$
\ell_{i,c}=
\begin{cases}
-w_c\log(p_{i,c}), & y_{i,c}=1,\\
-\log(1-p_{i,c}), & y_{i,c}=0.
\end{cases}
$$

`pos_weight` chỉ nhân vào thành phần nhãn dương. Lớp hiếm có $w_c$ lớn hơn nên
việc bỏ sót lớp đó bị phạt mạnh hơn. Cách này thường tăng recall nhưng cũng có
thể tăng false positive, vì vậy cần hiệu chỉnh threshold sau huấn luyện.

Mã triển khai:

```python
counts = class_counts(train_dataset)
pos_weight = ((len(train_dataset) - counts) / counts.clamp_min(1)).clamp(
    1.0, 20.0
)
criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
```

## 4. Công thức dự đoán bằng threshold

Với threshold $\tau_c$ của lớp $c$:

$$
\hat{y}_{i,c}=\mathbb{1}\left[p_{i,c}\geq\tau_c\right]
=
\begin{cases}
1, & p_{i,c}\geq\tau_c,\\
0, & p_{i,c}<\tau_c.
\end{cases}
$$

Trong đó $\mathbb{1}[\cdot]$ là hàm chỉ thị. Một mẫu có thể vượt threshold của
nhiều lớp và nhận nhiều nhãn.

## 5. Chọn threshold tối ưu cho từng lớp

Threshold chỉ được tìm trên validation. Với mỗi lớp, chương trình thử:

$$
\mathcal{T}=\{0{,}05,0{,}06,\ldots,0{,}95\}.
$$

Tại mỗi $t\in\mathcal{T}$:

$$
TP_c(t)=\sum_i\mathbb{1}[y_{i,c}=1\land p_{i,c}\geq t],
$$

$$
FP_c(t)=\sum_i\mathbb{1}[y_{i,c}=0\land p_{i,c}\geq t],
$$

$$
FN_c(t)=\sum_i\mathbb{1}[y_{i,c}=1\land p_{i,c}<t].
$$

F1 của lớp $c$ tại threshold $t$:

$$
F1_c(t)=\frac{2TP_c(t)}{2TP_c(t)+FP_c(t)+FN_c(t)}.
$$

Threshold tối ưu của lớp $c$ là:

$$
\tau_c^*=\underset{t\in\mathcal{T}}{\operatorname{argmax}}\;F1_c(t).
$$

Threshold chung được chọn bằng cách tối đa hóa macro-F1:

$$
\tau_{\mathrm{global}}^*
=\underset{t\in\mathcal{T}}{\operatorname{argmax}}
\frac{1}{C}\sum_{c=1}^{C}F1_c(t).
$$

Thí nghiệm tìm được threshold chung là 0,85, nhưng threshold riêng theo lớp cho
macro-F1 cao hơn.

## 6. Threshold đã chọn cho 21 lớp

| Lớp | Threshold | Lớp | Threshold |
|---|---:|---|---:|
| Music | 0,66 | Vehicle | 0,78 |
| Speech | 0,67 | Animal | 0,86 |
| Musical instrument | 0,79 | Bird | 0,88 |
| Car | 0,90 | Domestic animals, pets | 0,89 |
| Water | 0,93 | Siren | 0,95 |
| Fowl | 0,93 | Wind instrument, woodwind instrument | 0,94 |
| Inside, small room | 0,83 | Train | 0,93 |
| Engine | 0,90 | Drum | 0,90 |
| Chicken, rooster | 0,91 | Rail transport | 0,95 |
| Bowed string instrument | 0,88 | Aircraft | 0,92 |
| Percussion | 0,86 |  |  |

Các ngưỡng được lưu tại `checkpoints/beats_21_thresholds.json` và được khóa sau
khi chọn trên validation. Khi chạy test hoặc inference:

```python
probabilities = torch.sigmoid(logits)
predictions = probabilities >= class_thresholds
```

Không được tối ưu lại threshold trên test. Threshold ảnh hưởng precision,
recall và F1 nhưng không làm thay đổi mAP, vì mAP được tính từ thứ hạng xác suất
trước khi nhị phân hóa.
