# [UET] BÁO CÁO MÔN HỌC NLP 2025 - INT3406 3

## Bài 1: Xây dựng mô hình dịch máy bằng Transformer (Code from scratch)
Bài toán: Xây dựng mô hình dịch máy Seq2Seq với kiến trúc Transformer

### A. Xử lý dữ liệu

Nhóm thực hiện quy trình thu thập, làm sạch và tiền xử lý dữ liệu theo yêu cầu chuẩn của bài toán dịch máy.

- **Bộ dữ liệu:** Nhóm sử dụng bộ dữ liệu IWSLT15 English-Vietnamese (IWSLT15 En-Vi). Đây là bộ dữ liệu chuẩn thường được sử dụng trong các nghiên cứu về dịch máy. [Link dataset](https://www.kaggle.com/datasets/tuannguyenvananh/iwslt15-englishvietnamese)

- **Thống kê dữ liệu:**
	- Train data: Gồm 133,166 cặp câu Tiếng Anh (train.en.txt) và Tiếng Việt (train.vi.txt)
	- Validation data: Trích xuất ngẫu nhiên 10% từ tập train gốc, tương đương 13,317 cặp câu để theo dõi quá trình huấn luyện và tránh hiện tượng overfitting. Tập train thực tế còn lại 119,849 cặp câu.
	- Test data: Gồm 2 bộ: **tst2012** (1,553 cặp câu) và **tst2013** (xxx cặp câu)

- **Tiền xử lý dữ liệu (Preprocessing Data):**
	- **Làm sạch (Cleaning):** Loại bỏ các ký tự nhiễu, chỉ giữ lại chữ cái và số, chuẩn hóa khoảng trắng.
	- **Chuẩn hóa văn bản:** Chuyển toàn bộ văn bản về dạng chữ thường (lowercase) để giảm kích thước từ điển mà vẫn giữ được ngữ nghĩa cơ bản.

- **Tokenization:** Sử dụng phương pháp tách từ dựa trên từ đơn (word-level tokenization) cho base line.

- **Xây dựng vocabulary:**
	- Thêm 4 token đặc biệt: `<pad>` (0), `<sos>` (1), `<eos>` (2), `<unk>` (3).
	- Ngưỡng tần suất (min_freq): 2 (loại bỏ các từ chỉ xuất hiện 1 lần để giảm nhiễu bằng cách chuyển thành `<unk>`).
	- **Kích thước từ điển:**
		- Tiếng Anh (Source): xxxx từ.
		- Tiếng Việt (Target): yyyy từ.
- **Padding & Truncation:**
	- Áp dụng kỹ thuật Dynamic Padding trong DataLoader: Thay vì đệm (pad) toàn bộ dữ liệu theo độ dài cố định, nhóm thực hiện đệm theo độ dài của câu dài nhất trong từng batch. Điều này giúp giảm đáng kể chi phí tính toán cho các token `<pad>`.
	- Giới hạn độ dài câu tối đa (MAX_LEN) là 100 token.

- **Code xây dựng vocabulary:**
```python
class Vocabulary:
	pass
```

### B. Xây dựng kiến trúc Transformer
![](https://media.geeksforgeeks.org/wp-content/uploads/20251004124012585570/transformers.webp)

- **Transformer Embedding (Input/Output Embedding & Positional Encoding):**
	- **Input Embedding:** Chuyển đổi các token (dạng one-hot index) thành các vector dày đặc (dense vectors) có kích thước $$d_{model}$$. Vector này được nhân với $$\sqrt{d_{model}}$$ để chuẩn hoá.
	- **Positional Encoding:** Do kiến trúc Transformer xử lý song song và không có tính tuần tự (recurrence) như RNN, mô hình không tự nhận biết được thứ tự từ. Nên cần có Positional Encoding, nhóm sử dụng PE dạng hình sin (Sinusoidal) để cộng thông tin vị trí vào vector embedding.
	- **Code:**
	```python
	class TransformerEmbedding(nn.Module):
		pass
	```

- **Multihead attention:**
Đây là thành phần cốt lõi của Transformer, cho phép mô hình tập trung vào các phần khác nhau của câu đầu vào.
	- **Scaled Dot-Product Attention:** Tính toán độ tương đồng giữa Query ($$Q$$) và Key ($$K$$), sau đó chuẩn hóa bằng Softmax và nhân với Value ($$V$$).

	- $$Attention(Q, K, V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V$$

  	- Sau đó chia cho $\sqrt{d_k}$ để tránh hiện tượng gradient vanishing.
	
	- **Multi-Head:** Chia vector đặc trưng thành $h$ đầu (heads) riêng biệt để mô hình có thể học được nhiều không gian biểu diễn ngữ nghĩa khác nhau song song.
	
	- **Code:**
	```python
	class MultiHeadAttention(nn.Module):
		pass
	```

- **Position-wise Feed-Forward Network (FFN):**
	- Mỗi vị trí trong câu đi qua một mạng nơ-ron truyền thẳng giống nhau bao gồm hai lớp biến đổi tuyến tính và một hàm kích hoạt ReLU ở giữa.

	- $$FFN(x) = \text{max}(0, xW_1 + b_1)W_2 + b_2$$

	- **Code:**
	```python
	class PositionwiseFeedForward(nn.Module):
		pass
	```

- **Encoder Layer:**
Encoder bao gồm $$N$$ lớp xếp chồng lên nhau. Mỗi lớp có hai khối con:
	- **Multi-Head Self-Attention:** là Multi-Head attention nhưng Q,K,V đến từ cùng 1 nguồn.
	- **Position-wise Feed-Forward:** Xung quanh mỗi khối con, có kết nối tắt (Residual Connection) và chuẩn hóa lớp (Layer Normalization): $LayerNorm(x + Sublayer(x))$.
	- **Code:**
	```python
	class EncoderLayer(nn.Module):
		pass
	```

- **Decoder Layer:**
Decoder cũng gồm $N$ lớp, nhưng mỗi lớp có thêm một khối cross-attention để nhìn vào Encoder:
	- **Masked Multi-Head Self-Attention:** Đảm bảo vị trí $i$ chỉ có thể chú ý đến các vị trí trước nó ($<i$). Sử dụng mask tam giác dưới để thực hiện điều này (Look-ahead mask).
	- **Multi-Head Cross-Attention:** Query ($Q$) lấy từ lớp Decoder trước đó, trong khi Key ($K$) và Value ($V$) lấy từ đầu ra của Encoder. Khối này giúp Decoder "nhìn" vào câu nguồn để dịch
	- **Feed-Forward Network**.
	- **Code:**
	```python
	class DecoderLayer(nn.Module):
		pass
	```

- **Transformer:**
Ghép nối Encoder và Decoder, thêm lớp Linear cuối cùng và hàm Softmax để dự đoán xác suất của từ tiếp theo trong từ điển đích.
	- Code:
	```python
	class Transformer(nn.Module):
		pass
	```

### C. Training
Nhóm thiết lập quy trình huấn luyện với các hyperparameters và hàm loss tiêu chuẩn.
1. **Hyper-parameters:** Do tập dữ liệu IWSLT khá nhỏ (~130k câu) so với các tập dữ liệu lớn như WMT, nhóm chọn kích thước mô hình (Small Transformer) để đảm bảo mô hình hội tụ nhanh và tránh overfitting, đồng thời tiết kiệm tài nguyên tính toán.
```python
BATCH_SIZE = 16        # Kích thước batch phù hợp với GPU T4 (Colab)
D_MODEL = 256          # Kích thước vector ẩn (Giảm từ 512 để phù hợp dataset nhỏ)
N_HEAD = 4             # Số lượng attention heads
N_LAYER = 4            # Số lớp Encoder/Decoder
D_FF = 1024            # Kích thước ẩn của lớp FFN (4 * d_model)
DROPOUT = 0.1          # Hệ số dropout để tránh overfitting
EPOCHS = 10            # Số vòng lặp huấn luyện
LEARNING_RATE = 0.0005 # Tốc độ học khởi tạo
MAX_LEN = 100          # Độ dài câu tối đa
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

2. **Khởi tạo và Loss Function:**
	- Khởi tạo trọng số: Sử dụng phân phối Xavier Uniform (nn.init.xavier_uniform_) giúp quá trình huấn luyện ổn định hơn.
	- Optimizer: Sử dụng Adam Optimizer.
	- Loss Function: Sử dụng CrossEntropyLoss với ignore_index=PAD_IDX để mô hình không tính lỗi cho các padding token.
   	- Code:
	```python
	model = Transformer(...).to(DEVICE)
	```

3. **Training Loop:**
Nhóm thực hiện vòng lặp huấn luyện, theo dõi Loss trên tập Train và Validation sau mỗi epoch.
```python
# Train model
```
- Đồ thị Loss: ...
- Nhận xét: Loss giảm đều từ 6.x xuống 2.x sau 10 epochs, chứng tỏ mô hình đang học tốt.

### D. Testing - Evaluation
Nhóm sử dụng hai phương pháp đánh giá trên tập Test:
- BLEU score: BLEU Score (Bilingual Evaluation Understudy): Độ đo tiêu chuẩn đánh giá sự trùng khớp n-grams giữa câu máy dịch và câu tham chiếu.
- Gemini Score: Sử dụng LLM (gemini-2.5-flash) làm giám khảo để chấm điểm chất lượng bản dịch dựa trên thang điểm 1-10 về độ trôi chảy và chính xác ngữ nghĩa.

### E. Tối ưu
- Sử dụng Beam Search: Thay vì Greedy tại mỗi bước chọn từ có xác suất cao nhất, Beam Search duy trì $k$ (beam width) ứng viên tốt nhất tại mỗi bước để tìm ra chuỗi từ tối ưu toàn cục.
- Tối ưu tokenizer: Thử nghiệm thay thế tách từ đơn giản bằng BPE (Byte Pair Encoding) để xử lý tốt hơn các từ hiếm (OOV) và giảm kích thước từ điển.
- Tối ưu hyperparameters:
	- Tăng Dropout lên 0.2 khi train trên tập dữ liệu nhỏ để tăng khả năng tổng quát hóa.
	- Sử dụng Learning Rate Scheduler (Warmup + Decay) thay vì LR cố định.

### D. Kết quả cuối cùng
Bảng BLEU score, Gemini Score cho các phương pháp: RNN (base-line), Transformer cơ bản, Transformer v1, v2... (cải tiến tối ưu)

## Bài 2: Dịch máy y tế VLSP 2025
