# Tối ưu hóa MAStAC Training cho Mean Coverage

## Các vấn đề đã phát hiện và sửa:

### 1. **Vấn đề nghiêm trọng: Print statement trong training loop**
- **Vị trí**: `bayesian_model/networks/learner.py` line 255
- **Vấn đề**: `print("cam_rewards =", cam_rewards)` chạy mỗi step → làm chậm đáng kể
- **Đã sửa**: ✅ Xóa print statement

### 2. **Rendering được bật trong training**
- **Vị trí**: `configs/mastac_mate.yaml` - `render_mode: "human"`
- **Vấn đề**: Rendering mỗi step cực kỳ chậm (có thể làm chậm 100-1000x)
- **Đã sửa**: ✅ Đổi thành `render_mode: None`

### 3. **Log interval quá cao**
- **Vị trí**: `configs/mastac_mate.yaml` - `log_interval: 50000`
- **Vấn đề**: Không có feedback trong 50k steps đầu → không biết training có chạy không
- **Đã sửa**: ✅ Giảm xuống `log_interval: 1000`

### 4. **Learning starts quá cao**
- **Vị trí**: `configs/mastac_mate.yaml` - `learning_starts: 5000`
- **Vấn đề**: Phải đợi 5k steps mới bắt đầu training
- **Đã sửa**: ✅ Giảm xuống `learning_starts: 1000`

### 5. **Episode length không được track**
- **Vị trí**: `train_bayesian.py` - `evaluate_agent()` function
- **Vấn đề**: `lengths.append(0)` → không track episode length thực tế
- **Đã sửa**: ✅ Track episode_length đúng cách và thêm coverage_rate tracking

### 6. **Gradient steps không được sử dụng**
- **Vị trí**: `bayesian_model/networks/learner.py`
- **Vấn đề**: Config có `gradient_steps: 2` nhưng không được dùng
- **Đã sửa**: ✅ Implement gradient_steps trong learn() method

### 7. **Logging không đủ thông tin**
- **Vị trí**: `train_bayesian.py` - training loop
- **Vấn đề**: Logging thiếu thông tin về speed, coverage rate
- **Đã sửa**: ✅ Thêm steps/sec, coverage rate, và save eval results

## Các tối ưu hóa đã áp dụng:

1. **Tắt rendering hoàn toàn** → Tăng tốc 100-1000x
2. **Giảm log_interval** → Có feedback sớm hơn
3. **Giảm learning_starts** → Training bắt đầu sớm hơn
4. **Track metrics đúng cách** → Theo dõi coverage rate
5. **Logging chi tiết hơn** → Biết được training có chạy không

## Các vấn đề tiềm ẩn khác cần lưu ý:

### 1. **Update method có thể chậm**
- Mỗi update loop qua tất cả agents (8 agents: 4 cameras + 4 targets)
- Mỗi agent loop qua batch (128 samples)
- Mỗi sample phải compute next actions từ target actors
- **Gợi ý**: Nếu vẫn chậm, có thể giảm batch_size hoặc train_freq

### 2. **Buffer size**
- Buffer size = 300k có thể quá lớn cho 8 agents
- **Gợi ý**: Có thể giảm xuống 100k-200k nếu memory là vấn đề

### 3. **Network size**
- Actor: [256, 256] - có thể lớn
- Critic GNN: [256, 256] - có thể lớn
- **Gợi ý**: Nếu vẫn chậm, thử giảm xuống [128, 128]

### 4. **Mean Coverage Optimization**
- Hiện tại reward là `cam_rewards` và `tar_rewards` từ environment
- Cần đảm bảo environment reward đã optimize cho mean_coverage
- **Gợi ý**: Kiểm tra reward shaping trong MATE environment config

## Cách kiểm tra training có chạy:

1. **Sau 1000 steps**: Sẽ có log đầu tiên với buffer size và loss
2. **Sau 10000 steps**: Sẽ có evaluation đầu tiên với coverage rate
3. **Check log file**: `logs/mastac/training.log` sẽ có thông tin mỗi 1000 steps
4. **Check stats file**: `logs/mastac/train_stats.json` được update mỗi 1000 steps

## Expected Performance:

- **Trước**: 7 giờ không có kết quả (có thể do print + rendering)
- **Sau**: Nên thấy log đầu tiên sau ~1-2 phút (tùy vào hardware)
- **Training speed**: Nên đạt ít nhất 10-50 steps/second (tùy vào GPU)

## Next Steps:

1. Chạy lại training với config mới
2. Kiểm tra log sau 1000 steps đầu tiên
3. Nếu vẫn chậm, giảm batch_size hoặc network size
4. Monitor coverage rate trong evaluation để đảm bảo đang optimize đúng metric


