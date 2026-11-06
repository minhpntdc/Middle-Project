# Scientific Evaluation Report: Video temporal dynamics Experiment

## 1. Quantitative Benchmark Summary

| Model | Grid | Dim (D) | PCA 3PC Var (%) | Mean Cosine (Subject A) | Mean Cosine (Background B) | Effective Rank (RankMe) |
|---|---|---:|---:|---:|---:|---:|
| **VAE** | 32x32 | 4 | 89.43% | 0.1015 | -0.0336 | 3.75 |
| **ViT** | 14x14 | 768 | 31.0% | 0.6171 | 0.5091 | 132.01 |
| **V-JEPA 2** | 16x16 | 1024 | 31.94% | 0.4115 | 0.3398 | 178.44 |

## 2. Theoretical Grounding & Interpretation

1. **Pixel Reconstruction vs. Latent Prediction:**
   - **VAE** (*Kingma & Welling, 2013*): Nén vào $D=4$ kênh tiềm ẩn. Tỷ lệ phương sai PCA cao do số chiều thấp, nhưng các đặc trưng bị chi phối bởi cường độ sáng và màu sắc pixel cục bộ.
   - **ViT** (*Dosovitskiy et al., 2021*): Được huấn luyện có giám sát trên ImageNet, các patch thể hiện độ tương đồng cao trong cùng phân vùng ngữ nghĩa.
   - **V-JEPA 2** (*Bardes et al., 2024*): Tự giám sát trong không gian tiềm ẩn video. Nắm bắt cấu trúc hình học và động lực học thời gian mà không bị phân tâm bởi nhiễu tần số cao của pixel.

2. **Dimensional Collapse Assessment (*Garrido et al., ICLR 2023*):**
   - Điểm số **RankMe** đo lường entropy phổ giá trị riêng. Giá trị RankMe cao hơn thể hiện không gian biểu diễn đa dạng, không bị sụp đổ chiều.

3. **Scientific Caution:**
   - Độ tương đồng Cosine cao giữa các patch không đồng nghĩa với độ chính xác phân loại cao hơn; nó phản ánh mức độ tập trung hướng của các vector đặc trưng.
