# CS406 GeoGuessr — master experiment specification

Trạng thái: hướng nghiên cứu và preprocessing v1 đã chốt; các giá trị phụ thuộc UIT pilot được đánh dấu pending. Tài liệu này không cấp phép mở thêm experiment branches hoặc bắt đầu implementation training trước chỉ thị của người dùng.

## Mục tiêu và scope

Country/territory classification: một screenshot → một trong 124 labels, kèm Top-3/Top-5. Primary objective là benchmark score trên fixed split hiện tại. Báo cáo trade-off accuracy/long-tail, không bắt buộc một single best model giữa các phương pháp.

| Main experiment | Input/representation | Loss và sampling |
|---|---|---|
| E1 | ConvNeXt-Tiny, full image 896×384 | Standard CE, natural distribution |
| E1-HR | Cùng backbone, full image resolution cao hơn, pending pilot | Standard CE, natural distribution |
| E2 | Shared encoder: global 896×384 + 4 local 320×320 | Một standard CE sau fusion, natural distribution |
| E3/E1-natural | Frozen E1 feature → reset/retrain linear head | CE, natural sampling |
| E3/E1-cRT | Cùng frozen E1 và head protocol | CE, class-uniform sampling |
| E3/E2-natural | Frozen E2 fused feature → reset/retrain linear head | CE, natural sampling |
| E3/E2-cRT | Cùng frozen E2 và head protocol | CE, class-uniform sampling |

Hai natural-retraining branches là controls để giải thích nguồn gain, không phải hướng nghiên cứu riêng. Chưa có E1-HR+cRT. E1-HR không có compute-matching constraint; efficiency là secondary information. Không dùng chênh lệch score này để tuyên bố đã cô lập compute.

LLRD, head LR multiplier >1, alternative fusion, random crops, separate encoders, auxiliary losses, strong augmentation, alternative balancing và robustness evaluation chỉ là optional/future work. Fixed recipe choices không tự động tạo ablations.

## Dữ liệu và preprocessing

Nguồn chuẩn: [fixed artifacts](../artifacts/baseline124/README.md) và [preprocessing v1](preprocessing_v1.md). Giữ đúng membership: 49.996 unique images, train 34.997, val 7.497, test 7.502; split seed 406 không phải công thức để regenerate split. Label mapping có 124 outputs; support validation 106 labels, test 111.

Chỉ train images được dùng chọn UI/crop geometry. V1 đã freeze 384/402 source crops, footer 14px, static masks và navigation-arrow exception. E1/E1-HR/E2 dùng cùng UI policy. Không geometry search sau freeze.

Test history được mô tả bảo thủ là B: từng evaluate/inspect trong pipeline cũ, chưa có bằng chứng dùng để tune. Mọi quyết định mới chỉ dùng train + validation. Test khóa đến final evaluation. Geographic independence chưa được xác minh; secondary robustness được defer sang evaluation-spec, không nằm trong main scope.

## Architecture và pretrained starting point

- `torchvision.models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1`, file `convnext_tiny-983f1562.pth`.
- E1/E1-HR/E2 khởi tạo độc lập từ cùng file; không kế thừa E1 checkpoint/optimizer/scheduler. Lưu full weight-file SHA-256 và versions PyTorch/TorchVision.
- Giữ pretrained representation gồm feature extractor, global average pooling và final LayerNorm; chỉ thay linear ImageNet classifier bằng linear head cho 124 labels. Không vô tình xóa final LayerNorm khi thay classifier.
- E1/E1-HR dùng feature 768 chiều. E2 dùng cùng representation cho mọi view, average bốn local vectors, concatenate với global vector thành 1536 chiều → linear classifier 124 outputs.
- Không positional identity, extra attention/MLP, extra branch L2 normalization hoặc per-local auxiliary CE trong E2 v1. Chỉ một fused CE trên mỗi ảnh gốc; gradient truyền qua tất cả view.

## Main fine-tuning recipe

- Full fine-tuning từ epoch 1, có LR warmup, không classifier-only phase.
- AdamW; `LR_head = LR_backbone`; không LLRD. Chỉ tune base LR riêng từng architecture.
- `weight_decay=0.05` cho trainable Conv/Linear weights; bias, LayerNorm và layer-scale coefficients không decay. Mọi trainable parameter phải thuộc đúng một group; không bỏ sót parameter.
- `stochastic_depth_prob=0.1`, phân bố tuyến tính theo block depth từ 0 đến 0.1; không dựa vào library default. Không thêm head dropout trong v1.
- Brightness/contrast jitter nhẹ, một bộ params mỗi ảnh gốc dùng chung mọi view; không làm đổi mean-filled UI/padding. Amplitude số cụ thể pending training pilot.
- Không horizontal flip, random crop, RandAugment, Random Erasing, Mixup, CutMix hoặc label smoothing.
- AMP cho model training; deterministic preprocessing vẫn CPU float32. Numeric precision, clipping policy và các optimizer/scheduler numbers cần freeze trong pilot recipe.
- Cosine scheduler sau warmup; cùng max epochs, warmup policy, patience, effective batch size theo số ảnh gốc và checkpoint rule cho cả ba architecture. Các numeric values pending UIT pilot.
- Dùng gradient accumulation nếu cần để giữ effective batch size; năm view của một ảnh E2 vẫn tính là một mẫu. Scheduler theo optimizer updates, không theo từng view/microbatch. Chi tiết accumulation cuối epoch cần khai báo explicit ở implementation.

## Tuning và final seeds

Sau UIT pilot, freeze common training budget trước LR search chính thức. Mỗi architecture dùng cùng **4 LR candidates × 1 training seed**, cùng số trial và protocol. Chọn LR bằng validation Macro-F1 supported, công bố mọi trial.

Sau đó freeze selected configuration và chạy **3 training seeds** cho E1, E1-HR, E2. Không chạy 4 LR × 3 seeds. Split giữ nguyên. Báo cáo mean ± sample standard deviation từ từng seed, không chọn seed có test score đẹp nhất và không tự động ensemble các seeds.

Không giảm epoch/trials riêng cho architecture chậm hơn. Nếu cần giảm budget, thay đổi protocol chung trước official search và ghi nhận rõ. Chưa có hard total GPU-hours/deadline; Slurm job limit được xử lý bằng epoch-boundary resume.

## E3 protocol

Mỗi training seed kế thừa representation checkpoint nguồn được chọn bằng validation Macro-F1. Freeze toàn bộ representation, evaluation mode, tắt stochastic depth. E1 cache feature trước head; E2 cache fused feature trước head. Mỗi train/val image đúng một deterministic feature, không jitter hoặc nhiều augmented feature versions.

LR/optimizer/budget/initialization cho head được chốt bằng pilot nhỏ để bảo đảm hội tụ, không mở full LR-search branch. Trong mỗi cặp natural/cRT cùng representation: dùng cùng feature cache, cùng initial head state theo seed, optimizer, LR, number of draws/steps, scheduler và checkpoint rule. Chỉ sampling khác nhau.

cRT chọn đều một trong 124 lớp, rồi chọn đều một ảnh train thuộc lớp đó; xác suất mỗi ảnh là `1/(124*n_train[class])`. Không dùng weighted CE, logit adjustment hoặc balanced softmax đi kèm. Natural sampling giữ phân bố ảnh train tự nhiên. Độ dài virtual epoch và replacement mechanics phải explicit và freeze sau head pilot; không tự tăng số steps của cRT.

Extreme-tail 1–2 unique train images có thể bị lặp nhiều; báo cáo thận trọng về memorization, không suy diễn vài prediction đúng thành bằng chứng mạnh về generalization.

## Checkpoint selection và evaluation

`best.pt` chọn bằng validation Macro-F1 trên labels có ground-truth validation support; mọi metric của run lấy từ chính checkpoint đó. Không lấy best Top-1 ở epoch khác. Cùng selection rule cho E1/E1-HR/E2 và head retraining.

Lưu full validation history, learning curves, best/last checkpoints. Early stopping dùng patience. Tolerance và tie-break cụ thể pending pilot spec, phải freeze trước LR search; không chọn tùy ý sau khi xem curve. Giữa các phương pháp, phân tích nhiều metrics và trade-off, không ép winner theo scalar duy nhất.

Metrics chính: Top-1/3/5, Macro-F1 supported, Weighted-F1; per-class precision/recall/F1/support/predicted count; confusion matrix, confused pairs, predictions CSV. Ranking luôn trên 124 outputs; không loại logits của unsupported labels trước Top-k. Unsupported evaluation classes được liệt kê, không gán F1=0 để đưa vào supported macro average.

Frequency buckets dựa trên unique `n_train` của fixed manifest: 1–19, 20–99, 100–499, ≥500; bổ sung nhóm 1–2 bên trong 1–19. Membership cố định qua mọi model và resampling. Mỗi bucket ghi số class, số evaluated class, số sample, Macro-F1 và Top-1/3/5. Bucket Macro-F1 lấy từ per-class F1 tính trên toàn tập evaluation, không bỏ false positives từ bucket khác bằng cách lọc ground-truth rows trước.

## Reproducibility và UIT execution

Single GPU: Slurm partition `normal`, `gpu:2g.35gb:1`; portable code, UIT-specific paths đặt trong launch config/script. Data/models/logs ở `/datastore/binhvp/<my-folder>/`. Không Kaggle-specific paths. Folder cụ thể và job walltime là deployment parameters cần lấy từ môi trường UIT.

Mỗi run lưu complete config, seeds, source commit, architecture, pretrained ID/hash, label mapping, dataset and preprocessing hashes, environment/GPU/PyTorch/TorchVision/Pillow/CUDA/cuDNN versions, optimizer/scheduler/GradScaler state, epoch, best metric, history và RNG states CPU/CUDA/Python/NumPy cùng sampler/augmentation/DataLoader RNG policy. Resume tại epoch boundary phải giữ update counters và scheduler position; Slurm resume không tạo budget mới.

Không overwrite runs. Run identity nên gồm experiment, frozen-config hash, training seed và unique run ID. Cache identity gồm source checkpoint SHA-256, split/label/preprocessing hashes, view profile, feature definition/dtype và environment version. Cache features không được trộn giữa seeds hoặc source checkpoints.

## Các phần còn pending

Xem [pilot plan](pilot_plan.md): E1-HR resolution, microbatch/accumulation và common effective batch, bốn LR values, epochs/warmup/patience/tolerance, AMP/clipping, augmentation amplitude, và small head-retraining recipe. Các giá trị này cần dữ liệu pilot trên UIT; không tự coi là đã benchmark tại local.
