# Dữ liệu theo phiên bản

## v003 (hiện hành)

- `data/v003/`: workspace của `src/data/` (inputs, ledger, raw, pool, registry, archive). Đổi bằng biến môi trường `VNAV_DATA`.
- `data/exports/vn-av-df-data/vn-av-df-data-partN/`: part real đã chia split, đầu vào của `src/generators/`.
- `data/vn-av-df-data-partN/`: part đã chốt (real + fake, `manifest.jsonl`), đầu vào của train. Chi tiết: [src/data/README.md](../src/data/README.md), [PIPELINE.md](../PIPELINE.md).

## Trước v003: dataset_v1 và dataset_v2

Đã xoá khỏi máy ngày 27/10/2026. Cấu trúc thư mục, số clip và run Wav2Lip cũ được mô tả trong `data/README.md` ở tag `pre-v003`; kết quả ở [báo cáo 11/10](../docs/reports/2026-10-11_WAV2LIP_SYNC_RESULTS.md).
