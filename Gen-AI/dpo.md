# Direct Preference Optimization (DPO) — Dạy mô hình nhận biết câu trả lời nào tốt hơn

## Mục lục

1. [DPO đang giải quyết vấn đề gì?](#1-dpo-đang-giải-quyết-vấn-đề-gì)
2. [Ba kiểu huấn luyện: Pre-training, SFT và DPO](#2-ba-kiểu-huấn-luyện-pre-training-sft-và-dpo)
3. [Dữ liệu preference trông như thế nào?](#3-dữ-liệu-preference-trông-như-thế-nào)
4. [Từ “A tốt hơn B” đến xác suất](#4-từ-a-tốt-hơn-b-đến-xác-suất)
5. [Từ xác suất đến loss](#5-từ-xác-suất-đến-loss)
6. [LLM tính điểm cho một câu như thế nào?](#6-llm-tính-điểm-cho-một-câu-như-thế-nào)
7. [Vì sao không dùng log-probability thô?](#7-vì-sao-không-dùng-log-probability-thô)
8. [Reference model — chiếc thước đo cố định](#8-reference-model--chiếc-thước-đo-cố-định)
9. [Lắp ráp công thức DPO](#9-lắp-ráp-công-thức-dpo)
10. [Ví dụ tính tay từ đầu đến cuối](#10-ví-dụ-tính-tay-từ-đầu-đến-cuối)
11. [Tensor thay đổi kích thước ra sao?](#11-tensor-thay-đổi-kích-thước-ra-sao)
12. [Cài đặt DPO bằng PyTorch](#12-cài-đặt-dpo-bằng-pytorch)
13. [DPO khác SFT và RLHF/PPO thế nào?](#13-dpo-khác-sft-và-rlhfppo-thế-nào)
14. [Những điều DPO không tự động giải quyết](#14-những-điều-dpo-không-tự-động-giải-quyết)
15. [Bài tập tự luyện](#15-bài-tập-tự-luyện)
16. [Tóm tắt một trang](#16-tóm-tắt-một-trang)

---

## 1. DPO đang giải quyết vấn đề gì?

