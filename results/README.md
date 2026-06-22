# Results

Final metric files backing the paper's tables. All retrieval numbers are on the
identical full 14,018-study MIMIC-CXR test gallery.

| File | Paper table | Contents |
|------|-------------|----------|
| `retrieval_test_metrics.json` / `.csv` | Table I | Per-model bidirectional retrieval (R@1/5/10, median/mean rank) for the teacher and all four student pairs, both stages |
| `distillation_recovery.csv` | Table II | Student R@1 as a percentage of teacher R@1 |
| `efficiency.csv` | Table III | FLOPs, parameters, latency |
| `baselines.csv` | Table IV | Baseline retrieval (CLIP, MobileCLIP, TinyCLIP, ConVIRT, CheXzero, CXR-CLIP, MedCLIP, MGCA) |
| `gradcam_concentration.csv` | Appendix A | Per-study Grad-CAM++ attention concentration scores |

Numbers are transcribed from the paper tables; regenerate them from the
notebooks in `src/students/`, `src/baselines/`, and the Grad-CAM++ notebook to
reproduce.
