# Interpretability (Grad-CAM++)

`gradcam_students_vs_teacher.ipynb` applies Grad-CAM++ to the two strongest
students (MobileViT + DistilBioBERT, RepViT + DistilBioBERT) and the frozen
BioViL-T teacher, using the paired report embedding as the query direction.
Produces the per-study attention-concentration scores and the attention-map
figures reported in the paper (Appendix A).
