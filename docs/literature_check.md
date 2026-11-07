# Literature check (2026-11-08, web search by a sub-agent; only listed sources were seen)

Key facts
- CARE-PD UPDRS-gait labels are 0-3 (score 3 rare). CARE-PD's own interpretable baseline = Random Forest on gait
  features (cadence, step length/width/time, walking speed, margin of stability, foot lifting, arm swing, stoop
  posture); no explanation analysis reported.
- No published model found that combines differentiable clinical symptom operators + monotone additive heads +
  penalised residual + injection-based faithfulness test for PD gait (searches not exhaustive; a 2026 preprint may exist).

Closest prior work and our difference
- CARE-PD RF baseline: fixed features, no explanations -> ours learns operator parameters end-to-end, explanation is the model.
- Concept bottleneck models (Koh 2020), post-hoc CBM with residual (Yuksekgonul 2023), side channel (Havasi 2022),
  concept-residual leakage (Zabounidis 2023), CEM (Espinosa Zarlenga 2022): ours needs no concept labels; residual is
  penalised and its share is reported (the leakage question).
- Neural additive models (Agarwal 2021), NODE-GAM (Chang 2022), monotonic NAM (2022): fixed inputs; ours adds learnable
  symptom operators in front of monotone heads.
- SincNet (2018), LEAF (2021), DDSP (2020), SincPD (2025, ground-reaction force PD): learnable signal operators feeding a
  black box; ours feeds clinically ordered additive terms on pose data.
- Sabo 2020 JNER / 2022 JBHI, Lu 2020 MICCAI, Rupprechter 2021 Sensors, Deng 2024 npj PD: gait features or deep models
  with post-hoc importance/SHAP; ours is interpretable by design and tested with ground-truth injection.
- Faithfulness with ground truth: Zhou 2022 AAAI, Shah 2021 NeurIPS (BlockMNIST), Ismail 2020 NeurIPS, Turbe 2023 NMI;
  skeleton XAI evaluation: Pellano 2024 Sensors; sanity checks: Adebayo 2018; scene bias: Choi 2019.

References (verified by search result)
- Adeli et al., Care-PD, NeurIPS D&B 2025. https://arxiv.org/abs/2510.04312
- Koh et al., Concept Bottleneck Models, ICML 2020. https://proceedings.mlr.press/v119/koh20a.html
- Yuksekgonul, Wang, Zou, Post-hoc Concept Bottleneck Models, ICLR 2023. https://arxiv.org/abs/2205.15480
- Havasi, Parbhoo, Doshi-Velez, Addressing Leakage in Concept Bottleneck Models, NeurIPS 2022.
- Espinosa Zarlenga et al., Concept Embedding Models, NeurIPS 2022. https://arxiv.org/abs/2209.09056
- Zabounidis et al., Benchmarking and Enhancing Disentanglement in Concept-Residual Models, arXiv 2023. https://arxiv.org/abs/2312.00192
- Agarwal et al., Neural Additive Models, NeurIPS 2021. https://arxiv.org/abs/2004.13912
- Chang, Caruana, Goldenberg, NODE-GAM, ICLR 2022. https://arxiv.org/abs/2106.01613
- Monotonic Neural Additive Models (credit scoring), arXiv 2022. https://arxiv.org/abs/2209.10070
- Ravanelli & Bengio, SincNet, SLT 2018. https://arxiv.org/abs/1808.00158
- Zeghidour et al., LEAF, ICLR 2021. https://arxiv.org/abs/2101.08596
- Engel et al., DDSP, ICLR 2020. https://arxiv.org/abs/2001.04643
- SincPD, arXiv 2025. https://arxiv.org/abs/2502.17463
- Sabo et al., JNER 2020. https://doi.org/10.1186/s12984-020-00728-9
- Sabo et al., IEEE JBHI 2022. https://arxiv.org/abs/2105.03464
- Lu et al., MICCAI 2020. https://arxiv.org/abs/2007.08920
- Rupprechter et al., Sensors 2021 (title not seen exactly). https://openaccess.sgul.ac.uk/id/eprint/113607
- Deng et al., npj Parkinson's Disease 2024. https://doi.org/10.1038/s41531-024-00742-x
- Adebayo et al., Sanity Checks for Saliency Maps, NeurIPS 2018. https://arxiv.org/abs/1810.03292
- Ismail et al., Benchmarking DL Interpretability in Time Series, NeurIPS 2020. https://arxiv.org/abs/2010.13924
- Turbe et al., Nature Machine Intelligence 2023. https://arxiv.org/abs/2202.05656
- Zhou, Booth, Ribeiro, Shah, Do Feature Attribution Methods Correctly Attribute Features?, AAAI 2022. https://arxiv.org/abs/2104.14403
- Shah, Jain, Netrapalli, Do Input Gradients Highlight Discriminative Features?, NeurIPS 2021. https://arxiv.org/abs/2102.12781
- Choi et al., Why Can't I Dance in the Mall?, NeurIPS 2019. https://arxiv.org/abs/1912.05534
- Pellano, Struemke, Ihlen, From Movements to Metrics, Sensors 2024. https://arxiv.org/abs/2402.12790
Unverified (do not cite without checking): Mahinpei 2021; Li 2018 RESOUND; Liu 2020 certified monotonic; You 2017 deep lattice; GAMI-Net primary.
