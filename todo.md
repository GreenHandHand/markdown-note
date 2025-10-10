# Revision

## Q1: Are the objectives and the rationale of the study clearly stated?

Please provide suggestions to the author(s) on how to improve the clarity of the objectives and rationale of the study. Please number each suggestion so that author(s) can more easily respond.

### Reviewer \#2

1. Clarify the Core Research Problem Early in the Introduction:  
    The introduction outlines background well, but the specific research gap (e.g., "existing methods are accurate but computationally expensive due to reliance on optical flow and Transformer architectures") should be made more explicit in the first 2–3 paragraphs.
2. Explicitly Define the Objective in a Standalone Sentence:  
    While the goals are implied, a clear objective sentence would strengthen the flow. For example:
"The objective of this study is to develop an efficient and accurate temporal action localization model that eliminates the need for optical flow and reduces model complexity through knowledge distillation and convolutional simplification."
3. Provide a Brief Intuitive Justification for Each Component (AA, CS, CM, SC):  
  In the introduction or methodology overview, give a 1-line motivation for each module:
    - Why is cross-modal distillation important?
    - Why use CNNs instead of Transformers?
    - How does action attention address temporal sparsity?

4. Clearly Distinguish this Work from Prior Similar Efforts (e.g., Lee et al. [2]):
    - The manuscript builds on and improves the cross-modal distillation idea proposed by Lee et al., but the distinction would be clearer if a table or paragraph explicitly contrasted:
        - Feature reuse
        - Inference architecture
        - Efficiency gains

5. Introduce the Term “C2MS-Net” Earlier:  
    The acronym appears mid-way through the methodology. It should be introduced upfront in the abstract or early in the introduction for consistency.

6. Justify the Use of Fully Supervised Setup:  
    Since semi-supervised methods are briefly mentioned, a sentence explaining why the fully supervised approach is chosen (despite annotation costs) would help contextualize the rationale.

### Reviewer \#3

1. The objectives of the study are stated toward the end of the introduction, focusing on developing C2MS-Net with cross-modal and cross-structure distillation, action attention, and simplified convolution to achieve high efficiency without compromising accuracy in temporal action localization (TAL).
2. To further improve clarity, explicitly separate the primary objective (achieving an optimal speed–accuracy trade-off without reliance on optical flow) from the secondary objectives (introducing each module-level innovation). This will help the reader quickly understand the hierarchy of contributions.
3. Expand the rationale section by succinctly highlighting the current bottlenecks in existing TAL approaches — particularly computational overhead and reliance on optical flow — and linking these directly to how each proposed module addresses them.
4. Include a brief comparison with the closest related works at the end of the introduction to better emphasize the novelty of integrating all proposed modules into one unified framework.
5. Strengthen the practical motivation by mentioning application scenarios (e.g., real-time surveillance, sports analysis, embedded AI devices) where the proposed efficiency gains are critical.

.Ensure that the concluding paragraph of the introduction summarizes both the problem context and proposed solution in a single, concise statement to reinforce the study’s rationale before moving into the methods section.

### Reviewer \#5

yes


## Q2: If applicable, is the application/theory/method/study reported in sufficient detail to allow for its replicability and/or reproducibility?

Please provide suggestions to the author(s) on how to improve the replicability/reproducibility of their study. Please number each suggestion so that the author(s) can more easily respond.

### Reviewer \#2

Yes.

The study is reported in sufficient detail to allow for general replicability, especially with the inclusion of architecture diagrams, loss formulations, evaluation metrics, and dataset usage. However, to ensure full reproducibility, the authors should include key implementation details such as exact hyperparameter settings, training schedules for distillation phases, versioning of software libraries, and weight values used in loss functions. Clarifying these aspects would strengthen reproducibility further.
1. Provide Specific Hyperparameters Used for Training:  
    Clearly mention learning rate, batch size, number of epochs, optimizer used (e.g., Adam, SGD), weight decay, dropout, etc.  
    These details are critical for replicating the model behavior.
2. Clarify the Distillation Weights (a, b, c, λ):  
    In the loss functions (Equations 3, 8, and 10), weights a, b, c, and λ are introduced but their actual values are not disclosed.  
    Please specify the exact numerical values used during training.
3. Share Training Duration and Convergence Criteria:  
    Indicate how long the model takes to train (wall-clock time or number of iterations) and how convergence was assessed (e.g., early stopping, validation loss plateau).
4. List Software Library Versions:  
    Mention the versions of PyTorch, CUDA, Python, and other major libraries/tools (e.g., THOP) to ensure compatibility.
5. Clarify Preprocessing Steps for I3D Features:  
    The paper uses pre-extracted I3D features with a sliding window, but exact frame extraction steps, window overlap handling, and any normalization should be described.  
    State whether the feature extractor weights were fine-tuned or kept fixed.
6. Include Details of the Cross-Modal Distillation Training Pipeline:  
    Since three rounds of distillation are involved, briefly outline the training pipeline sequence:  
    - Which models are trained first?
    - Are the teacher models frozen or updated during student training?
7. Ensure that the GitHub Code is Sufficiently Documented:  
    The paper references an existing GitHub repo for ActionFormer.  
    Clarify whether all modifications (e.g., MFTM, CNN backbone, AA module) are available in the public repo or will be provided as part of this work.
8. Mention the Evaluation Thresholds and Post-Processing Settings:  
    Clearly state values of α, β, and top-k filtering used in post-processing.  
    These values affect final mAP scores and should be replicable.
9. Add a Diagram or Table Showing the Distillation Flow:  
    To support reproducibility of the three-stage distillation, a visual depiction or training schedule table would help readers implement the workflow more easily.
10. Clarify Dataset Split Strategy:  
    It is stated that the THUMOS14 validation set is used for training and the test set for evaluation.  
    Confirm whether any frame subsampling or data augmentation (e.g., random crop, scale, flip) was applied.

### Reviewer \#3

Yes.

1. The methodology is described in sufficient detail, including model architecture, module descriptions, and evaluation datasets, enabling replication by experienced researchers.
2. To further enhance reproducibility, consider including hyperparameter values (learning rate schedule, optimizer type, weight decay, batch size, number of training epochs) in a consolidated table for quick reference.
3. Provide implementation details for the knowledge distillation setup (e.g., loss weighting, teacher–student training schedule, and initialization).
4. Share code and pre-trained models in a public repository to facilitate direct validation and benchmarking.

### Reviewer \#5

Yes.

## Q3: If applicable, are statistical analyses, controls, sampling mechanism, and statistical reporting (e.g., P-values, CIs, effect sizes) appropriate and well described?

Please clearly indicate if the manuscript requires additional peer review by a statistician. Kindly provide suggestions to the author(s) on how to improve the statistical analyses, controls, sampling mechanism, or statistical reporting. Please number each suggestion so that the author(s) can more easily respond.

### Reviewer \#2

Yes.

1. Report Standard Deviations or Variability When Applicable:  
    For inference time and performance (Table 5), if multiple runs were conducted, it would be helpful to report mean ± std to quantify variability.
    If only a single run was reported, please mention this explicitly.
2. Add Sample Size Information for Inference Metrics:  
    Specify the number of videos or frames used to compute average inference time and FLOPs, even if the test set is standard (THUMOS14 test set).
3. Describe Any Randomness Handling (e.g., seeds):  
    Mention whether random seeds were fixed during training and evaluation to ensure deterministic results.
4. Clarify Statistical Validity of Comparison:  
    The paper compares C2MS-Net with other models from literature (Table 4). It would help to acknowledge that some comparisons may be under slightly different conditions (e.g., frame rate, hardware, training setup), unless all models were re-run under a common setting.
5. Justify Use of Thresholds for Filtering (α, β in Post-processing):  
    The post-processing step includes thresholds for filtering low-confidence predictions and short actions. A brief justification of how these thresholds were chosen (e.g., empirical tuning or prior work) would increase transparency.

### Reviewer \#3

Yes.

1. The manuscript presents results with appropriate evaluation metrics for temporal action localization (e.g., mAP at multiple IoU thresholds), which are standard in the field.
2. The statistical reporting is clear; however, including confidence intervals (CIs) for the reported mAP scores would improve the interpretation of performance variability.
3. If feasible, add significance testing (e.g., paired t-test or Wilcoxon signed-rank test) when comparing the proposed method to baselines to strengthen claims of superiority.
4. Provide more details on the sampling mechanism for dataset splits (train/validation/test) to ensure clarity on data partitioning and avoid potential data leakage.

### Reviewer \#5

yes.

## Q4: Could the manuscript benefit from additional tables or figures, or from improving or removing (some of the) existing ones?

Please provide specific suggestions for improvements, removals, or additions of figures or tables. Please number each suggestion so that author(s) can more easily respond.

### Reviewer \#2

1. Add a Summary Table for the Distillation Pipeline:  
The paper discusses three distinct stages of distillation (cross-structure, cross-modal, simplified convolution).
    - ➤ Suggest adding a table that outlines:
        - Stage number
        - Teacher & student model types
        - Modules involved
        - Loss functions used
        - Whether weights are frozen or trained

    This would help readers understand the training flow more easily.

2. Improve Figure Captions for Key Diagrams (e.g., Fig. 3, 5, 7):  
    Current captions are brief. Please include what the figure represents, what role it plays, and how it connects to the rest of the pipeline.
    - ➤ Example: “Figure 3: Cross-Structure Distillation. This figure shows the distillation process where the Transformer-based teacher guides a CNN-based student model.”

3. Add a Table Listing All Key Hyperparameters:  
    Many values (e.g., learning rate, dropout, batch size, λ, a, b, c) are used but not grouped anywhere.
    - ➤ A "Training Settings" table would improve clarity and reproducibility.

4. Improve Readability of Table 4 (Accuracy Comparison):  
    The table has a large number of entries; consider:
    - Boldfacing the best-performing methods
    - Adding horizontal lines after every 3–4 rows to visually group results
    - Highlighting your method (C2MS) in a distinct color/shading

5. Include More Visual Examples or Error Cases (Optional):  
    In Figure 8 (visual comparisons), add one more example:
    - A case where C2MS fails or misclassifies—this adds transparency and provides insights for future improvements.

6. Add a Graph Showing mAP vs. Inference Time Trade-Off (Optional):  
    To highlight the practical value of C2MS, you could plot:
    - X-axis: inference time (ms)
    - Y-axis: mAP (average of all tIoU)

    Each method (C2MS, ActionFormer, Lee et al.) as a point
    - ➤ This makes the efficiency-vs-performance trade-off visually intuitive.

7. Remove Redundant Captions or Duplicated Information in Tables (if any):  
Ensure that performance tables do not repeat exact information already in narrative unless it’s for quick lookup. Maintain conciseness.

### Reviewer \#3

The current figures are clear and visually informative; however, adding a comparative visual example of detection results (e.g., temporal action localization plots comparing baseline and proposed method outputs) would enhance interpretability.

Consider including a summary table of hyperparameters for quick reference, which would be helpful for reproducibility and clarity.

In tables reporting mAP and other metrics, highlight the best-performing results in bold and second-best in italic for easier comparison.

If space permits, add a diagram illustrating the knowledge distillation process between teacher and student networks to give readers a clearer conceptual view of the workflow.

Ensure that figure captions are self-contained, explaining all abbreviations and symbols without requiring the reader to refer back to the text.

### Reviewer \#5 

yes

## Q5: If applicable, are the interpretation of results and study conclusions supported by the data?

Please provide suggestions (if needed) to the author(s) on how to improve, tone down, or expand the study interpretations/conclusions. Please number each suggestion so that the author(s) can more easily respond.

### Reviewer \#2

Yes.

1. Temper the Strength of Claims on Performance Superiority:

In some parts (e.g., abstract, conclusion), the phrasing implies that the method outperforms all other approaches, but this is only true at lower tIoU thresholds.
➤ Suggest revising to:

“The proposed C2MS-Net performs competitively with or surpasses state-of-the-art methods, particularly at lower and mid-range tIoU thresholds, while offering substantial gains in computational efficiency.”

2. Acknowledge Slight Performance Drop at Higher Thresholds:

Table 4 shows that at tIoU = 0.6 and 0.7, C2MS performs slightly worse than Lee et al. [2].
➤ Recommend briefly discussing why this happens—possibly due to not using end-to-end training or reduced temporal resolution—and noting it as an avenue for future improvement.

3. Expand on Real-World Applicability:

You could enhance the conclusion by highlighting real-world use cases that would benefit from faster inference and lighter models (e.g., embedded systems, mobile video processing, real-time surveillance).

4. Discuss Limitations and Future Work More Explicitly:

A short paragraph outlining limitations (e.g., only evaluated on THUMOS14, not end-to-end, requires multi-stage training) would add balance.
➤ This improves the credibility and sets the stage for meaningful future research.

5. Quantify Training vs. Inference Trade-off in Conclusion:

You mention a 4× increase in training time and 40× improvement in inference speed in the discussion.
➤ Reiterating this in the conclusion with specific numbers gives weight to the practicality of the approach.

### Reviewer \#3

Yes.

The interpretations and conclusions are generally well supported by the experimental results presented in the manuscript.

### Reviewer \#5

yes.

## Q6: Have the authors clearly emphasized the strengths of their study/theory/methods/argument?

Please provide suggestions to the author(s) on how to better emphasize the strengths of their study. Please number each suggestion so that the author(s) can more easily respond.

### Reviewer \#2

1. Include a Dedicated “Contributions” Bullet List in the Introduction:
While the contributions are described, presenting them in a clear, numbered or bulleted list (e.g., “Our key contributions are:”) in the Introduction will help highlight strengths concisely and early on.

2. Reiterate Key Strengths in the Conclusion Section:
The conclusion should summarize the key takeaways and explicitly restate what sets this work apart (e.g., “C2MS-Net achieves ~50x inference speedup with comparable accuracy to SOTA models”).

3. Use Comparative Language When Presenting Results:
In tables or the Results section, emphasize the improvement over specific baselines using comparative statements:
E.g., “Compared to ActionFormer, our method achieves a 98% reduction in inference time while maintaining comparable mAP.”

4. Visually Highlight Strengths in Tables (Optional):
Use bold or shaded rows to draw attention to your method's strengths in tables (e.g., smallest model size, fastest inference, highest mAP at certain thresholds).

5. Highlight the Novelty of the Combined Distillation Strategy:
The use of both cross-modal and cross-structure distillation is a distinctive contribution.
➤ Consider emphasizing that prior work typically uses only one type of distillation or requires Transformer-based inference.

6. Frame the Efficiency Gains as Real-World Ready:
Emphasize that the model is deployment-friendly—can run on edge devices, mobile platforms, or real-time systems.
➤ This adds a practical dimension to the theoretical strength.

7. Use Visual Summary (Optional):
Consider adding a figure or graphical summary that shows:
Performance vs. inference time
Model size vs. accuracy
➤ This can succinctly reinforce the overall benefit of your approach.

### Reviewer \#3

The authors have highlighted the novelty of combining cross-modal and cross-structural knowledge distillation for temporal action localization, which is a clear strength of the work.

To further emphasize the contribution, consider adding a dedicated “Strengths and Contributions” subsection in the conclusion or introduction that explicitly lists the key innovations and advantages over prior methods.

Include a side-by-side visual comparison showing how the proposed approach improves over baselines in challenging cases (e.g., occlusions, background clutter) to make the advantages more tangible.

Quantify the efficiency benefits (e.g., reduced computation time, fewer parameters) alongside accuracy improvements to showcase practical applicability.

When discussing related work, highlight specific shortcomings of existing approaches and directly link them to how the proposed method addresses these gaps.

### Reviewer \#5

no.

## Q7: Have the authors clearly stated the limitations of their study/theory/methods/argument?

Please list the limitations that the author(s) need to add or emphasize. Please number each limitation so that author(s) can more easily respond.

### Reviewer \#2

1. Increased Training Time Due to Multi-Stage Distillation:
The proposed method requires three rounds of knowledge distillation, which increases training time by up to 4× compared to some end-to-end baselines.
➤ This may limit its practicality in rapid development or low-resource settings.

2. Evaluation on a Single Dataset (THUMOS14):
The model is only evaluated on THUMOS14, which is standard but relatively small and domain-specific.
➤ Generalizability to larger or more diverse datasets (e.g., ActivityNet, HACS) remains untested.

3. Lack of End-to-End Learning Pipeline:
The approach relies on pre-extracted features (I3D) rather than processing raw video frames end-to-end.
➤ This may miss low-level optimization opportunities and could limit performance on unseen modalities.

4. Slight Accuracy Drop at Higher tIoU Thresholds:
In comparative results (Table 4), the method slightly underperforms compared to Lee et al. [2] at higher tIoU levels (0.6, 0.7).
➤ This could be due to less temporal precision from the CNN-based student or removal of temporal modeling granularity.

5. Limited Ablation on Loss Weight Sensitivity:
While ablation studies cover architectural components well, sensitivity analysis of the distillation loss weights (λ, a, b, c) is missing.
➤ The performance might vary significantly based on these hyperparameters.

6. Pseudo-Optical Flow Quality Not Quantitatively Validated:
The model generates pseudo-optical flow from RGB features, but there’s no quantitative comparison with ground truth optical flow or its impact on downstream tasks (other than mAP).

7. Hardware-Specific Performance Gains May Not Generalize:
Reported inference time (49 ms) is based on specific hardware (RTX2080Ti).
➤ Real-world performance on edge devices or embedded platforms could vary.

### Reviewer \#3

The authors have provided a limitations section (page 29), noting that the current model is not end-to-end trained and that it relies on pre-trained I3D models without parameter adjustment, limiting adaptability. This is a valuable acknowledgment. However, the section could be strengthened by:

Discussing the limitation of dataset diversity, as current experiments focus on a few benchmarks, which may not fully represent real-world conditions.

Noting the computational complexity of the method and possible deployment challenges in resource-constrained environments.

Commenting on the generalizability to other modalities or noisy/incomplete input data.

Highlighting potential future work directions to address these limitations, giving readers a clearer research roadmap.

### Reviewer \#5 

bo

## Q8: Does the manuscript structure, flow or writing need improving (e.g., the addition of subheadings, shortening of text, reorganization of sections, or moving details from one section to another)?

Please provide suggestions to the author(s) on how to improve the manuscript structure and flow. Please number each suggestion so that author(s) can more easily respond.

### Reviewer \#2

1. Introduce a Clear "Key Contributions" Bullet List in the Introduction:
While the contributions are described in text, a standalone, numbered bullet list at the end of the introduction would enhance clarity and quickly orient readers to the paper’s core ideas.

2. Add a Subheading for Each Distillation Stage in the Method Section:
The description of cross-structure, cross-modal, and simplified convolution distillation stages is packed into a large section.
➤ Introduce subheadings for each stage to improve readability:
3.1 Cross-Structure Distillation
3.2 Cross-Modal Distillation
3.3 Simplified Convolution Distillation

3. Shorten Long Paragraphs with Dense Technical Content:
Several paragraphs (especially in the methodology) are technically dense and long, making them hard to digest.
➤ Break into smaller chunks with one core idea per paragraph and more intuitive transitions.

4. Clarify the Flow of Training vs Inference Pipeline:
The relationship between teacher and student models across different stages could be confusing.
➤ Consider adding a training pipeline diagram or table, and clearly delineate training vs inference in the text.

5. Move Implementation Details into a Dedicated Subsection:
Hardware, software libraries, and training environment info are scattered.
➤ Create a section like 4.3 Implementation Details under “Experiments” to consolidate this information.

6. Improve Transitions Between Sections (e.g., from Related Work to Method):
Use short transitional paragraphs at the beginning or end of each major section to guide the reader through the structure of the paper.

7. Create a Distinct “Limitations and Future Work” Section:
Right now, limitations are not clearly identified.
➤ Add a dedicated section near the conclusion to reflect critically on limitations and next steps.

8. Consider Rephrasing Redundant Phrases in Abstract and Intro:
The phrase “removes the need for optical flow” appears multiple times in the abstract and introduction.
➤ Consider varying wording to improve flow and avoid redundancy.

### Reviewer \#3

The manuscript is generally well-structured, but certain refinements could improve readability and logical flow:

Refine section transitions – Some sections jump abruptly between concepts. Adding short transition sentences will help the reader follow the narrative more smoothly.

Improve subheading clarity – Ensure each subheading clearly reflects the content of the section and guides the reader through the methodology, experiments, and results.

Condense overly descriptive parts – Certain paragraphs in the introduction and related work could be shortened by removing redundant phrases or merging similar points.

Reorganize results discussion – Integrating some discussion points directly alongside the presentation of corresponding results (rather than in a separate section) may make the interpretation more immediate and engaging.

Move technical details to supplementary material – If possible, shift highly detailed mathematical derivations or dataset descriptions to the appendix or supplementary material to maintain main-text readability.

Improve figure and table placement – Align figures and tables closer to the text where they are first referenced, ensuring minimal scrolling back and forth.

### Reviewer \#5

yes.


## Q9: Could the manuscript benefit from language editing?

Reviewer \#2: Yes

Reviewer \#3: Yes

Reviewer \#5: No

## Final

Associate Editor: Three reviewers provided feedback on the manuscript and were generally positive. However, they raised significant concerns regarding the unclear description of motivation and scientific gaps, lacking comparison with the state of the art, and unsatisfactory writing. The authors are suggested to address the reviewers' concerns in a revised submission.

### Reviewer \#2

1. Overall Assessment:
This manuscript presents a well-motivated and technically sound method, C2MS-Net, for efficient temporal action localization. The approach integrates cross-modal and cross-structure knowledge distillation, offering performance comparable to state-of-the-art models while significantly reducing inference time and model size.

2. Clarity of Contributions:
Please consider adding a bullet-point summary of contributions at the end of the Introduction to improve clarity and help readers quickly understand the novelty of your work.

3. Training Pipeline Explanation:
The training pipeline involves multiple distillation stages. Consider including a table or diagram summarizing the stages (e.g., which model is teacher/student, frozen/trained, loss used). This will help readers follow the distillation flow more easily.

4. Hyperparameter Transparency:
Clearly specify all loss weights (a, b, c, λ), learning rates, batch sizes, number of epochs, and any data augmentation or preprocessing applied to ensure reproducibility.

5. Limitations Section:
Add a short "Limitations and Future Work" section discussing points such as:
-Increased training time due to multi-stage distillation
-Generalization beyond THUMOS14
-Slight accuracy drop at higher tIoU thresholds

6. Improve Figure Captions and Subheadings:
Improve figure captions by making them more descriptive. Also, add subheadings for individual components in the Methodology section (e.g., Action Attention, MFTM, Distillation Stages).

7. Grammar and Flow:
The manuscript would benefit from professional language editing to correct minor grammatical issues, improve sentence flow, and reduce redundancy.

8. Additional Dataset Evaluation (Optional):
While THUMOS14 is a standard benchmark, results on an additional dataset (e.g., ActivityNet or HACS) would further validate the generality of the proposed approach.

### Reviewer \#3

This field is optional. If you have any additional suggestions beyond those relevant to the questions above, please number and list them here.

### Reviewer \#5

The manuscript entitle 'Enhancing Temporal Action Localization through Cross-Modal and Cross-Structural Knowledge Distillation' was well written. Experimental results also seems good. However authors need to do major revisions in the manuscript.

1. Motivation and research gap should be describe in the manuscript

2. Problems of the existing systems should be analysed

3. Authors should include survey table in section 2.

4. simulation setup of the proposed model (parameters and its range) should be tabulated

5. Authors should include comparison of the proposed model with the state of the art

6. Limitation of the study should be given in the manuscript

7. Language of the manuscript should be revised

8. To improve the strength of the manuscript, authors need to include the following article in the survey section

Sunil Kumar, V. , Renukadevi, S. , Yashaswini, B.M. , Malagi, V.P. , Pareek, P.K. , Feature Fusing with Vortex-Based Classification of Sentiment Analysis Using Multimodal Data , Lecture Notes in Electrical EngineeringThis link is disabled., 2024, 1104 LNEE, pp. 463-480.
Sagari, S.M. , Malagi, V.P. , Sasi, S. , Euri - A Deep Ensemble Architecture For Oral Lesion Segmentation And Detection , International Journal of Intelligent Systems and Applications in Engineering, 2024, 12(3s), pp. 242-249.
