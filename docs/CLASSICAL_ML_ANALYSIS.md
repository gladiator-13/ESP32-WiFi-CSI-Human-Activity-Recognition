# Classical Machine-Learning Analysis

## ESP32 Wi-Fi CSI Human Activity Recognition

**Experiment:** Within-location evaluation\
**Classes:** 20 activities (A01--A20)\
**Input to classical models:** 468 handcrafted features per window\
**Test set:** 2,375 windows\
**Window configuration:** 200 samples per window, 50% overlap

------------------------------------------------------------------------

## 1. Experiment summary

Two classical machine-learning baselines were trained using handcrafted
statistical features extracted from normalized CSI windows:

-   Random Forest
-   RBF-kernel Support Vector Machine (SVM)

Hyperparameters were selected using validation macro-F1. The selected
models were then evaluated on the untouched test set.

### Selected configurations

  --------------------------------------------------------------------------
  Model                 Selected configuration           Validation macro-F1
  --------------------- ----------------------- ----------------------------
  Random Forest         `n_estimators=300`,                           0.1978
                        `max_depth=None`,       
                        `min_samples_leaf=1`,   
                        `max_features=sqrt`     

  SVM                   `C=1`, `gamma=0.01`,                          0.1950
                        RBF kernel              
  --------------------------------------------------------------------------

The SVM used a `StandardScaler` within its training pipeline.

## 2. Overall test performance

  Metric              Random Forest      SVM
  ----------------- --------------- --------
  Accuracy                   0.2131   0.1958
  Macro precision            0.2083   0.1904
  Macro recall               0.2133   0.1967
  Macro F1-score         **0.2075**   0.1919

Random Forest achieved the higher test macro-F1 by approximately **1.56
percentage points**.

The difference is modest. These results establish Random Forest as the
stronger of the two current classical baselines, but neither model
provides strong 20-class recognition performance.

## 3. Per-activity F1 comparison

  Class   Activity               Random Forest F1   SVM F1
  ------- -------------------- ------------------ --------
  A01     Standing                          0.244    0.267
  A02     Sitting                           0.207    0.220
  A03     Walk                              0.413    0.455
  A04     Jog in place                      0.355    0.335
  A05     Turn around                       0.418    0.349
  A06     Sit down, stand up                0.314    0.240
  A07     Jump in place                     0.197    0.141
  A08     Bend and pick up                  0.149    0.197
  A09     Lunges                            0.372    0.329
  A10     Wave                              0.115    0.126
  A11     Clap                              0.150    0.177
  A12     Boxing                            0.010    0.030
  A13     Arms up and down                  0.073    0.036
  A14     Push forward                      0.195    0.204
  A15     Arm circles                       0.117    0.070
  A16     Typing                            0.216    0.197
  A17     Phone call                        0.138    0.144
  A18     Writing notes                     0.211    0.193
  A19     Reading a book                    0.167    0.129
  A20     Combing hair                      0.090    0.000

### Observations

-   A03 (Walk) was among the strongest classes for both models.
-   A05 (Turn around) and A09 (Lunges) were relatively stronger for
    Random Forest.
-   A12 (Boxing), A13 (Arms up and down), A15 (Arm circles), and A20
    (Combing hair) were difficult for at least one model.
-   SVM had zero F1 for A20 in this test evaluation.
-   No model was consistently better for every activity.

These are descriptive observations from this test set. They do not, by
themselves, establish why an activity is difficult.

## 4. Confusion-matrix analysis

The confusion matrices were row-normalized for visualization. Each row
represents an actual activity, and each column represents a predicted
activity. The diagonal represents correct predictions; off-diagonal
cells represent errors.

The numerical analysis reports **directed confusions**. For example, A13
predicted as A14 is distinct from A14 predicted as A13.

### 4.1 Strongest global directed confusions

#### Random Forest

  -------------------------------------------------------------------------
  Actual activity Predicted              Misclassified Percentage of actual
                  activity                     windows                class
  --------------- --------------- -------------------- --------------------
  A13 Arms up and A14 Push                          42               31.58%
  down            forward                              

  A02 Sitting     A01 Standing                      30               26.32%

  A17 Phone call  A11 Clap                          25               21.93%

  A19 Reading a   A01 Standing                      25               21.93%
  book                                                 

  A07 Jump in     A09 Lunges                        24               18.05%
  place                                                

  A07 Jump in     A04 Jog in                        23               17.29%
  place           place                                

  A10 Wave        A13 Arms up and                   23               20.18%
                  down                                 

  A04 Jog in      A07 Jump in                       22               19.30%
  place           place                                

  A18 Writing     A16 Typing                        22               19.30%
  notes                                                

  A20 Combing     A10 Wave                          22               19.30%
  hair                                                 

  A09 Lunges      A07 Jump in                       21               18.42%
                  place                                

  A19 Reading a   A18 Writing                       20               17.54%
  book            notes                                

  A06 Sit down,   A04 Jog in                        19               14.29%
  stand up        place                                

  A11 Clap        A20 Combing                       19               16.67%
                  hair                                 

  A19 Reading a   A02 Sitting                       19               16.67%
  book                                                 

  A03 Walk        A09 Lunges                        18               15.79%

  A12 Boxing      A04 Jog in                        18               15.79%
                  place                                

  A12 Boxing      A20 Combing                       18               15.79%
                  hair                                 

  A16 Typing      A15 Arm circles                   18               15.79%

  A02 Sitting     A16 Typing                        17               14.91%
  -------------------------------------------------------------------------

#### SVM

  -------------------------------------------------------------------------
  Actual activity Predicted              Misclassified Percentage of actual
                  activity                     windows                class
  --------------- --------------- -------------------- --------------------
  A13 Arms up and A14 Push                          45               33.83%
  down            forward                              

  A16 Typing      A19 Reading a                     31               27.19%
                  book                                 

  A19 Reading a   A18 Writing                       31               27.19%
  book            notes                                

  A07 Jump in     A04 Jog in                        26               19.55%
  place           place                                

  A01 Standing    A19 Reading a                     25               18.80%
                  book                                 

  A09 Lunges      A06 Sit down,                     25               21.93%
                  stand up                             

  A20 Combing     A15 Arm circles                   25               21.93%
  hair                                                 

  A17 Phone call  A11 Clap                          22               19.30%

  A11 Clap        A15 Arm circles                   21               18.42%

  A18 Writing     A16 Typing                        21               18.42%
  notes                                                

  A20 Combing     A10 Wave                          21               18.42%
  hair                                                 

  A02 Sitting     A01 Standing                      20               17.54%

  A04 Jog in      A07 Jump in                       20               17.54%
  place           place                                

  A19 Reading a   A02 Sitting                       20               17.54%
  book                                                 

  A18 Writing     A02 Sitting                       19               16.67%
  notes                                                

  A02 Sitting     A16 Typing                        18               15.79%

  A06 Sit down,   A04 Jog in                        18               13.53%
  stand up        place                                

  A07 Jump in     A09 Lunges                        18               13.53%
  place                                                

  A12 Boxing      A06 Sit down,                     18               15.79%
                  stand up                             

  A13 Arms up and A11 Clap                          18               13.53%
  down                                                 
  -------------------------------------------------------------------------

## 5. Recurring confusion patterns and interpretation

### 5.1 A13 (Arms up and down) versus A14 (Push forward)

This is the strongest directed confusion for both models:

-   Random Forest: 42 windows (31.58% of actual A13 windows).
-   SVM: 45 windows (33.83% of actual A13 windows).

This recurring error makes the A13-to-A14 distinction a high-priority
case for further investigation. Both activities involve arm movement,
but the confusion matrices alone cannot establish the physical or
signal-level cause.

### 5.2 Dynamic activities

Examples include:

-   A07 Jump in place → A04 Jog in place.
-   A04 Jog in place → A07 Jump in place.
-   A07 Jump in place → A09 Lunges.
-   A09 Lunges → A07 Jump in place or A06 Sit down, stand up.
-   A06 Sit down, stand up → A04 Jog in place.

Both models show confusion among several dynamic activities. This
suggests that the current handcrafted representation does not
consistently separate all of these activity classes.

### 5.3 Static posture and daily-life activities

Examples include:

-   A02 Sitting → A01 Standing.
-   A02 Sitting → A16 Typing.
-   A19 Reading a book → A18 Writing notes.
-   A18 Writing notes → A16 Typing.
-   A16 Typing → A19 Reading a book (particularly for SVM).

These classes may share aspects of posture or movement. Props and
recording conditions may also influence CSI measurements, but the
current model results do not isolate those effects.

### 5.4 Arm and hand movements

Examples include:

-   A10 Wave → A13 Arms up and down.
-   A11 Clap → A15 Arm circles (SVM).
-   A20 Combing hair → A10 Wave or A15 Arm circles.
-   A17 Phone call → A11 Clap.

These confusions indicate that several arm/hand activity classes remain
difficult for the handcrafted-feature baselines.

## 6. Main findings

1.  Random Forest is the stronger overall classical baseline on this
    within-location test set.
2.  SVM performs better for some individual activities, including A03
    Walk, A01 Standing, A02 Sitting, A08 Bend and pick up, A10 Wave, A11
    Clap, A12 Boxing, A14 Push forward, and A17 Phone call.
3.  A13 → A14 is the most prominent shared directed confusion in the
    supplied global rankings.
4.  Dynamic activities such as jogging, jumping, lunges, and
    sit-down/stand-up are frequently confused with one another.
5.  Several daily-life activities, particularly typing, writing notes,
    and reading, are also confused.
6.  Per-class performance is uneven, so macro-F1 and per-class recall/F1
    remain important metrics.

## 7. Important evaluation caveats

-   The test set was used for final evaluation and error analysis, not
    hyperparameter selection.
-   The 2-second windows overlap by 50%. Consequently, neighboring
    windows from the same recording are correlated and should not be
    treated as independent observations.
-   The confusion counts are window-level counts.
-   These results do not establish the cause of any confusion. Further
    signal inspection and controlled experiments are needed.
-   The current results are within-location results. They do not
    establish cross-location generalization.
-   Recording-level aggregation should be considered as a complementary
    evaluation, while retaining the existing held-out test protocol.

## 8. Recommended next experiments

### 8.1 1D-CNN

Train a 1D-CNN directly on normalized CSI windows with shape:

`(N, 200, 52)`

This allows the model to learn temporal and subcarrier patterns from the
CSI sequence instead of relying only on the 468 handcrafted statistical
features.

### 8.2 CNN-BiLSTM-Attention

Evaluate the planned CNN-BiLSTM-Attention architecture to combine local
convolutional patterns with temporal dependencies and attention over the
sequence.

### 8.3 Preserve experimental rigor

-   Keep the existing participant-wise train/validation/test separation.
-   Select hyperparameters and checkpoints using validation data only.
-   Do not tune models on the test set.
-   Fit any learned preprocessing transformations on training data only.
-   Report accuracy, macro precision, macro recall, macro-F1, per-class
    metrics, and confusion matrices.
-   Evaluate the LOLO splits separately to measure cross-location
    generalization.
-   Consider recording-level aggregation as a complementary result
    because windows overlap.

------------------------------------------------------------------------

## 9. Current milestone status

  Component                                Status
  ---------------------------------------- ----------
  CSI preprocessing and QA                 Complete
  Participant-wise splits                  Complete
  Window generation and validation         Complete
  Training-only normalization              Complete
  Handcrafted feature extraction           Complete
  Random Forest within-location baseline   Complete
  SVM within-location baseline             Complete
  Classical model comparison               Complete
  Numerical confusion analysis             Complete
  1D-CNN                                   Next
  CNN-BiLSTM-Attention                     Planned
  LOLO model evaluation                    Pending

**Current best classical baseline:** Random Forest, test macro-F1 =
**0.2075 (20.75%)**.

The next planned milestone is training the 1D-CNN on the normalized CSI
time-series windows.
