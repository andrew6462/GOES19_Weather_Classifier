# Evaluation

The project was evaluated as a binary classification problem: given a GOES patch, predict whether it should be labeled clear or cloudy. The main question was whether the later systems, especially the raw-patch CNN, improved on the earlier feature-based baselines under a fair held-out scene split.

## Evaluation Protocol

The primary quantitative evaluation used a grouped scene split. For both the Experiment D baseline and the CNN, the test set was created with `GroupShuffleSplit(test_size=0.25, random_state=42)` grouped by `scene_key`, which yielded 79,725 held-out patches. The CNN then used a second grouped split on the remaining scenes to create a validation set for early stopping and threshold tuning. This means the final test numbers come from scenes that were not seen during either fitting or threshold selection.

The main metrics were accuracy, cloud precision, cloud recall, and cloud F1. Because this is a practical weather classification task, false positives and false negatives were also examined directly through confusion-derived counts and rates. The frozen baseline from the smaller early dataset is still included for historical context, but the fairest comparison for the final method is Experiment D versus the CNN because they share the same cleaned-label patch definition and the same 79,725-patch test split.

## Quantitative Results

| Run | Test Patches | Threshold | Accuracy | Cloud Precision | Cloud Recall | Cloud F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Frozen baseline (small early dataset) | 2,646 | 0.50 | 0.8919 | 0.9332 | 0.8572 | 0.8936 |
| Baseline on full grouped dataset | 116,142 | 0.50 | 0.8505 | 0.8867 | 0.8202 | 0.8522 |
| Experiment A | 116,142 | 0.45 | 0.8542 | 0.8755 | 0.8425 | 0.8586 |
| Experiment B | 116,142 | 0.45 | 0.8522 | 0.8803 | 0.8319 | 0.8554 |
| Experiment C | 89,405 | 0.50 | 0.9069 | 0.9378 | 0.8868 | 0.9116 |
| Experiment D | 79,725 | 0.50 | 0.9157 | 0.9428 | 0.8979 | 0.9198 |
| Earlier CNN run | 79,725 | 0.50 | 0.9421 | 0.9557 | 0.9360 | 0.9457 |
| Latest CNN run | 79,725 | 0.45 | 0.9477 | 0.9434 | 0.9606 | 0.9519 |

## False Positives And False Negatives

The strongest apples-to-apples comparison is Experiment D versus the latest CNN because both evaluate the same task on the same split.

| Model | False Positives | False Positive Rate | False Negatives | False Negative Rate |
| --- | ---: | ---: | ---: | ---: |
| Experiment D | 2,338 | 6.36% | 4,386 | 10.21% |
| Latest CNN | 2,477 | 6.74% | 1,693 | 3.94% |

Relative to Experiment D, the latest CNN produced 139 more false positives but 2,693 fewer false negatives. This is the central tradeoff in the final model: it is slightly more willing to call a patch cloudy, but it misses far fewer true cloudy patches. For this project, that trade was worthwhile because recall and F1 improved sharply while precision remained almost unchanged.

## What The Results Showed

The experiment ladder established that the baseline improved when the labels were cleaned and when larger 96x96 patches were used. Experiment D was the best logistic-regression model, reaching 0.9157 accuracy and 0.9198 cloud F1. That result justified using the Experiment D data definition as the reference point for the later CNN.

The earlier CNN already surpassed Experiment D by a large margin, which showed that raw spatial context mattered. The latest CNN run improved again, reaching 0.9477 accuracy and 0.9519 cloud F1 on the shared test split. Compared with Experiment D, the latest CNN improved cloud recall from 0.8979 to 0.9606 and improved F1 from 0.9198 to 0.9519. Compared with the earlier CNN, the latest run gave up some precision but improved recall, overall accuracy, and F1. The latest run also trained longer, stopping after 13 epochs instead of 5, and selected a threshold of 0.45 from validation F1 search rather than using a fixed 0.50 threshold.

## Did The Work Meet The Project Expectations?

Yes. The project goal was not only to build a classifier, but to improve it in a way that could be defended quantitatively. The final CNN met that expectation because it outperformed every feature-based baseline and the previous CNN result on the most relevant held-out split. The improvement was especially strong on cloudy-patch recall, which indicates that the model learned useful spatial information that the hand-crafted summary statistics did not capture.

One caution is that the final gain is not purely architectural. The latest CNN also benefits from validation-based threshold selection, which moved the operating point toward higher recall. That is still a legitimate part of the system design, but it should be reported clearly so that the recall gain is not misattributed solely to the network architecture.

## Scalability And Remaining Limits

The final comparable CNN dataset contained 316,466 cleaned 96x96 patches, split into 188,792 training patches, 47,949 validation patches, and 79,725 test patches. The latest run completed successfully with batch size 512 in the local `tf-metal` environment, which suggests the current approach scales to datasets of at least this size on the available hardware.

At the same time, the evaluation is still limited to the currently matched scene collection. The project has not yet measured how performance changes when more days, more seasons, other cloud regimes, or additional geographic domains are added. Those are the next scale-up questions, but within the present dataset the final method clearly outperformed the earlier baselines.
