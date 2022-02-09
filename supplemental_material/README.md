# Supplemental Material
This directory contains the supplemental material of the paper "Object size prediction from hand movement using a single RGB sensor". Namely, the following information is presented below:
- the confusion matrices for each movement completion interval (20%, 40%, 60%, 80%, 100%), feature set (Ws-D, Ws-I) and data-split strategy (all-in, one-out).
- the feature importance heatmaps for each movement completion interval (20%, 40%, 60%, 80%, 100%) and feature set (Ws-D, Ws-I), as computed for the ET model and the "all-in" data split.
- the progression of each kinematic variable (TI-Ap, TM-Ap, IM-Ap, Wrist x-coordinate, Wrist y-coordinate) over the R-t-G movement.
- the accuracy boxlots for each movement completion interval (20%, 40%, 60%, 80%, 100%), feature set (Ws-D, Ws-I) and data-split strategy (all-in, one-out).
- a table containing the average and the standard deviation of the accuracy rates for each movement completion interval (20%, 40%, 60%, 80%, 100%), feature set (Ws-D, Ws-I) and data-split strategy (all-in, one-out).

## Confusion Matrices

    "all-in" data split
    Ws-D feature set

<p align="center">
<img src="assets/conf_mtx_wsd_all-in.png"></img>
</p>


    "all-in" data split
    Ws-I feature set

<p align="center">
<img src="assets/conf_mtx_wsi_all-in.png"></img>
</p>

    "one-out" data split
    Ws-D feature set

<p align="center">
<img src="assets/conf_mtx_wsd_one-out.png"></img>
</p>


    "one-out" data split
    Ws-I feature set

<p align="center">
<img src="assets/conf_mtx_wsi_one-out.png"></img>
</p>


## Feature Importances

    Ws-D feature set

<p align="center">
<img src="assets/feature_importance_wsd.png"></img>
</p>

    Ws-I feature set

<p align="center">
<img src="assets/feature_importance_wsi.png"></img>
</p>
