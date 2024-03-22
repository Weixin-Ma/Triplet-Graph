# Author:   Weixin Ma       weixin.ma@connect.polyu.hk

import numpy as np
import os
from sklearn import metrics
from matplotlib import pyplot as plt


def fast_eval_new():
    score_result = np.genfromtxt("./score.txt", dtype='float32').reshape(-1, 3)

    label         = score_result[:, 2].reshape(-1, 1)
    score         = score_result[:, 0].reshape(-1, 1)
    refined_score = score_result[:, 1].reshape(-1, 1)

    #without projection selection
    precision, recall, pr_thresholds = metrics.precision_recall_curve(label, score)
    F1_score = 2 * precision * recall / (precision + recall + 1e-10)    
    F1_max_score = np.max(F1_score)

    #with projection selection
    precision_refine, recall_refine, pr_thresholds_refine = metrics.precision_recall_curve(label, refined_score)
    F1_score_refine = 2 * precision_refine * recall_refine / (precision_refine + recall_refine +1e-10)
    F1_max_score_refine = np.max(F1_score_refine)

    #draw curve
    if not os.path.exists('./evaluation_results'):
        os.mkdir('./evaluation_results')

    np.savetxt( "./evaluation_results/precision.txt", precision_refine , fmt = '%f', delimiter = ' ')
    np.savetxt( "./evaluation_results/recall.txt", recall_refine , fmt = '%f', delimiter = ' ')
    

    #######calculate Extended Precision
    #without projection selection
    EP = 0
    min_recall    = np.min(recall)
    min_recall_id = np.where(recall==np.min(recall))
    P_r0s         = precision[min_recall_id]
    P_r0          = np.max(P_r0s)
    if P_r0<1:
        R_p100        = 0.0
        EP            = (P_r0 + R_p100) * 0.5
        #print("min_recall=", min_recall, "min_recall_id=", min_recall_id, "P_r0=",P_r0, ", R_p100=", R_p100, ", EP=",EP)
    
    elif P_r0==1:
        max_precis_ids   = np.where(precision==np.max(precision))
        R_p100_s         = recall[max_precis_ids]
        R_p100           = np.max(R_p100_s)
        EP               = (P_r0 + R_p100) * 0.5
        #print("min_recall=", min_recall, "min_recall_id=", min_recall_id, "P_r0=",P_r0, ", R_p100=", R_p100, ", EP=",EP)
    
    #with projection selection
    EP_refine = 0
    min_recall_refine    = np.min(recall_refine)
    min_recall_refine_id = np.where(recall_refine==np.min(recall_refine))
    P_r0_refine          = precision_refine[min_recall_refine_id]
    P_r0_refine          =np.max(P_r0_refine)
    if P_r0_refine<1:
        R_p100_refine        = 0.0
        EP_refine            = (P_r0_refine + R_p100_refine) * 0.5
        #print("min_recall=", min_recall_refine, "min_recall_id=", min_recall_refine_id, "P_r0=",P_r0_refine, ", R_p100=", R_p100_refine, ", EP=",EP_refine)
    elif P_r0_refine==1:
        max_precis_refine_ids   = np.where(precision_refine==np.max(precision_refine))
        R_p100_refine_s         = recall_refine[max_precis_refine_ids]
        R_p100_refine           = np.max(R_p100_refine_s)
        EP_refine               = (P_r0_refine + R_p100_refine) * 0.5
        #print("min_recall=", min_recall_refine, "min_recall_id=", min_recall_refine_id, "P_r0=",P_r0_refine, ", R_p100=", R_p100_refine, ", EP=",EP_refine)

    #localization accuracy
    loop_amount= int(np.sum(label))

    #####for rte
    rte_result = np.genfromtxt("./rte.txt", dtype='float32').reshape(-1, 1)
    rte        = rte_result[:loop_amount, ]
    rte_mean   = np.mean(rte)
    rte_std    = np.std(rte)


    #####for rre
    rre_result=  np.genfromtxt("./rre.txt", dtype='float32').reshape(-1, 2) 
    rre_ori    = rre_result[:, 0]
    rre    = rre_ori[:loop_amount, ]
    rre_mean   = np.mean(rre)
    rre_std    = np.std(rre)

    print("Total pairs: ",label.shape[0])
    print(end='\n')
    print("F1 (w/o projection selection):", F1_max_score)
    print("F1 (w   projection selection):", F1_max_score_refine)
    print("EP (w/o projection selection):", EP)
    print("EP (w   projection selection):", EP_refine)    
    print("RTE: ", rte_mean, " ± ",rte_std)
    print("RRE: ", rre_mean, " ± ",rre_std)


    plt.plot(recall_refine, precision_refine, color='darkorange', lw=1.5, label='P-R curve')
    plt.axis([0, 1, 0, 1])
    plt.title('Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc="lower right")
    plt.show()
    
if __name__ == '__main__':
    fast_eval_new()

