import os
import nibabel as nib
import numpy as np
import pickle
import xlsxwriter
import matplotlib
from statistics import mean
import matplotlib.pyplot as plt
#from sklearn.metrics.cluster import normalized_mutual_info_score as nmi
from math import log10, sqrt
from surface_distance import compute_average_surface_distance, compute_robust_hausdorff, compute_surface_distances


matplotlib.use('TKAgg')

# Path to where you saved decathlon dataset
nifti_dir = 'D:\\project_data\\spleen_dataset\\Task09_Spleen_f\\Task09_Spleen\\labelsTr\\'
# Path to where you saved inference results from the cluster
inference_dir = "D:\\project_data\\spleen_dataset\\Task09_Spleen_f\\results_train_inference"


all_cases = ["2", "3","6","8","9","10","12","13","14","16","17","18","19","20","21","22","24","25","26","27","28","29","31","32","33","38","40","41","44","45","46","47","49","52","53","56","59","60","61","62","63"]
test_cases = ["2", "3","6","8","9","10","12","13","14","16","17","18","19","20","21","22","24","25","26","27","28","29","31","32","33","38","40","41","44","45","46","47","49","52","53","56","59","60","61","62","63"]

print(test_cases)

full_list = []
[full_list.append('spleen_' + case_nr) for case_nr in test_cases]

def getLargestCC(segmentation):
labels = label(segmentation)assert( labels.max() != 0 ) # assume at least 1 CC    
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    return largestCC

metrics = ['Dice', 'JD', 'HD']
organs = ['Spleen']

print(f'length is:{len(metrics)}')

methods = ["spleen_seg_no_augment_epoch_1000", "spleen_seg_with_augment_epoch_1000"]  # create an empty list to store enumerated files

print(f'the list for methods is\n{methods}')

worksheet_names = ['Evaluation_no_aug','Evaluation_w_aug']

output_file_name = '\\seg_results_spleen_j.xlsx'

# Put here path to where save the output file
workbook = xlsxwriter.Workbook('D:\\project_data\\spleen_dataset\\Task09_Spleen_f\\Results_evaluation'+ output_file_name)

print('starting metrics calculation')
method_number = 0
for method in methods:

    print(f'the case we use is\n{method}')
    worksheet = workbook.add_worksheet(worksheet_names[method_number])
    method_number += 1

    worksheet.write(0, 0, 'Case nr')

    for il in range(0, len(organs)):
        for ill in range(0, len(metrics)):
            worksheet.write(2, 1 + il*3 + ill, organs[il] + '_' + metrics[ill])
    print(f'the ground truth data is\n{full_list}')
    global_metrics = [[] for i in range(3)]

    # for i in range(0, 1):   # validate dataset
    for i in range(0, len(full_list)):  # val dataset
        print('we will compare to')
        case_nr = full_list[i]
        print(full_list[i])
        worksheet.write(3+i, 0, case_nr)

        try:

            baseline_im_name = full_list[i] + '.nii.gz'
            baseline_im_path = nifti_dir + baseline_im_name

            case_name = full_list[i]
            seg_im_path = inference_dir + '\\' + method + '\\' + case_name + '\\' + case_name + '_seg.nii.gz'

            seg_im_nii = nib.load(seg_im_path)
            seg_im = np.asarray(seg_im_nii.get_data())
            seg_im_header = seg_im_nii.header
            pixdim = seg_im_header['pixdim']
            vdim = pixdim[1:4]

            seg_spleen_im = np.zeros_like(seg_im)
            seg_spleen_im[seg_im == 1] = 1

            seg_organs = [seg_spleen_im]

            ###################################################
            baseline_im_nii = nib.load(baseline_im_path)
            baseline_im = np.asarray(baseline_im_nii.get_data())

            baseline_spleen_im = np.zeros_like(baseline_im)
            baseline_spleen_im[baseline_im == 1] = 1

            baseline_organs = [baseline_spleen_im]

            print('Start calculating metrics')
            for il in range(len(organs)):

                for ill in range(len(metrics)):  # SSD_CT
                    organ_seg = seg_organs[il]
                    organ_baseline = baseline_organs[il]

                    if metrics[ill] == 'Dice':

                        dice = np.sum(organ_seg[organ_baseline == 1])*2.0 / (np.sum(organ_seg) + np.sum(organ_baseline))
                        worksheet.write(3 + i, 1 + il*3 + ill, round(dice*100, 3))
                        global_metrics[il*3 + ill].append(dice)

                    elif metrics[ill] == 'JD':

                        or_organ = organ_seg + organ_baseline
                        or_organ[or_organ > 1] = 1
                        jaccard = np.sum(organ_seg[organ_baseline == 1]) / (np.sum(or_organ))
                        worksheet.write(3 + i, 1 + il*3 + ill, round(jaccard*100, 2))
                        global_metrics[il*3 + ill].append(jaccard)

                    elif metrics[ill] == 'HD':

                        surface_distances = compute_surface_distances(np.array(organ_baseline, dtype=bool),
                                                                      np.array(organ_seg, dtype=bool),
                                                                      (vdim[0], vdim[1], vdim[2]))
                        hd = compute_robust_hausdorff(surface_distances, percent=95)
                        worksheet.write(3 + i, 1 + il*3 + ill, round(hd, 2))
                        global_metrics[il*3 + ill].append(hd)

        except:
            print('Generating distribution data from case: ', case_nr, ' failed.')

    global_metrics_names = ['Dice_spleen', 'JD_spleen', 'HD_spleen']

    for j in range(len(global_metrics_names)):
        if j < len(global_metrics):
            worksheet.write(3 + len(full_list) + 2, j + 1, 'AVG_' + global_metrics_names[j])
            if global_metrics[j]:
                worksheet.write(3 + len(full_list) + 3, j + 1, round(mean(global_metrics[j]), 3))

workbook.close()

print('Finito')
