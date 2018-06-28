# xyz Nov 2017

import numpy as np
import os,sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR+'/all_datasets_meta')
from datasets_meta import DatasetsMeta

def pos_mean(arr, axis=None):
    return np.sum( arr, axis ) / np.sum( arr>0, axis )

def nan_to_num( arr ):
    is_nan = np.isnan(arr)
    arr = np.nan_to_num(arr) - is_nan*1e-7
    return arr

class EvaluationMetrics():
    @staticmethod
    def get_class_accuracy(c_TP_FN_FP, numpoint_block, dataset_name, class_ls=None, IsIncludeAveClass=True):
        '''
            c_TP_FN_FP: [num_batch, batch_size,num_class,3]
        '''
        c_TP_FN_FP = c_TP_FN_FP.astype(np.float)
        TP = c_TP_FN_FP[...,0]  # [batch_size,class]
        FN = c_TP_FN_FP[...,1]
        FP = c_TP_FN_FP[...,2]
        num_batch = c_TP_FN_FP.shape[0]
        batch_size = c_TP_FN_FP.shape[1]
        total_num = numpoint_block * batch_size * num_batch
        block_acc = np.sum(TP,-1)/numpoint_block
        ave_block_acc = np.mean(block_acc)
        std_block_acc = np.std( block_acc )
        block_acc_histg = np.histogram( block_acc, bins=np.arange(0,1.2,0.1) )[0].astype(np.float32) / block_acc.size

        precision = np.nan_to_num(TP/(TP+FP))
        recall = np.nan_to_num(TP/(TP+FN))
        IOU = np.nan_to_num(TP/(TP+FN+FP))
        # class num weighted ave
        real_Pos = TP+FN
        normed_real_TP = real_Pos/np.sum(real_Pos)
        ave_class_num_weighted = {}
        ave_class_num_weighted['pre'] = np.sum( precision*normed_real_TP )
        ave_class_num_weighted['recall'] = np.sum( recall*normed_real_TP )  # is equal to ave_block_acc
        ave_class_num_weighted['IOU'] = np.sum( IOU*normed_real_TP )

        # class ave accuracy: do not include non point block
        class_non_zero_bn = np.sum( np.sum(real_Pos!=0,0),0 )
        class_precision = nan_to_num( np.sum(np.sum(precision,0),0)/class_non_zero_bn )
        class_recall = nan_to_num( np.sum(np.sum(recall,0),0)/class_non_zero_bn )
        class_IOU = nan_to_num( np.sum(np.sum(IOU,0),0)/class_non_zero_bn )

        ave_class = {}
        ave_class['pre'] = pos_mean(class_precision)
        ave_class['recall'] = pos_mean(class_recall)
        ave_class['IOU'] = pos_mean(class_IOU)

        # gen str
        delim = '' # ','
        def getstr(array,str_format='%0.2f,'):
            #if mean!=None:
            #    mean_str = '%5s'%(str_format%mean) + delim
            #else:
            #    mean_str = '%5s'%('  ')
            #    if delim != '': mean_str = mean_str + ' '
            return  delim.join(['%6s'%(str_format%v) for v in array])

        ave_class_acc_str = 'weighted class pre/rec/IOU: %0.3f  %0.3f  %0.3f  N=%fM  points ave/std:  %0.3f  %0.3f'% \
            ( ave_class_num_weighted['pre'], ave_class_num_weighted['recall'],
             ave_class_num_weighted['IOU'],total_num/1000000.0, ave_block_acc, std_block_acc)
        if IsIncludeAveClass:
            ave_class_acc_str += '\nclass ave pre/rec/IOU : %0.3f/ %0.3f/ %0.3f' %(
                        ave_class['pre'],ave_class['recall'],ave_class['IOU'])
        if class_ls != None:
            class_acc_str = ave_class_acc_str + '\n\t       average'+delim  + delim.join(['%9s'%c for c in class_ls])+'\n'
        else:
            class_acc_str = ''
        class_acc_str += 'class_pre: '+getstr(class_precision)+'\n'
        class_acc_str += 'class_rec: '+getstr(class_recall)+'\n'
        class_acc_str += 'class_IOU: '+getstr(class_IOU)+'\n'
        class_acc_str += 'number(K): '+getstr(np.trunc(np.sum(np.sum(real_Pos,0),0)/1000.0),str_format='%d,') + '\n'
        #class_acc_str += 'class  id: '+getstr(np.arange(precision.shape[1]),str_format='%d,') + '\n'

        label2class = DatasetsMeta.g_label2class[dataset_name]
        class_name_ls = [label2class[label][0:5] for label in np.arange(precision.shape[-1])]
        class_acc_str += 'classname: '+getstr( class_name_ls ,str_format='%s,')
        return ave_block_acc, std_block_acc, block_acc_histg,  class_acc_str,ave_class_acc_str

    @staticmethod
    def get_TP_FN_FP(NUM_CLASSES,pred_val,cur_label):
        assert pred_val.shape == cur_label.shape
        batch_size = pred_val.shape[0]
        c_TP_FN_FP = np.zeros(shape=(batch_size,NUM_CLASSES,3))
        for i in range(pred_val.shape[0]):
            for j in range(pred_val.shape[1]):
                p = pred_val[i,j]
                l = cur_label[i,j]
                c_TP_FN_FP[i,l,0] += p==l
                c_TP_FN_FP[i,l,1] += p!=l
                c_TP_FN_FP[i,p,2] += p!=l
        return c_TP_FN_FP

    @staticmethod
    def report_pred( file_name, pred_val, label_category, dataset_name):
        label2class = DatasetsMeta.g_label2class[dataset_name]
        pred_logit = np.argmax( pred_val,-1 )
        is_correct = label_category == pred_logit
        print('writing pred log:%s'%(file_name))
        with open(file_name,'w') as f:
            classes_str = ' '.join([label2class[l] for l in range(len(label2class))])+'\n'
            for i in range(pred_val.shape[0]):
                for j in range(pred_val.shape[1]):
                    if is_correct[i,j]:
                        str_ij = '  Y '
                    else:
                        str_ij = '! N '
                    str_ij += '%s'%( label2class[label_category[i,j]] ) + '\t'
                    score_order = np.argsort( pred_val[i,j] )
                    for k in range( min(7,len(score_order)) ):
                        label_k = score_order[len(score_order) - k -1]
                        str_ij += '%s:%0.1f'%(label2class[label_k],pred_val[i,j,label_k]) + '  '
                    str_ij += '\n'
                    f.write(str_ij)
                f.flush()
        print('finish pred log:%s'%(file_name))

