

import argparse
import glob
import os.path as osp


def parse_result(eval_folder, keyword= ': INFO  * Acc@1 ' ):
    first_term, second_term = None, None
    for line in open(osp.join(eval_folder, 'log_rank0.txt')):
        if keyword in line:
            if keyword == ': INFO  * Acc@1 ':
                items = line.strip('\n').split(' ')
                # top1_acc, top5_acc = float(items[-3]), float(items[-1])
                first_term, second_term = float(items[-3]), float(items[-1])
            elif keyword == ': INFO  * mAP ':
                items = line.strip('\n').split(' ')
                first_term = float(items[-1])
                second_term = None
            break
    return first_term, second_term


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--main_dir', '-dir', required= True, type=str,  )
    args = parser.parse_args()

    main_dir = args.main_dir

    eval_dir_list = glob.glob(osp.join( main_dir, 'eval*/' ))

    eval_setting_keywords = [
        # 'k400',
                             'ucf_split1', 'ucf_split2', 'ucf_split3',
                             'hmdb_split1', 'hmdb_split2', 'hmdb_split3',
                             'k600_split1', 'k600_split2', 'k600_split3',
                             'minissv2',
                             'charades',
                            'uavhuman_split1', 'uavhuman_split2',
                            'momentsintime',

                     ]

    eval_name_result_top1_dict = dict()
    eval_name_result_top5_dict = dict()
    for eval_dir in eval_dir_list:
        eval_folder_name = eval_dir.split('/')[-2]
        if 'charades' in eval_folder_name:
            top1_acc, top5_acc = parse_result(eval_dir, keyword=': INFO  * mAP ')
        else:
            top1_acc, top5_acc = parse_result(eval_dir, keyword=': INFO  * Acc@1 ')
        items = eval_folder_name.split('_')
        if '_v' in eval_folder_name:
            eval_name = '_'.join(items[1:-1])
        else:
            eval_name = '_'.join(items[1:])
        if eval_name in eval_setting_keywords:
            # print(eval_name)
            eval_name_result_top1_dict.update({eval_name: top1_acc})
            eval_name_result_top5_dict.update({eval_name: top5_acc})
    f_write = open(osp.join( main_dir, 'eval_summary.txt' ), 'w+')
    idx = 0
    n_results = len(eval_setting_keywords)

    results_to_print_top1 = []
    results_to_print_top5 = []
    while idx < n_results:
        if idx in [11]:
            avg_among = 2
            keyword_list = [eval_setting_keywords[idx + i] for i in range(avg_among)]
            f_write.write(' '.join(keyword_list) + ' avg' + '\n')
            result_list_top1 = [eval_name_result_top1_dict[eval_setting_keywords[idx + i]] for i in range(avg_among)]
            avg_top1 = sum(result_list_top1) / len(result_list_top1)
            result_list_top5 = [eval_name_result_top5_dict[eval_setting_keywords[idx + i]] for i in range(avg_among)]
            avg_top5 = sum(result_list_top5) / len(result_list_top5)
            f_write.write(f'{avg_top1:.3f}\n')
            results_to_print_top1.append(f'{avg_top1:.3f}')
            results_to_print_top5.append(f'{avg_top5:.3f}')
            idx += avg_among
        elif idx in [0, 3, 6]:
            avg_among = 3
            keyword_list = [eval_setting_keywords[idx + i] for i in range(avg_among)]
            f_write.write(' '.join(keyword_list) + ' avg' + '\n')
            result_list_top1 = [eval_name_result_top1_dict[eval_setting_keywords[idx + i]] for i in range(avg_among)]
            avg_top1 = sum(result_list_top1) / len(result_list_top1)
            result_list_top5 = [eval_name_result_top5_dict[eval_setting_keywords[idx + i]] for i in range(avg_among)]
            avg_top5 = sum(result_list_top5) / len(result_list_top5)
            f_write.write(f'{avg_top1:.3f}\n')
            results_to_print_top1.append(f'{avg_top1:.3f}')
            results_to_print_top5.append(f'{avg_top5:.3f}')
            idx += avg_among

        elif idx in [9,10,13 ]:
            keyword_ = eval_setting_keywords[idx]
            result_top1 = eval_name_result_top1_dict[keyword_]
            f_write.write(keyword_ + '\n')
            f_write.write(str(result_top1) + '\n')
            results_to_print_top1.append(str(result_top1))

            result_top5 = eval_name_result_top5_dict[keyword_]
            results_to_print_top5.append(str(result_top5))
            idx += 1
    f_write.write('\n')
    f_write.write(' '.join(results_to_print_top1[:7]) + '\n')
    # f_write.write(' '.join(results_to_print_top1[8:]) + '\n')
    f_write.write('\n')
    f_write.write('Top5\n')
    f_write.write(' '.join(results_to_print_top5[:7]) + '\n')
    # f_write.write(' '.join(results_to_print_top5[8:]) + '\n')
    f_write.close()
    t = 1