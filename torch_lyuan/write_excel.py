#----------------description----------------# 
# Author       : ,: Zihao Zhao
# E-mail       : ,: zhzhao18@fudan.edu.cn
# Company      : ,: Fudan University
# Date         : ,: 2020-10-23 14:12:06
# LastEditors  : Zihao Zhao
# LastEditTime : 2020-11-02 19:45:43
# FilePath     : /speech-to-text-wavenet/torch_lyuan/write_excel.py
# Description  : ,: 
#-------------------------------------------# 
import os
import numpy as np
import xlwt
import xlrd
from xlutils.copy import copy

def write_excel(excel_name, exp_name, train_loss_list, val_loss_list):
    # train_loss_list = [1.32, 1.543, 1.111, 1.098]
    # val_loss_list = [1.32, 1.543, 1.111, 1.098]

    if not os.path.exists(excel_name):
        base_row = 0
        wb = xlwt.Workbook(encoding='ascii')
        ws = wb.add_sheet('sheet1')
    else:
        base_row = blank_raw(excel_name)
        data = xlrd.open_workbook(excel_name, formatting_info=True)
        wb = copy(wb=data)
        ws = wb.get_sheet(0)

    name_row = base_row + 1
    epoch_row = base_row + 2
    train_loss_row = base_row + 3
    val_loss_row = base_row + 4

    ws.write(name_row, 0, exp_name)
    ws.write(epoch_row, 0, 'epoch')
    ws.write(train_loss_row, 0, 'train_loss')
    ws.write(val_loss_row, 0, 'val_loss')

    epoch_list = range(len(train_loss_list))
    for i, e in enumerate(epoch_list):
        ws.write(epoch_row, i+1, e)

    for i, t in enumerate(train_loss_list):
        ws.write(train_loss_row, i+1, t)

    for i, v in enumerate(val_loss_list):
        if v == np.array(val_loss_list).min():
            style = xlwt.XFStyle() 
            font = xlwt.Font() 
            font.colour_index = 2
            style.font = font
            ws.write(val_loss_row, i+1, v, style)
        else:
            ws.write(val_loss_row, i+1, v)

    wb.save(excel_name)
    print("results saved in", excel_name)


def write_pattern_count(excel_name, exp_name, nnz_list, count_list):
    # train_loss_list = [1.32, 1.543, 1.111, 1.098]
    # val_loss_list = [1.32, 1.543, 1.111, 1.098]

    if not os.path.exists(excel_name):
        base_row = 0
        wb = xlwt.Workbook(encoding='ascii')
        ws = wb.add_sheet('sheet1')
    else:
        base_row = blank_raw(excel_name)
        data = xlrd.open_workbook(excel_name, formatting_info=True)
        wb = copy(wb=data)
        ws = wb.get_sheet(0)

    name_row = base_row + 1
    ptid_row = base_row + 2
    nnz_row = base_row + 3
    count_row = base_row + 4

    style = xlwt.XFStyle() 
    font = xlwt.Font() 
    font.colour_index = 2
    style.font = font
    ws.write(name_row, 0, exp_name, style)
    ws.write(ptid_row, 0, 'epoch')
    ws.write(nnz_row, 0, 'pattern_nnz')
    ws.write(count_row, 0, 'pattern_count')

    ptid_list = range(len(count_list))
    for i, e in enumerate(ptid_list):
        ws.write(ptid_row, i+1, int(e))

    for i, t in enumerate(nnz_list):
        ws.write(nnz_row, i+1, int(t))
    for i, t in enumerate(count_list):
        ws.write(count_row, i+1, int(t))

    wb.save(excel_name)
    print("results saved in", excel_name)


def write_pattern_curve_analyse(excel_name, exp_name, patterns, pattern_match_num, pattern_coo_nnz, pattern_nnz,
                                                        pattern_num_memory_dict, pattern_num_coo_nnz_dict):
    # train_loss_list = [1.32, 1.543, 1.111, 1.098]
    # val_loss_list = [1.32, 1.543, 1.111, 1.098]

    # print(pattern_num_memory_dict)
    if not os.path.exists(excel_name):
        base_row = 0
        wb = xlwt.Workbook(encoding='ascii')
        ws = wb.add_sheet('sheet1')
    else:
        base_row = blank_raw(excel_name)
        data = xlrd.open_workbook(excel_name, formatting_info=True)
        wb = copy(wb=data)
        ws = wb.get_sheet(0)

    name_row = base_row + 1
    ptid_row = base_row + 2
    match_num_row = base_row + 3
    coo_nnz_row = base_row + 4
    nnz_num_row = base_row + 5

    pattern_num_row = base_row + 7
    pattern_num_memory_row = base_row + 8
    pattern_num_coo_nnz_row = base_row + 9



    ws.write(name_row, 0, exp_name)
    ws.write(ptid_row, 0, 'pattern id')
    ws.write(match_num_row, 0, 'match_num')
    ws.write(coo_nnz_row, 0, 'coo_nnz')
    ws.write(nnz_num_row, 0, 'nnz_num')

    ws.write(pattern_num_row, 0, 'pattern_num')
    ws.write(pattern_num_memory_row, 0, 'pattern_num_memory')
    ws.write(pattern_num_coo_nnz_row, 0, 'pattern_num_coo_nnz')

    ptid_list = range(len(patterns))
    for i, e in enumerate(ptid_list):
        ws.write(ptid_row, i+1, int(e))
    for i, t in enumerate(pattern_match_num):
        ws.write(match_num_row, i+1, int(t))
    for i, t in enumerate(pattern_coo_nnz):
        ws.write(coo_nnz_row, i+1, int(t))
    for i, t in enumerate(pattern_nnz):
        ws.write(nnz_num_row, i+1, int(t))

    # ptnum_list = range(len(pattern_num_memory_dict))
    for i, p_num in enumerate(pattern_num_memory_dict.keys()):
        ws.write(pattern_num_row, i+1, int(p_num))
        ws.write(pattern_num_memory_row, i+1, int(pattern_num_memory_dict[p_num]))
        ws.write(pattern_num_coo_nnz_row, i+1, int(pattern_num_coo_nnz_dict[p_num]))

    wb.save(excel_name)
    print("results saved in", excel_name)


def write_test_acc(excel_name, exp_name, 
                        f1, val_loss, tps, preds, poses):
    # train_loss_list = [1.32, 1.543, 1.111, 1.098]
    # val_loss_list = [1.32, 1.543, 1.111, 1.098]

    # print(pattern_num_memory_dict)
    if not os.path.exists(excel_name):
        base_row = 0
        ws.write(base_row, 0, 'exp_name')
        ws.write(base_row, 1, 'f1')
        ws.write(base_row, 2, 'val_loss')
        ws.write(base_row, 3, 'tps')
        ws.write(base_row, 4, 'preds')
        ws.write(base_row, 5, 'poses')
        wb = xlwt.Workbook(encoding='ascii')
        ws = wb.add_sheet('sheet1')
    else:
        base_row = blank_raw(excel_name)
        data = xlrd.open_workbook(excel_name, formatting_info=True)
        wb = copy(wb=data)
        ws = wb.get_sheet(0)
        ws.write(base_row, 0, exp_name)
        ws.write(base_row, 1, float(f1))
        ws.write(base_row, 2, float(val_loss))
        ws.write(base_row, 3, int(tps))
        ws.write(base_row, 4, int(preds))
        ws.write(base_row, 5, int(poses))

    wb.save(excel_name)
    print("results saved in", excel_name)



def blank_raw(excel_name):
    wb = xlrd.open_workbook(excel_name)
    sheet1 = wb.sheet_by_index(0)
    rowNum = sheet1.nrows
    return rowNum



if __name__ == "__main__":

    train_loss_list = [1.32, 1.543, 1.111, 1.098]
    val_loss_list = [1.32, 1.543, 1.111, 1.098]
    for i in range(4):
        write_excel("test.xls", str(i), train_loss_list, val_loss_list)