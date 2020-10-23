#----------------description----------------# 
# Author       : ,: Zihao Zhao
# E-mail       : ,: zhzhao18@fudan.edu.cn
# Company      : ,: Fudan University
# Date         : ,: 2020-10-23 14:12:06
# LastEditors  : Zihao Zhao
# LastEditTime : 2020-10-23 16:23:41
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