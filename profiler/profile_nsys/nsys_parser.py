#!/usr/bin/env python
# encoding: utf-8

import os
import re
import csv
import sys
import json
import traceback
import argparse

from prettytable import PrettyTable


def select_table_data(all_data, table_name):
    value_list = []

    start_idx = 0
    for i in range(len(all_data)):
        if table_name in all_data[i]:
            start_idx = i + 2
            break
    key_list = [key.strip() for key in all_data[start_idx].split(',')]
    if len(key_list) == 0:
        return [], []

    end_idx = start_idx = start_idx + 1
    for line in all_data[start_idx:]:
        if len(line.strip()) == 0:
            break
        end_idx += 1
    csv_content = csv.reader(all_data[start_idx:end_idx])

    for line in csv_content:
        kern_dict = {k.strip(): v.strip() for k, v in zip(key_list, line)}
        value_list.append(kern_dict)
    return key_list, value_list


def common_summary(summary, new_data, key_list):
    summary[key_list[0]] = summary.get(key_list[0], 0.0) + float(
        new_data[key_list[0]])
    summary[key_list[1]] = summary.get(key_list[1], 0) + int(
        new_data[key_list[1]])
    summary[key_list[2]] = summary.get(key_list[2], 0) + int(
        new_data[key_list[2]])
    summary[key_list[3]] = summary[key_list[1]] / summary[key_list[2]]
    summary[key_list[4]] = min(summary.get(key_list[4], sys.maxsize),
                               int(new_data[key_list[4]]))
    summary[key_list[5]] = max(summary.get(key_list[5], 0),
                               int(new_data[key_list[5]]))


def common_query(table_data, key_list, field_idx, re_str, category):
    summary = {}
    detail = []
    field = key_list[field_idx]
    pattern = re.compile(re_str)
    for line in table_data:
        res = pattern.search(line[field])
        if res is not None:
            common_summary(summary, line, key_list)
            detail.append({key: line[key] for key in key_list})
    if len(detail) > 0:
        summary['Category'] = category
    return summary, detail


def concate_table(data_list, summary, detail):
    if len(detail) > 0:
        data_list.append((summary, detail))


def common_postprocess(row, indent=True, pretty=True):
    MAX_COLUME_WIDTH = 100

    row_data = list(row.values())
    row_data[0] = round(float(row_data[0]), 1)
    row_data[1] = round(float(row_data[1]) / 1000, 1)
    row_data[2] = int(row_data[2])
    row_data[3] = round(float(row_data[3]) / 1000, 1)
    row_data[4] = round(float(row_data[4]) / 1000, 1)
    row_data[5] = round(float(row_data[5]) / 1000, 1)
    if row_data[6].isnumeric():
        row_data[6] = round(float(row_data[5]) / 1000, 1)
        
    if pretty:
        if len(row_data[-1]) > MAX_COLUME_WIDTH:
            row_data[-1] = row_data[-1][:MAX_COLUME_WIDTH - 3] + "..."

    if indent:
        row_data = [" " * 2 + str(col_data) for col_data in row_data]

    return row_data


def summary_postprocess(summary_dict):
    summary_list = list(summary_dict.values())
    summary_list[0] = round(float(summary_list[0]), 1)
    summary_list[1] = round(float(summary_list[1]) / 1000, 1)
    summary_list[3] = round(float(summary_list[3]) / 1000, 1)
    summary_list[4] = round(float(summary_list[4]) / 1000, 1)
    summary_list[5] = round(float(summary_list[5]) / 1000, 1)
    return summary_list


def show_category_table(table_header, data_list):
    if len(data_list) == 0:
        return
    table_list = data_list

    table_frame = PrettyTable(table_header)
    for (summary, detail) in table_list:
        table_frame.add_row(summary_postprocess(summary))
        for row in detail:
            table_frame.add_row(common_postprocess(row))

    table_frame.align = "l"
    print(table_frame)


def show_table(table_header, table_data):
    table_frame = PrettyTable(table_header)
    for row in table_data:
        table_frame.add_row(common_postprocess(row))
    table_frame.align = "l"
    print(table_frame)

def reverse_query(src_query_result, table_data, key_list, key_idx, category):
    summary = {}

    select_set = set()
    key_field = key_list[key_idx]
    for res in src_query_result:
        select_set.update([item[key_field] for item in res])

    remain_list = []
    for row in table_data:
        if row[key_field] not in select_set:
            common_summary(summary, row, key_list)
            remain_list.append({key: row[key] for key in key_list if row})

    if len(remain_list) > 0:
        summary['Category'] = category

    return summary, remain_list

def add_display_data(tips, display_column_name, common_postprocess):
    display_meta = {
        "col_name": display_column_name,
        "postprocess": common_postprocess
    }
    new_tips = []
    for tip in tips:
        tip.update(display_meta)
        new_tips.append(tip)
    return new_tips

def get_model_precision(kernel_list, keys):
    fp16_re_str = r"_hgemm_|_h884gemm_|_h1688gemm_|gemm_fp16_|_hcudnn_|_h884cudnn_|_h1688cudnn_"
    fp16_pattern = re.compile(fp16_re_str)
    for kern in kernel_list:
        res = fp16_pattern.search(kern[keys[-1]])
        if res is not None:
            print("Model Precision: FP16 - \"{}\"".format(
                list(kern.values())[-1]))
            return "FP16"
    print("Model Precision: FP32")
    return "FP32"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("nsys_stdout", type=str, help="nsight system csv log file")
    parser.add_argument("--save_path", type=str, help="path to save the displayed data")
    parser.add_argument("--ops", type=str, help="OPs to profile")
    args = parser.parse_args()

    nsys_stdout = []
    with open(args.nsys_stdout) as fr:
        nsys_stdout = fr.readlines()

    gpukey, gpusum = select_table_data(nsys_stdout, "gpusum")
    cudaapikey, cudaapisum = select_table_data(nsys_stdout, "cudaapisum")
    """
    gpumemsizekey, gpumemsizesum = select_table_data(nsys_stdout, "gpumemsizesum")
    print(gpumemsizekey, "\n", gpumemsizesum)
    """

    pick_col = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    pick_col_name = [gpukey[idx] for idx in pick_col if gpukey]
    tc_gemm_re = r"(_(s|h)(1688|884|161616)gemm_|_[hib]mma_.+traits)"
    cc_gemm_re = r"(_sgemm_|_hgemm_|_gcgemm_|_cgemm_|void gem.+<)"
    tc_conv_re = r"_(s|h)(1688|884)cudnn"
    #conv_kernel_list = ["nchwToNhwcKernel<", "nhwcToNchwKernel<", "flip_filter", "im2col4d_kernel", "fermiPlusCgemmLDS128_batched", "transpose_readWrite_alignment_kernel", "compute_gemm_pointers"]
    cc_conv_re = r"(_(s|h)cudnn_|convolve_.+[<(]|fft2d_[rc]2[rc]_|conv2d_|depthwiseConvFP32Kernel|DepthwiseConv2d)"
    mem_re = r"CUDA mem"

    gemm_tensorcore, gemm_tensorcore_detail = common_query(
        gpusum, pick_col_name, -1, tc_gemm_re, "MatMul on Tensor Core")
    gemm_cudacore, gemm_cudacore_detail = common_query(
        gpusum, pick_col_name, -1, cc_gemm_re, "MatMul on CUDA Core")
    conv_tensorcore, conv_tensorcore_detail = common_query(
        gpusum, pick_col_name, -1, tc_conv_re, "Convolution on Tensor Core")
    conv_cudacore, conv_cudacore_detail = common_query(
        gpusum, pick_col_name, -1, cc_conv_re, "Convolution on CUDA Core")
    mem_op, mem_op_detail = common_query(
        gpusum, pick_col_name, -1, mem_re, "Memory Operation")

    display_column_name = [
        "Time (%)", "Total Time (us)", "Num Calls", "Average(us)",
        "Minimum(us)", "Maximum(us)", "StdDev(us)", "Category", "Name"
    ]
    # show_table(display_column_name, gpusum)
    # print(gemm_cudacore_detail)
    if args.ops == "MatMul":
        rst_list = gemm_tensorcore_detail + gemm_cudacore_detail
    elif args.ops == "Conv2D":
        rst_list = conv_tensorcore_detail + conv_cudacore_detail
    else:
        raise ValueError(args.ops)
    if args.save_path is not None:
        with open(args.save_path, 'a') as fp:
            fp.write(str(rst_list)+"\n")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Error: Catch Exception: {}".format(e))
        traceback.print_exc()
