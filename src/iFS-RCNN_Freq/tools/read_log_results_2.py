import os
import math
import argparse
import numpy as np
from tabulate import tabulate
import re

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--res-dir', type=str, default='', help='Path to the results')
    parser.add_argument('--shot-list', type=int, nargs='+', default=[1, 2, 3, 5], help='List of shots')
    args = parser.parse_args()

    wf = open(os.path.join(args.res_dir, 'results.txt'), 'w')
    for shot in args.shot_list:
        file_paths = []
        for fname in os.listdir(args.res_dir):
            if '{}shot'.format(shot) not in fname:
                continue
            _dir = os.path.join(args.res_dir, fname)
            file_paths.append(os.path.join(_dir, 'log.txt'))

        header = ["boxAP", "AP50", "AP75", "APs", "APm", "APl", "AR1", "AR10", "AR100", "ARs", "ARm", "ARl",
                  "segAP", "AP50", "AP75", "APs", "APm", "APl", "AR1", "AR10", "AR100", "ARs", "ARm", "ARl"]
        results_by_iter = {}  # Dictionary to store results per iteration

        for fid, fpath in enumerate(sorted(file_paths)):
            with open(fpath, 'r') as f:
                lines = f.readlines()

                iter_num = None
                inference_start = False
                sum_res = []
                current_iter_results = []

                for idx, line in enumerate(lines):
                    # Detect iteration number from training logs
                    iter_match = re.search(r'iter: (\d+)', line)
                    if iter_match:
                        iter_num = int(iter_match.group(1))

                    # Detect start of inference
                    if "fsdet.evaluation.evaluator INFO: Start inference on 2655 images" in line:
                        inference_start = True
                        sum_res = []
                        continue

                    # Collect evaluation results after inference starts
                    if inference_start and "fsdet.evaluation.camo_evaluation INFO: [" in line and len(sum_res) < 2:
                        tmp = line.split("INFO: [")[1][:-2] + " " + lines[idx + 1][:-2] + " "
                        sum_res.append(tmp)

                    # End of inference block, process results
                    if inference_start and "fsdet.evaluation.testing INFO: copypaste: Task: segm" in line:
                        if sum_res:
                            # Combine segm and bbox results
                            combined_res = [sum_res[0] + sum_res[1]]
                            combined_res = [float(i) for i in combined_res[0].split()]
                            if iter_num is not None:
                                if iter_num not in results_by_iter:
                                    results_by_iter[iter_num] = []
                                results_by_iter[iter_num].append([fid] + combined_res)
                        inference_start = False
                        sum_res = []

                # Process any remaining results
                if sum_res and iter_num is not None:
                    combined_res = [sum_res[0] + sum_res[1]] if len(sum_res) == 2 else [sum_res[0]]
                    combined_res = [float(i) for i in combined_res[0].split()]
                    if iter_num not in results_by_iter:
                        results_by_iter[iter_num] = []
                    results_by_iter[iter_num].append([fid] + combined_res)

        # Generate tables for each iteration
        for iter_num in sorted(results_by_iter.keys()):
            results = results_by_iter[iter_num]
            if not results:
                continue

            results_np = np.array(results)
            avg = np.mean(results_np, axis=0).tolist()
            cid = [1.96 * s / math.sqrt(results_np.shape[0]) for s in np.std(results_np, axis=0)]

            table = tabulate(
                results,
                tablefmt="pipe",
                floatfmt=".2f",
                headers=[''] + header,
                numalign="left",
            )

            wf.write(f'--> {shot}-shot, Iteration {iter_num}\n')
            wf.write(f'{table}\n\n')
            wf.flush()

    wf.close()
    print(f'Reformat all results -> {os.path.join(args.res_dir, "results.txt")}')

if __name__ == '__main__':
    main()