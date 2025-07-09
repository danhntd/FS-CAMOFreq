import os
import math
import argparse
import numpy as np
from tabulate import tabulate

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--res-dir', type=str, default='', help='Path to the results')
    parser.add_argument('--shot-list', type=int, nargs='+', default=[1,2,3,5], help='')
    args = parser.parse_args()

    wf = open(os.path.join(args.res_dir, 'results.txt'), 'w')
    sum_res_seeds = []
    for shot in args.shot_list:

        file_paths = []
        for fid, fname in enumerate(os.listdir(args.res_dir)):
            if '{}shot'.format(shot) not in fname:
                continue
            _dir = os.path.join(args.res_dir, fname)
            file_paths.append(os.path.join(_dir, 'log.txt'))

        header, results = [], []
        header = ["segAP", "AP50", "AP75", "APs", "APm", "APl", "AR1", "AR10", "AR100", "ARs", "ARm", "ARl", "boxAP", "AP50", "AP75", "APs", "APm", "APl", "APl", "AR1", "AR10", "AR100", "ARs", "ARm", "ARl"]
        for fid, fpath in enumerate(sorted(file_paths)):
            # print(file_paths)
            lineinfos = open(fpath).readlines()

            sum_res = []
            for idx, line in reversed(list(enumerate(lineinfos))):
                if "fsdet.evaluation.camo_evaluation INFO: [" in line and len(sum_res) < 2: # sum_res = [seg, det]
                  tmp = line.split("INFO: [")[1][:-2] + " " + lineinfos[idx+1][:-2] + " "
                  sum_res.append(tmp)
            sum_res = [sum_res[0] + sum_res[1]]
            sum_res = [float(i) for i in sum_res[0].split()]

            results.append([fid] + sum_res)
            print(results)

        results_np = np.array(results)
        avg = np.mean(results_np, axis=0).tolist()
        cid = [1.96 * s / math.sqrt(results_np.shape[0]) for s in np.std(results_np, axis=0)]
        #results.append(['μ'] + avg[1:])
        #results.append(['c'] + cid[1:])

        table = tabulate(
            results,
            tablefmt="pipe",
            floatfmt=".2f",
            headers=[''] + header,
            numalign="left",
        )

        wf.write('--> {}-shot\n'.format(shot))
        wf.write('{}\n\n'.format(table))
        wf.flush()
    wf.close()

    print('Reformat all results -> {}'.format(os.path.join(args.res_dir, 'results.txt')))


if __name__ == '__main__':
    main()