#!/usr/bin/env python3
"""LoopJev 按阶段运行入口；统一模块搜索路径，保留实验源码指纹。"""
import os
from pathlib import Path
import runpy
import sys

SCRIPTS = Path(__file__).resolve().parent
ENTRIES = {
    'step5': {'offline': 'step5/offline_exit.py', 'test': 'step5/test_offline_exit.py', 'analyze': 'step5/analyze_exit.py', 'test-analysis': 'step5/test_analyze_exit.py', 'joint': 'step5/joint_exit.py', 'test-joint': 'step5/test_joint_exit.py', 'risk-data': 'step5/risk_data.py', 'risk-fit': 'step5/risk_fit.py', 'risk-all': 'step5/run_risk.sh', 'test-risk': 'step5/test_risk.py'},
    'setup': {'env': 'setup/setup_env.sh', 'download': 'setup/download_assets.py'},
    'step0': {'smoke': 'step0/step0_smoke.py'},
    'step1': {'extract': 'step1/step1_extract.py', 'probe': 'step1/step1_probe.py', 'finish': 'step1/step1_finish.py'},
    'step2': {'train': 'step2/step2_train.py', 'compare': 'step2/step2_compare.py'},
    'step4': {'all': 'step4/run_all.sh', 'external-eval': 'step4/openjev_eval.py', 'external-compare': 'step4/openjev_compare.py', 'external-env': 'step4/setup_openjev_env.sh', 'train': 'step4/dense_train.py', 'compare': 'step4/dense_compare.py', 'matrix': 'step4/run_dense.sh', 'test-dense': 'step4/test_dense.py', 'test-external': 'step4/test_openjev.py'},
    'step3': {'train': 'step3/step3_train.py', 'compare': 'step3/step3_compare.py', 'matrix': 'step3/run_step3.sh'},
}


def main():
    args = sys.argv[1:]
    if not args or args[0] in ('--help', '-h', '--list'):
        print('用法: python scripts/run.py <阶段> <命令> [原脚本参数...]')
        for step, commands in ENTRIES.items():
            print(f'  {step}: {", ".join(commands)}')
        print('例: python scripts/run.py step3 train --help')
        return
    if len(args) < 2 or args[0] not in ENTRIES or args[1] not in ENTRIES[args[0]]:
        raise SystemExit('未知阶段或命令；使用 --list 查看可用入口。')
    # All executable entries use their canonical stage directories.
    entry = SCRIPTS / ENTRIES[args[0]][args[1]]
    if not entry.is_file():
        raise SystemExit(f'入口缺失: {entry}')
    os.chdir(SCRIPTS.parent)
    if entry.suffix == '.sh':
        os.execv('/bin/bash', ['/bin/bash', str(entry), *args[2:]])
    sys.path[:0] = [str(SCRIPTS / name) for name in ['common', 'step0', 'step1', 'step2', 'step3', 'step4', 'step5', 'setup']]
    from runtime_paths import configure
    configure()
    sys.argv = [str(entry), *args[2:]]
    runpy.run_path(str(entry), run_name='__main__')


if __name__ == '__main__':
    main()
