# Open-Jev 独立环境重建

完整环境目录被Git忽略；本目录只提交说明和依赖快照。不要在现有Ouro/loopjev环境里升级依赖。

## 当前环境快照

- Python：3.11.17；平台：Linux-5.10.0-216.0.0.115.oe2203sp4.x86_64-x86_64-with-glibc2.35。
- PyTorch：2.14.1+cu130；其CUDA运行时：13.0。
- Transformers：5.10.2；PEFT：0.19.1；Accelerate：1.13.0。

这是实际安装环境，不是官方最低依赖示例。驱动、GPU和系统库不由pip freeze保存；
这些快照不保证跨平台直接安装成功，重建后需重新验证导入和真实权重冒烟。

## 方式一：Conda YAML

从项目根目录运行，使用未存在的新目录，不覆盖当前环境：

```bash
conda env create --prefix "$PWD/.envs/openjev-rebuild" --file .envs/openjev-environment.yml
conda activate "$PWD/.envs/openjev-rebuild"
```

YAML使用--no-builds导出，已移除机器专属prefix，保留真实Conda渠道与Conda/Pip依赖。
该文件是Linux环境快照；无build号意味着不保证得到完全相同的二进制构建。

## 方式二：Python与pip清单

```bash
conda create --prefix "$PWD/.envs/openjev-rebuild" python=3.11.17 pip
conda activate "$PWD/.envs/openjev-rebuild"
python -m pip install -r configs/openjev-requirements.txt
```

requirements记录安装版本，不记录包索引、wheel来源及系统依赖。如果默认包源找不到相同CUDA构建，
需选择对应官方PyTorch wheel源并保持版本；不要静默换成CPU版或另一个CUDA版本。
若必须换版本，应记录新实验环境，并重新做兼容性验证。
本地editable Open-Jev已改为固定提交的Git依赖；packaging的Conda构建路径已改为实际版本声明。
两种方式二选一，不必先导入完整YAML后再重复安装requirements。

## 官方源码与验证

先按 [repos/README.md](../repos/README.md) 恢复固定提交的完整源码。
本项目评测从repos/Open-Jev直接导入，无需额外editable安装；清单中的Git依赖若存在则固定commit。

```bash
python -c "import torch,transformers,peft,accelerate; print(torch.__version__, torch.version.cuda, transformers.__version__, peft.__version__, accelerate.__version__)"
python scripts/run.py step4 test-external
python scripts/run.py step4 external-eval --audit-only --out-dir work/openjev_rebuild_audit
```

审计不会运行GPU推理；真实权重冒烟需等待GPU空闲。具体命令见 [Step4 README](../scripts/step4/README.md)。
当前流水线使用 `.envs/openjev/bin/python`；重建验证完成前不要替换这个路径。

## 更新原则

依赖变更后重新导出两份清单，检查本机路径、私有URL和凭据；不提交完整环境或pip缓存。
修改环境前检查运行任务，不在任务运行中安装或升级包。
