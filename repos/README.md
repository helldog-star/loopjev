# 外部源码版本

完整第三方仓库不提交到LoopJev，只记录地址和固定版本。以下命令从项目根目录运行。
已有目录先检查，不覆盖本地文件；clone后使用detached HEAD固定版本。

## Open-Jev

仓库：https://github.com/Zefan-Cai/Open-Jev.git

固定提交：`bd4118882f733574a3250a4b65fe4d884130c08b`；记录时工作区干净，无需额外patch。

```bash
git clone https://github.com/Zefan-Cai/Open-Jev.git repos/Open-Jev
git -C repos/Open-Jev checkout --detach bd4118882f733574a3250a4b65fe4d884130c08b
```

## jevk5

仓库：https://github.com/allebee/jevk5.git

固定提交：`f26426d16f59e8bbe1470e5b162cc89329e29b29`；记录时工作区干净，无需额外patch。

```bash
git clone https://github.com/allebee/jevk5.git repos/jevk5
git -C repos/jevk5 checkout --detach f26426d16f59e8bbe1470e5b162cc89329e29b29
```

Open-Jev是Step4实际加载的官方源码；jevk5作为外部参考保留其版本，不代表已完成相应模型评测。
将来如需修改第三方源码，单独提交patch及应用说明，或记录fork地址和提交；不要把完整仓库复制进父仓库。
