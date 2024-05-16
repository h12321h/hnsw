# hnsw

本项目实现了Hierarchical-Navigable-Small- World（HNSW）简单的相关数据结构及其算法，并在此基础上对其查询操作进⾏并⾏化优化。

## 编译

1. 进入在项目根目录下
2. 使用构建系统进行编译：
    ```shell
    make
    ```
3. 清理编译过程中生成的所有中间文件和目标文件
    ```shell
    make clean
    ```

## 运行测试及结果获取

### 正确性测试：
```shell
make grade
```

可以看到输出`average recall:` 表示召回率  

### 参数M的影响
修改`./util/parameter.hpp`中的`M_max` 和 `M`，执行
```shell
make grade
```

可以看到输出 `average recall:`表示召回率，`single query time`表示单次查询时延

### 性能测试
```shell
make test
```
可以看到输出`average insert time`表示插⼊时延

`Parallel:querying`下,`average recall:`表示并行召回率，`single query time`表示并行查询时延

`Serial:querying`下,`average recall:`表示串行召回率，`single query time`表示串行查询时延