# 常见算法
## **排序算法**
- 插入排序
    - 直接插入排序
``` go
func InsertionSort(arr []int) []int {
    for i := 1; i < len(arr); i++ {
        key := arr[i]
        j := i - 1
        for ; j >= 0 && arr[j] > key; j-- {
            arr[j+1] = arr[j]
        }
        arr[j+1] = key
    }
    return arr
}
```
    
  - 折半插入排序


``` go
func BinaryInsertionSort(arr []int) []int {
    for i := 1; i < len(arr); i++ {
        key := arr[i]
        low, high := 0, i-1
        for low <= high {
            mid := low + (high-low)/2
            if arr[mid] > key {
                high = mid - 1
            } else {
                low = mid + 1
            }
        }
        for j := i - 1; j >= low; j-- {
            arr[j+1] = arr[j]
        }
        if low < i {
            arr[low] = key
        }
    }
    return arr
}
```
  - 希尔排序
``` go
func ShellSort(arr []int) []int {
    n := len(arr)
    gap := n / 2
    for gap > 0 {
        for i := gap; i < n; i++ {
            temp := arr[i]
            j := i
            for ; j >= gap && arr[j-gap] > temp; j -= gap {
                arr[j] = arr[j-gap]
            }
            arr[j] = temp
        }
        gap /= 2
    }
    return arr
}
```
``` go
func main() {
    // 示例用法
    arr := []int{5, 4, 3, 2, 1}
    sortedArr := InsertionSort(arr)
    fmt.Println("直接插入排序结果:", sortedArr)

    arr2 := []int{5, 4, 3, 2, 1}
    sortedArr2 := BinaryInsertionSort(arr2)
    fmt.Println("折半插入排序结果:", sortedArr2)

    arr3 := []int{5, 4, 3, 2, 1}
    sortedArr3 := ShellSort(arr3)
    fmt.Println("希尔排序结果:", sortedArr3)
}
//单元测试
```
- 选择排序
    - 简单选择排序
    - 堆排序
    - 锦标赛排序
- 交换排序
    - 冒泡排序
    - 快速排序
    - 快速选择排序
- 归并排序
    - 二路归并排序
    - 多路归并排序
- 基数排序
    - 最高位优先（MSD）基数排序
    - 最低位优先（LSD）基数排序
- 计数排序
- 桶排序
- 外部排序
    - 多路归并外部排序
    - 置换-选择排序

## **搜索算法**
- 顺序搜索
- 二分搜索
    - 普通二分搜索
    - 二分搜索变体（如查找第一个大于等于目标的元素等）
- 哈希搜索
    - 开放寻址法
        - 线性探测
        - 二次探测
        - 双重哈希
    - 拉链法
- 跳表搜索
- 二叉搜索树搜索
    - 普通二叉搜索树
    - AVL树搜索
    - 红黑树搜索
    - B树搜索
        - B+树搜索
        - B*树搜索
- 索引搜索
    - 倒排索引搜索
    - 数据库索引搜索

## **图算法**
- 最小生成树算法
    - Prim算法
    - Kruskal算法
    - Boruvka算法
- 最短路径算法
    - Dijkstra算法
    - Bellman-Ford算法
    - Floyd-Warshall算法
    - A*算法
    - SPFA算法
- 拓扑排序
- 图的遍历
    - 深度优先搜索（DFS）
        - 递归实现
        - 非递归实现（栈）
    - 广度优先搜索（BFS）
        - 队列实现
- 强连通分量算法
    - Tarjan算法
    - Kosaraju算法
- 割点与桥算法
    - Tarjan算法求割点
    - 求桥算法
- 二分图匹配算法
    - 匈牙利算法
    - 最大流最小割定理相关算法（如Ford-Fulkerson算法等）

## **字符串匹配算法**
- 暴力匹配算法
- KMP算法
- BM算法
- Sunday算法
- Rabin-Karp算法

## **动态规划算法**
- 斐波那契数列
- 最长公共子序列（LCS）
- 最长递增子序列（LIS）
- 背包问题
    - 0-1背包问题
    - 完全背包问题
    - 多重背包问题
- 矩阵连乘问题
- 编辑距离问题

## **贪心算法**
- 活动安排问题
- 哈夫曼编码
- 最小生成树（Prim和Kruskal算法本质也是贪心）
- 找零问题

## **分治算法**
- 归并排序
- 快速排序
- 二分搜索
- 大整数乘法
- 棋盘覆盖问题

## **回溯算法**
- 八皇后问题
- 全排列问题
- 数独求解
- 组合总和问题

## **分支限界算法**
- 旅行商问题（TSP）
- 0-1背包问题（优化解法）
- 任务分配问题

---
# 算法导论

以下是《算法导论》这本书的详细目录：

### 基础知识
- 算法在计算中的作用
- 算法基础
- 函数的增长
- 递归式
- 概率分析和随机算法

### 排序和顺序统计量
- 排序算法
- 堆排序
- 快速排序
- 线性时间排序
- 顺序统计量

### 数据结构
- 基本数据结构
- 散列表
- 二叉查找树
- 红黑树
- 动态顺序统计
- 区间树
- 外部排序
- 线性规划

### 高级设计和分析技术
- 动态规划
- 贪心算法
- 平摊分析

### 高级数据结构
- 斐波那契堆
- 用于不相交集合的数据结构
- 图的基本算法
- 最小生成树
- 单源最短路径
- 全对最短路径
- 最大流

### 图算法
- 图算法简介
- 最小生成树算法
- 单源最短路径算法
- 全对最短路径算法
- 最大流算法
- 图的匹配问题
- 图的着色问题

### 算法问题选编
- 动态等价关系
- 字符串匹配
- 计算几何
- 数论算法
- 多项式与快速傅里叶变换
- 线性规划
- 最大流
- 图的匹配
- 计算复杂性

### 附录
- 求和公式
- 概率分布
- 矩阵运算
- 线性规划基础

 