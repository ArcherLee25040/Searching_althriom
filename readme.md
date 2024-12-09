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
在上述代码中：
直接插入排序：从第二个元素开始，将当前元素与前面已排序好的元素依次比较，若小于前面的元素则将前面的元素往后移一位，直到找到合适位置插入当前元素，通过循环逐步将未排序的元素插入到已排序部分合适位置，最终实现整个数组有序。
折半插入排序：同样从第二个元素起，先通过折半查找（利用low、mid、high定位合适插入位置）在已排序部分确定当前元素要插入的位置，然后将插入位置之后的元素后移一位，插入当前元素，相较于直接插入排序，它减少了比较次数，提高了一定效率。
希尔排序：先按照一定间隔（gap）对数组进行分组，对每组进行插入排序，然后不断缩小间隔，重复这个过程，当间隔缩小到 1 时，整个数组基本接近有序，再进行一次普通插入排序即可，它通过这种分组预排序的方式，让元素能更快地移动到最终位置，提升排序效率。

---

- 选择排序
  - 简单选择排序
``` go
func SimpleSelectionSort(arr []int) []int {
    for i := 0; i < len(arr)-1; i++ {
        minIndex := i
        for j := i + 1; j < len(arr); j++ {
            if arr[j] < arr[minIndex] {
                minIndex = j
            }
        }
        if minIndex!= i {
            arr[i], arr[minIndex] = arr[minIndex], arr[i]
        }
    }
    return arr
}
```

  - 堆排序
``` go
func HeapSort(arr []int) []int {
    buildMaxHeap(arr)
    for i := len(arr) - 1; i > 0; i-- {
        arr[0], arr[i] = arr[i], arr[0]
        maxHeapify(arr, 0, i)
    }
    return arr
}

func buildMaxHeap(arr []int) {
    n := len(arr)
    for i := n/2 - 1; i >= 0; i-- {
        maxHeapify(arr, i, n)
    }
}

func maxHeapify(arr []int, i, n int) {
    left := 2*i + 1
    right := 2*i + 2
    largest := i
    if left < n && arr[left] > arr[largest] {
        largest = left
    }
    if right < n && arr[right] > arr[largest] {
        largest = right
    }
    if largest!= i {
        arr[i], arr[largest] = arr[largest], arr[i]
        maxHeapify(arr, largest, n)
    }
}
```

  - 锦标赛排序
``` go
type TreeNode struct {
    val   int
    left  *TreeNode
    right *TreeNode
    win   bool
}

func TournamentSort(arr []int) []int {
    if len(arr) == 0 {
        return []int{}
    }
    tree := buildTournamentTree(arr)
    sortedArr := make([]int, len(arr))
    for i := 0; i < len(arr); i++ {
        sortedArr[i] = extractWinner(tree).val
        updateTree(tree, sortedArr[i])
    }
    return sortedArr
}

func buildTournamentTree(arr []int) *TreeNode {
    if len(arr) == 1 {
        return &TreeNode{val: arr[0]}
    }
    mid := len(arr) / 2
    left := buildTournamentTree(arr[:mid])
    right := buildTournamentTree(arr[mid:])
    winner := left
    if right.val < left.val {
        winner = right
    }
    return &TreeNode{val: winner.val, left: left, right: right}
}

func extractWinner(tree *TreeNode) *TreeNode {
    if tree.left == nil && tree.right == nil {
        return tree
    }
    leftWinner := extractWinner(tree.left)
    rightWinner := extractWinner(tree.right)
    winner := leftWinner
    if rightWinner.val < leftWinner.val {
        winner = rightWinner
    }
    return winner
}

func updateTree(tree *TreeNode, val int) {
    if tree.val == val {
        tree.win = true
        return
    }
    if tree.left!= nil && tree.left.val == val {
        tree.left.win = true
        tree.val = minOf(tree.right.val, val)
        return
    }
    if tree.right!= nil && tree.right.val == val {
        tree.right.win = true
        tree.val = minOf(tree.left.val, val)
        return
    }
    if tree.left!= nil {
        updateTree(tree.left, val)
    }
    if tree.right!= nil {
        updateTree(tree.right, val)
    }
}

func minOf(a, b int) int {
    if a < b {
        return a
    }
    return b
}
```

``` go
//测试案例
func main() {
    // 简单选择排序测试
    arr1 := []int{5, 4, 3, 2, 1}
    sortedArr1 := SimpleSelectionSort(arr1)
    fmt.Println("简单选择排序结果:", sortedArr1)

    // 堆排序测试
    arr2 := []int{5, 4, 3, 2, 1}
    sortedArr2 := HeapSort(arr2)
    fmt.Println("堆排序结果:", sortedArr2)

    // 锦标赛排序测试
    arr3 := []int{5, 4, 3, 2, 1}
    sortedArr3 := TournamentSort(arr3)
    fmt.Println("锦标赛排序结果:", sortedArr3)
}
```
下面对这几种排序算法进行简单说明：
简单选择排序
它的基本思想是在未排序的序列中找到最小（或最大）元素，将其存放到序列的起始位置，然后再从剩余未排序元素中继续寻找最小（或最大）元素，放到已排序序列的末尾，以此类推，直到所有元素均排序完毕。每次遍历都要找出剩余元素中的最值，时间复杂度为 ，空间复杂度为 ，是一种简单但效率相对不高的排序算法。
堆排序
首先将待排序的数组构建成一个最大堆（大顶堆），此时堆顶元素就是最大值，将堆顶元素与数组末尾元素交换，然后对前面的 n - 1 个元素重新调整为最大堆，重复这个过程，直到整个数组有序。堆排序的时间复杂度平均、最好、最坏情况均为 ，空间复杂度为 ，是一种不稳定但效率相对较好的排序算法，常用于对排序稳定性要求不高且需要高效处理大量数据的场景。
锦标赛排序
也被称为树形选择排序，它通过构建二叉树（类似锦标赛的树形结构）来比较元素大小，每次选出胜者（较小值）往上传递，最终根节点就是最小值，取出根节点后更新树结构继续找下一个最小值，以此类推完成排序。它的时间复杂度也是 ，空间复杂度取决于构建树的结构，相对复杂一些，但在一些特定场景下（如外部排序等有合并操作的场景）可以有较好的应用，能较好地利用树形结构来减少比较次数、提高排序效率

---


- 交换排序
    - 冒泡排序
```go
package main

import "fmt"

// BubbleSort实现冒泡排序
func BubbleSort(arr []int) []int {
    n := len(arr)
    for i := 0; i < n-1; i++ {
        for j := 0; j < n-i-1; j++ {
            if arr[j] > arr[j+1] {
                arr[j], arr[j+1] = arr[j+1], arr[j]
            }
        }
    }
    return arr
}

func main() {
    arr := []int{5, 4, 3, 2, 1}
    sortedArr := BubbleSort(arr)
    fmt.Println(sortedArr)
}
```
   - 快速排序
```go
package main

import "fmt"

// QuickSort实现快速排序
func QuickSort(arr []int) []int {
    if len(arr) <= 1 {
        return arr
    }
    pivot := arr[0]
    var left, right []int
    for _, num := range arr[1:] {
        if num <= pivot {
            left = append(left, num)
        } else {
            right = append(right, num)
        }
    }
    return append(append(QuickSort(left), pivot), QuickSort(right)...)
}
func main() {
    arr := []int{5, 4, 3, 2, 1}
    sortedArr := QuickSort(arr)
    fmt.Println(sortedArr)
}
```

   - 快速选择排序
```go
package main

import "fmt"

// QuickSelect用于找到切片中第k小的元素
func QuickSelect(arr []int, k int) int {
    if len(arr) == 1 {
        return arr[0]
    }
    pivot := arr[0]
    left, right := []int{}, []int{}
    for _, num := range arr[1:] {
        if num <= pivot {
            left = append(left, num)
        } else {
            right = append(right, num)
        }
    }
    if k < len(left) {
        return QuickSelect(left, k)
    } else if k < len(arr)-len(right) {
        return pivot
    }
    return QuickSelect(right, k-(len(arr)-len(right)))
}
func main() {
    arr := []int{5, 4, 3, 2, 1}
    k := 2
    result := QuickSelect(arr, k-1)
    fmt.Printf("第 %d 小的元素是: %d\n", k, result)
}
```

- 归并排序
    - 二路归并排序
```go
package main

import "fmt"

// mergeThree用于合并三个已经有序的子数组
func mergeThree(a, b, c []int) []int {
    result := make([]int, 0, len(a)+len(b)+len(c))
    i, j, k := 0, 0, 0
    for i < len(a) && j < len(b) && k < len(c) {
        min := a[i]
        if b[j] < min {
            min = b[j]
        }
        if c[k] < min {
            min = c[k]
        }
        result = append(result, min)
        if min == a[i] {
            i++
        } else if min == b[j] {
            j++
        } else {
            k++
        }
    }
    for i < len(a) {
        result = append(result, a[i])
        i++
    }
    for j < len(b) {
        result = append(result, b[j])
        j++
    }
    for k < len(c) {
        result = append(result, c[k])
        k++
    }
    return result
}

// MultiWayMergeSort实现三路归并排序（这里是简单示意，可进一步抽象通用化）
func MultiWayMergeSort(arr []int) []int {
    n := len(arr)
    if n <= 1 {
        return arr
    }
    part1 := arr[:n/3]
    part2 := arr[n/3 : 2*n/3]
    part3 := arr[2*n/3:]
    sorted1 := MultiWayMergeSort(part1)
    sorted2 := MultiWayMergeSort(part2)
    sorted3 := MultiWayMergeSort(part3)
    return mergeThree(sorted1, sorted2, sorted3)
}
func main() {
    arr := []int{5, 4, 3, 2, 1}
    sortedArr := MergeSort(arr)
    fmt.Println(sortedArr)
}
```
   - 多路归并排序
```go
package main

import "fmt"

// mergeThree用于合并三个已经有序的子数组
func mergeThree(a, b, c []int) []int {
    result := make([]int, 0, len(a)+len(b)+len(c))
    i, j, k := 0, 0, 0
    for i < len(a) && j < len(b) && k < len(c) {
        min := a[i]
        if b[j] < min {
            min = b[j]
        }
        if c[k] < min {
            min = c[k]
        }
        result = append(result, min)
        if min == a[i] {
            i++
        } else if min == b[j] {
            j++
        } else {
            k++
        }
    }
    for i < len(a) {
        result = append(result, a[i])
        i++
    }
    for j < len(b) {
        result = append(result, b[j])
        j++
    }
    for k < len(c) {
        result = append(result, c[k])
        k++
    }
    return result
}

// MultiWayMergeSort实现三路归并排序（这里是简单示意，可进一步抽象通用化）
func MultiWayMergeSort(arr []int) []int {
    n := len(arr)
    if n <= 1 {
        return arr
    }
    part1 := arr[:n/3]
    part2 := arr[n/3 : 2*n/3]
    part3 := arr[2*n/3:]
    sorted1 := MultiWayMergeSort(part1)
    sorted2 := MultiWayMergeSort(part2)
    sorted3 := MultiWayMergeSort(part3)
    return mergeThree(sorted1, sorted2, sorted3)
}

func main() {
    arr := []int{5, 4, 3, 2, 1}
    sortedArr := MultiWayMergeSort(arr)
    fmt.Println(sortedArr)
}
```
- 基数排序
    - 最高位优先（MSD）基数排序
```go
package main

import (
    "fmt"
)

// 获取数组中的最大值
func getMax(arr []int) int {
    max := arr[0]
    for _, num := range arr {
        if num > max {
            max = num
        }
    }
    return max
}

// 计数排序，用于基数排序中的按位排序
func countingSortByDigit(arr []int, exp int) {
    n := len(arr)
    output := make([]int, n)
    count := make([]int, 10)

    // 统计每个桶中的元素数量
    for i := 0; i < n; i++ {
        index := (arr[i] / exp) % 10
        count[index]++
    }

    // 计算累计计数
    for i := 1; i < 10; i++ {
        count[i] += count[i-1]
    }

    // 根据计数将元素放入正确的位置
    for i := n - 1; i >= 0; i-- {
        index := (arr[i] / exp) % 10
        output[count[index]-1] = arr[i]
        count[index]--
    }

    // 将排序后的结果复制回原数组
    for i := 0; i < n; i++ {
        arr[i] = output[i]
    }
}

// MSD基数排序
func radixSortMSD(arr []int) {
    max := getMax(arr)

    // 从最高位开始，对每一位进行计数排序
    exp := 1
    for max/exp > 0 {
        countingSortByDigit(arr, exp)
        exp *= 10
    }
}

```
   - 最低位优先（LSD）基数排序
```go
package main

import (
    "fmt"
)

// 计数排序，用于基数排序中的按位排序
func countingSortByDigit(arr []int, exp int) {
    n := len(arr)
    output := make([]int, n)
    count := make([]int, 10)

    // 统计每个桶中的元素数量
    for i := 0; i < n; i++ {
        index := (arr[i] / exp) % 10
        count[index]++
    }

    // 计算累计计数
    for i := 1; i < 10; i++ {
        count[i] += count[i-1]
    }

    // 根据计数将元素放入正确的位置
    for i := n - 1; i >= 0; i-- {
        index := (arr[i] / exp) % 10
        output[count[index]-1] = arr[i]
        count[index]--
    }

    // 将排序后的结果复制回原数组
    for i := 0; i < n; i++ {
        arr[i] = output[i]
    }
}

// LSD基数排序
func radixSortLSD(arr []int) {
    max := getMax(arr)
    exp := 1

    // 从最低位开始，对每一位进行计数排序
    for max/exp > 0 {
        countingSortByDigit(arr, exp)
        exp *= 10
    }
}
```
- 计数排序
```go
package main

import (
    "fmt"
)

// 计数排序
func countingSort(arr []int) {
    n := len(arr)
    if n == 0 {
        return
    }

    // 找到数组中的最大值和最小值
    max := arr[0]
    min := arr[0]
    for _, num := range arr {
        if num > max {
            max = num
        }
        if num < min {
            min = num
        }
    }

    // 计算计数数组的长度
    rangeVal := max - min + 1
    count := make([]int, rangeVal)

    // 统计每个元素的出现次数
    for _, num := range arr {
        count[num-min]++
    }

    // 计算累计计数
    for i := 1; i < rangeVal; i++ {
        count[i] += count[i-1]
    }

    // 根据计数将元素放入正确的位置
    output := make([]int, n)
    for i := n - 1; i >= 0; i-- {
        index := arr[i] - min
        output[count[index]-1] = arr[i]
        count[index]--
    }

    // 将排序后的结果复制回原数组
    for i := 0; i < n; i++ {
        arr[i] = output[i]
    }
}
```

- 桶排序
```go
package main

import (
    "fmt"
)

// 桶排序
func bucketSort(arr []float64) {
    n := len(arr)
    if n == 0 {
        return
    }

    // 创建桶
    buckets := make([][]float64, n)
    for i := range buckets {
        buckets[i] = []float64{}
    }

    // 将元素分配到相应的桶中
    for _, num := range arr {
        index := int(num * float64(n))
        if index >= n {
            index = n - 1
        }
        buckets[index] = append(buckets[index], num)
    }

    // 对每个桶进行排序
    for _, bucket := range buckets {
        insertionSort(bucket)
    }

    // 将排序后的桶合并
    k := 0
    for _, bucket := range buckets {
        for _, num := range bucket {
            arr[k] = num
            k++
        }
    }
}

// 插入排序，用于桶内排序
func insertionSort(arr []float64) {
    n := len(arr)
    for i := 1; i < n; i++ {
        key := arr[i]
        j := i - 1
        for j >= 0 && arr[j] > key {
            arr[j+1] = arr[j]
            j--
        }
        arr[j+1] = key
    }
}
```
- 外部排序
    - 多路归并外部排序
```go
package main

import (
    "fmt"
)

// 多路归并
func multiwayMerge(runs [][]int) []int {
    n := len(runs)
    if n == 0 {
        return []int{}
    }
    if n == 1 {
        return runs[0]
    }

    result := []int{}
    indices := make([]int, n)

    // 初始化索引
    for i := range indices {
        indices[i] = 0
    }

    // 进行多路归并
    for {
        minVal := 0
        minRun := -1
        allFinished := true

        // 找到当前最小的元素及其所在的子数组
        for i := 0; i < n; i++ {
            if indices[i] < len(runs[i]) {
                allFinished = false
                if minRun == -1 || runs[i][indices[i]] < minVal {
                    minVal = runs[i][indices[i]]
                    minRun = i
                }
            }
        }

        // 如果所有子数组都已处理完，结束归并
        if allFinished {
            break
        }

        result = append(result, minVal)
        indices[minRun]++
    }

    return result
}

// 多路归并外部排序
func externalSortMultiway(arr []int, numRuns int) []int {
    n := len(arr)
    runSize := (n + numRuns - 1) / numRuns

    // 将数组分成多个子数组
    runs := make([][]int, numRuns)
    for i := 0; i < numRuns; i++ {
        start := i * runSize
        end := (i + 1) * runSize
        if end > n {
            end = n
        }
        runs[i] = make([]int, end-start)
        copy(runs[i], arr[start:end])
        insertionSort(runs[i])
    }

    return multiwayMerge(runs)
}

// 插入排序，用于子数组排序
func insertionSort(arr []int) {
    n := len(arr)
    for i := 1; i < n; i++ {
        key := arr[i]
        j := i - 1
        for j >= 0 && arr[j] > key {
            arr[j+1] = arr[j]
            j--
        }
        arr[j+1] = key
    }
}
```

   - 置换-选择排序
```go
package main

import (
    "fmt"
)

// 置换-选择排序
func replacementSelection(arr []int) []int {
    n := len(arr)
    result := []int{}
    minHeap := make([]int, 0)

    // 初始化最小堆
    for i := 0; i < n; i++ {
        minHeap = append(minHeap, arr[i])
        siftUp(minHeap, len(minHeap)-1)
    }

    // 进行置换-选择排序
    for len(minHeap) > 0 {
        minVal := minHeap[0]
        result = append(result, minVal)

        // 将下一个元素加入最小堆
        if i := i + 1; i < n {
            minHeap[0] = arr[i]
            siftDown(minHeap, 0)
        } else {
            minHeap = minHeap[1:]
        }
    }

    return result
}

// 上浮操作，用于维护最小堆的性质
func siftUp(heap []int, i int) {
    for ; i > 0 && heap[(i-1)/2] > heap[i]; i = (i-1)/2 {
        heap[(i-1)/2], heap[i] = heap[i], heap[(i-1)/2]
    }
}

// 下沉操作，用于维护最小堆的性质
func siftDown(heap []int, i int) {
    n := len(heap)
    for {
        minIndex := i
        l := 2*i + 1
        r := 2*i + 2
        if l < n && heap[l] < heap[minIndex] {
            minIndex = l
        }
        if r < n && heap[r] < heap[minIndex] {
            minIndex = r
        }
        if minIndex == i {
            break
        }
        heap[i], heap[minIndex] = heap[minIndex], heap[i]
        i = minIndex
    }
}
```
测试
```go
func main() {
    // 基数排序测试
    arr := []int{170, 45, 75, 90, 802, 24, 2, 66}
    radixSortMSD(arr)
    fmt.Println("MSD Radix Sort:", arr)

    arr = []int{170, 45, 75, 90, 802, 24, 2, 66}
    radixSortLSD(arr)
    fmt.Println("LSD Radix Sort:", arr)

    // 计数排序测试
    arr = []int{4, 2, 2, 8, 3, 3, 1}
    countingSort(arr)
    fmt.Println("Counting Sort:", arr)

    // 桶排序测试
    arr = []float64{0.42, 0.32, 0.33, 0.52, 0.37, 0.47, 0.51}
    bucketSort(arr)
    fmt.Println("Bucket Sort:", arr)

    // 外部排序测试
    arr = []int{9, 8, 7, 6, 5, 4, 3, 2, 1}
    sorted := externalSortMultiway(arr, 3)
    fmt.Println("Multiway Merge External Sort:", sorted)

    arr = []int{9, 8, 7, 6, 5, 4, 3, 2, 1}
    sorted = replacementSelection(arr)
    fmt.Println("Replacement Selection:", sorted)
}
```


## **搜索算法**
- 顺序搜索
```go
package main

import "fmt"

func sequentialSearch(arr []int, target int) int {
    for i := 0; i < len(arr); i++ {
        if arr[i] == target {
            return i
        }
    }
    return -1
}

func main() {
    arr := []int{5, 3, 8, 2, 9, 1, 7, 4, 6}
    target := 7
    result := sequentialSearch(arr, target)
    if result!= -1 {
        fmt.Printf("目标元素 %d 在数组中的索引为 %d\n", target, result)
    } else {
        fmt.Printf("未找到目标元素 %d\n", target)
    }
}

```

- 二分搜索
    - 普通二分搜索
```go
func binarySearch(arr []int, target int) int {
    left, right := 0, len(arr)-1
    for left <= right {
        mid := left + (right-left)/2
        if arr[mid] == target {
            return mid
        } else if arr[mid] > target {
            right = mid - 1
        } else {
            left = mid + 1
        }
    }
    return -1
}
```
   - 二分搜索变体（如查找第一个大于等于目标的元素等）
```go
func findFirstGreaterEqual(arr []int, target int) int {
    left, right := 0, len(arr)-1
    result := len(arr)
    for left <= right {
        mid := left + (right-left)/2
        if arr[mid] >= target {
            result = mid
            right = mid - 1
        } else {
            left = mid + 1
        }
    }
    return result
}
func findLastLessEqual(arr []int, target int) int {
    left, right := 0, len(arr)-1
    result := -1
    for left <= right {
        mid := left + (right-left)/2
        if arr[mid] <= target {
            result = mid
            left = mid + 1
        } else {
            right = mid - 1
        }
    }
    return result
}func searchInRotatedSortedArray(arr []int, target int) int {
    left, right := 0, len(arr)-1
    for left <= right {
        mid := left + (right-left)/2
        if arr[mid] == target {
            return mid
        }
        if arr[left] <= arr[mid] {
            if target >= arr[left] && target < arr[mid] {
                right = mid - 1
            } else {
                left = mid + 1
            }
        } else {
            if target > arr[mid] && target <= arr[right] {
                left = mid + 1
            } else {
                right = mid - 1
            }
        }
    }
    return -1
}
func main() {
    // 普通二分搜索测试
    arr := []int{1, 3, 5, 7, 9, 11, 13}
    target := 7
    result := binarySearch(arr, target)
    if result!= -1 {
        fmt.Printf("普通二分搜索：目标元素 %d 在数组中的索引为 %d\n", target, result)
    } else {
        fmt.Printf("普通二分搜索：未找到目标元素 %d\n", target)
    }

    // 查找第一个大于等于目标的元素测试
    arr = []int{1, 3, 5, 7, 9, 11, 13}
    target = 6
    result = findFirstGreaterEqual(arr, target)
    fmt.Printf("查找第一个大于等于目标的元素：索引为 %d\n", result)

    // 查找最后一个小于等于目标的元素测试
    arr = []int{1, 3, 5, 7, 9, 11, 13}
    target = 6
    result = findLastLessEqual(arr, target)
    fmt.Printf("查找最后一个小于等于目标的元素：索引为 %d\n", result)

    // 在旋转排序数组中查找目标元素测试
    arr = []int{4, 5, 6, 7, 0, 1, 2}
    target = 0
    result = searchInRotatedSortedArray(arr, target)
    if result!= -1 {
        fmt.Printf("在旋转排序数组中查找目标元素：目标元素 %d 在数组中的索引为 %d\n", target, result)
    } else {
        fmt.Printf("在旋转排序数组中查找目标元素：未找到目标元素 %d\n", target)
    }
}
```
- 哈希搜索
    - 开放寻址法
        - 线性探测
```go
const tableSize = 101

type HashTable struct {
    table    []int
    size     int
}

func NewHashTable() *HashTable {
    return &HashTable{
        table:    make([]int, tableSize),
        size:     0,
        // 初始化为特定值表示空位，这里用 -1 表示
        table:    make([]int, tableSize),
        size:     0,
    }
}

func (ht *HashTable) hashFunction(key int) int {
    return key % tableSize
}

func (ht *HashTable) linearProbeInsert(key int) {
    index := ht.hashFunction(key)
    for ht.table[index]!= -1 {
        index = (index + 1) % tableSize
    }
    ht.table[index] = key
    ht.size++
}

func (ht *HashTable) linearProbeSearch(key int) bool {
    index := ht.hashFunction(key)
    for ht.table[index]!= -1 {
        if ht.table[index] == key {
            return true
        }
        index = (index + 1) % tableSize
    }
    return false
}
```
   - 二次探测
```go
func (ht *HashTable) quadraticProbeInsert(key int) {
    index := ht.hashFunction(key)
    i := 0
    for ht.table[index]!= -1 {
        i++
        offset := i * i
        index = (index + offset) % tableSize
    }
    ht.table[index] = key
    ht.size++
}

func (ht *HashTable) quadraticProbeSearch(key int) bool {
    index := ht.hashFunction(key)
    i := 0
    for ht.table[index]!= -1 {
        if ht.table[index] == key {
            return true
        }
        i++
        offset := i * i
        index = (index + offset) % tableSize
    }
    return false
}
```
   - 双重哈希
```go
func (ht *HashTable) doubleHashFunction(key int) int {
    return 1 + (key % (tableSize - 2))
}

func (ht *HashTable) doubleHashInsert(key int) {
    index := ht.hashFunction(key)
    step := ht.doubleHashFunction(key)
    for ht.table[index]!= -1 {
        index = (index + step) % tableSize
    }
    ht.table[index] = key
    ht.size++
}

func (ht *HashTable) doubleHashSearch(key int) bool {
    index := ht.hashFunction(key)
    step := ht.doubleHashFunction(key)
    for ht.table[index]!= -1 {
        if ht.table[index] == key {
            return true
        }
        index = (index + step) % tableSize
    }
    return false
}
```
   - 拉链法
```go
type ListNode struct {
    key   int
    next  *ListNode
}

type HashTableWithChaining struct {
    table    []*ListNode
    size     int
}

func NewHashTableWithChaining() *HashTableWithChaining {
    return &HashTableWithChaining{
        table: make([]*ListNode, tableSize),
        size:  0,
    }
}

func (ht *HashTableWithChaining) hashFunction(key int) int {
    return key % tableSize
}

func (ht *HashTableWithChaining) insert(key int) {
    index := ht.hashFunction(key)
    node := &ListNode{key: key}
    if ht.table[index] == nil {
        ht.table[index] = node
    } else {
        current := ht.table[index]
        for current!= nil {
            if current.key == key {
                return
            }
            if current.next == nil {
                break
            }
            current = current.next
        }
        current.next = node
    }
    ht.size++
}

func (ht *HashTableWithChaining) search(key int) bool {
    index := ht.hashFunction(key)
    current := ht.table[index]
    for current!= nil {
        if current.key == key {
            return true
        }
        current = current.next
    }
    return false
}
func main() {
    // 开放寻址法 - 线性探测测试
    ht := NewHashTable()
    ht.linearProbeInsert(5)
    ht.linearProbeInsert(15)
    fmt.Println("线性探测搜索 5:", ht.linearProbeSearch(5))
    fmt.Println("线性探测搜索 10:", ht.linearProbeSearch(10))

    // 开放寻址法 - 二次探测测试
    ht2 := NewHashTable()
    ht2.quadraticProbeInsert(5)
    ht2.quadraticProbeInsert(15)
    fmt.Println("二次探测搜索 5:", ht2.quadraticProbeSearch(5))
    fmt.Println("二次探测搜索 10:", ht2.quadraticProbeSearch(10))

    // 开放寻址法 - 双重哈希测试
    ht3 := NewHashTable()
    ht3.doubleHashInsert(5)
    ht3.doubleHashInsert(15)
    fmt.Println("双重哈希搜索 5:", ht3.doubleHashSearch(5))
    fmt.Println("双重哈希搜索 10:", ht3.doubleHashSearch(10))

    // 拉链法测试
    ht4 := NewHashTableWithChaining()
    ht4.insert(5)
    ht4.insert(15)
    fmt.Println("拉链法搜索 5:", ht4.search(5))
    fmt.Println("拉链法搜索 10:", ht4.search(10))
}
```
- 跳表搜索
```go
import (
    "math/rand"
    "time"
)

// SkipListNode 代表跳表节点
type SkipListNode struct {
    key   int
    val   interface{}
    next  []*SkipListNode
}

// SkipList 代表跳表结构
type SkipList struct {
    head  *SkipListNode
    level int
}

// NewSkipList 创建一个新的跳表
func NewSkipList() *SkipList {
    return &SkipList{
        head: &SkipListNode{
            next: make([]*SkipListNode, 16),
        },
        level: 1,
    }
}

// randomLevel 随机生成节点的层数
func (sl *SkipList) randomLevel() int {
    level := 1
    for rand.Intn(2) == 1 {
        level++
    }
    return level
}

// Insert 往跳表中插入元素
func (sl *SkipList) Insert(key int, val interface{}) {
    update := make([]*SkipListNode, sl.level)
    cur := sl.head
    for i := sl.level - 1; i >= 0; i-- {
        for cur.next[i]!= nil && cur.next[i].key < key {
            cur = cur.next[i]
        }
        update[i] = cur
    }
    cur = cur.next[0]
    if cur!= nil && cur.key == key {
        cur.val = val
        return
    }
    newLevel := sl.randomLevel()
    if newLevel > sl.level {
        for i := sl.level; i < newLevel; i++ {
            update[i] = sl.head
        }
        sl.level = newLevel
    }
    newNode := &SkipListNode{
        key:   key,
        val:   val,
        next:  make([]*SkipListNode, newLevel),
    }
    for i := 0; i < newLevel; i++ {
        newNode.next[i] = update[i].next[i]
        update[i].next[i] = newNode
    }
}

// Search 在跳表中搜索元素
func (sl *SkipList) Search(key int) (interface{}, bool) {
    cur := sl.head
    for i := sl.level - 1; i >= 0; i-- {
        for cur.next[i]!= nil && cur.next[i].key < key {
            cur = cur.next[i]
        }
    }
    cur = cur.next[0]
    if cur!= nil && cur.key == key {
        return cur.val, true
    }
    return nil, false
}
func main() {
    sl := NewSkipList()
    sl.Insert(1, "value1")
    sl.Insert(3, "value3")
    val, found := sl.Search(1)
    if found {
        fmt.Println("跳表搜索到元素:", val)
    } else {
        fmt.Println("未找到元素")
    }
}
```
- 二叉搜索树搜索
    - 普通二叉搜索树
```go
// TreeNode 二叉树节点结构体
type TreeNode struct {
    val   int
    left  *TreeNode
    right *TreeNode
}

// BinarySearchTree 代表二叉搜索树
type BinarySearchTree struct {
    root *TreeNode
}

// NewBinarySearchTree 创建一个新的二叉搜索树
func NewBinarySearchTree() *BinarySearchTree {
    return &BinarySearchTree{}
}

// Insert 往二叉搜索树中插入节点
func (bst *BinarySearchTree) Insert(val int) {
    if bst.root == nil {
        bst.root = &TreeNode{val: val}
        return
    }
    cur := bst.root
    for {
        if val < cur.val {
            if cur.left == nil {
                cur.left = &TreeNode{val: val}
                return
            }
            cur = cur.left
        } else {
            if cur.right == nil {
                cur.right = &TreeNode{val: val}
                return
            }
            cur = cur.right
        }
    }
}

// Search 在二叉搜索树中搜索节点
func (bst *BinarySearchTree) Search(val int) bool {
    cur := bst.root
    for cur!= nil {
        if val == cur.val {
            return true
        } else if val < cur.val {
            cur = cur.left
        } else {
            cur = cur.right
        }
    }
    return false
}

```
  - AVL树搜索
```go
// AVLTreeNode AVL树节点结构体，相比普通二叉树节点多了高度信息
type AVLTreeNode struct {
    val   int
    left  *AVLTreeNode
    right *AVLTreeNode
    height int
}

// AVLTree 代表AVL树
type AVLTree struct {
    root *AVLTreeNode
}

// getHeight 获取节点的高度，空节点高度为 -1
func (node *AVLTreeNode) getHeight() int {
    if node == nil {
        return -1
    }
    return node.height
}

// updateHeight 更新节点的高度，根据左右子树高度来更新
func (node *AVLTreeNode) updateHeight() {
    node.height = max(node.getHeight(node.left), node.getHeight(node.right)) + 1
}

// Insert 往AVL树中插入节点（这里省略了完整的平衡调整代码）
func (avl *AVLTree) Insert(val int) {
    avl.root = avl.insertNode(avl.root, val)
}

func (avl *AVLTree) insertNode(node *AVLTreeNode, val int) *AVLTreeNode {
    if node == nil {
        return &AVLTreeNode{val: val, height: 0}
    }
    if val < node.val {
        node.left = avl.insertNode(node.left, val)
    } else {
        node.right = avl.insertNode(node.right, val)
    }
    node.updateHeight()
    return node
}

// Search 在AVL树中搜索节点
func (avl *AVLTree) Search(val int) bool {
    cur := avl.root
    for cur!= nil {
        if val == cur.val {
            return true
        } else if val < cur.val {
            cur = cur.left
        } else {
            cur = cur.right
        }
    }
    return false
}

func max(a, b int) int {
    if a > b {
        return a
    }
    return b
}

```
  - 红黑树搜索
```go
// Color 定义节点颜色
type Color bool

const (
    RED   Color = true
    BLACK Color = false
)

// RedBlackTreeNode 红黑树节点结构体，包含颜色信息
type RedBlackTreeNode struct {
    val   int
    left  *RedBlackTreeNode
    right *RedBlackTreeNode
    color Color
}

// RedBlackTree 代表红黑树
type RedBlackTree struct {
    root *RedBlackTreeNode
}

// Insert 往红黑树中插入节点（这里省略了完整的平衡调整代码）
func (rbt *RedBlackTree) Insert(val int) {
    rbt.root = rbt.insertNode(rbt.root, val)
    rbt.root.color = BLACK
}

func (rbt *RedBlackTree) insertNode(node *RedBlackTreeNode, val int) *RedBlackTreeNode {
    if node == nil {
        return &RedBlackTreeNode{val: val, color: RED}
    }
    if val < node.val {
        node.left = rbt.insertNode(node.left, val)
    } else {
        node.right = rbt.insertNode(node.right, val)
    }
    return node
}

// Search 在红黑树中搜索节点
func (rbt *RedBlackTree) Search(val int) bool {
    cur := rbt.root
    for cur!= nil {
        if val == cur.val {
            return true
        } else if val < cur.val {
            cur = cur.left
        } else {
            cur = cur.right
        }
    }
    return false
}

```
  - B树搜索
```go
// BTreeNode B树节点结构体
type BTreeNode struct {
    keys    []int
    children []*BTreeNode
    isLeaf  bool
}

// BTree B树结构体，指定阶数
type BTree struct {
    root   *BTreeNode
    degree int
}

// NewBTree 创建一个新的B树，指定阶数
func NewBTree(degree int) *BTree {
    return &BTree{
        root:   &BTreeNode{isLeaf: true},
        degree: degree,
    }
}

// Search 在B树中搜索元素
func (bt *BTree) Search(key int) bool {
    return bt.searchNode(bt.root, key)
}

func (bt *BTree) searchNode(node *BTreeNode, key int) bool {
    i := 0
    for i < len(node.keys) && key > node.keys[i] {
        i++
    }
    if i < len(node.keys) && key == node.keys[i] {
        return true
    }
    if node.isLeaf {
        return false
    }
    return bt.searchNode(node.children[i], key)
}

```
  - B+树搜索
```go
// BPlusTreeNode B+树节点结构体，和B树节点有一定区别
type BPlusTreeNode struct {
    keys    []int
    children []*BPlusTreeNode
    isLeaf  bool
    next    *BPlusTreeNode
}

// BPlusTree B+树结构体，指定阶数
type BPlusTree struct {
    root   *BPlusTreeNode
    degree int
}

// NewBPlusTree 创建一个新的B+树，指定阶数
func NewBPlusTree(degree int) *BPlusTree {
    return &BPlusTree{
        root:   &BPlusTreeNode{isLeaf: true},
        degree: degree,
    }
}

// Search 在B+树中搜索元素
func (bpt *BPlusTree) Search(key int) bool {
    return bpt.searchNode(bpt.root, key)
}

func (bpt *BPlusTree) searchNode(node *BPlusTreeNode, key int) bool {
    i := 0
    for i < len(node.keys) && key > node.keys[i] {
        i++
    }
    if node.isLeaf {
        return false
    }
    return bpt.searchNode(node.children[i], key)
}
```
  - B*树搜索
```go
// BStarTreeNode B*树节点结构体，结构类似B树节点但有不同的维护规则
type BStarTreeNode struct {
    keys    []int
    children []*BStarTreeNode
    isLeaf  bool
}

// BStarTree B*树结构体，指定阶数
type BStarTree struct {
    root   *BStarTreeNode
    degree int
}

// NewBStarTree 创建一个新的B*树，指定阶数
func NewBStarTree(degree int) *BStarTree {
        return &BStarTree{
                root:   &BStarTreeNode{isLeaf: true},
                degree: degree,
        }
}

// Search 在B*树中搜索元素
func (bst *BStarTree) Search(key int) bool {
        return bst.searchNode(bst.root, key)
}

func (bst *BStarTree) searchNode(node *BStarTreeNode, key int) bool {
        i := 0
        for i < len(node.keys) && key > node.keys[i] {
                i++
        }
        if i < len(node.keys) && key == node.keys[i] {
                return true
        }
        if node.isLeaf {
                return false
        }
        return bst.searchNode(node.children[i], key)
}
```
- 索引搜索
    - 倒排索引搜索
```go
package main

import (
    "fmt"
    "strings"
)

// InvertedIndex 倒排索引结构体，关键词对应文档编号列表
type InvertedIndex map[string][]int

// BuildInvertedIndex 根据文档内容构建倒排索引
func BuildInvertedIndex(documents []string) InvertedIndex {
    index := make(InvertedIndex)
    for docID, doc := range documents {
        words := strings.Fields(doc)
        for _, word := range words {
            index[word] = append(index[word], docID)
        }
    }
    return index
}

// SearchInvertedIndex 在倒排索引中搜索关键词对应的文档编号
func (index InvertedIndex) SearchInvertedIndex(keyword string) []int {
    return index[keyword]
}
func main() {
    documents := []string{
        "apple banana",
        "banana orange",
        "apple orange",
    }
    invertedIndex := BuildInvertedIndex(documents)

    keyword := "banana"
    result := invertedIndex.SearchInvertedIndex(keyword)
    fmt.Printf("关键词 %s 在文档编号: %v 中出现\n", keyword, result)
}
```
  - 数据库索引搜索
```go
package main

import (
    "database/sql"
    "fmt"

    _ "github.com/go-sql-driver/mysql"
)
func main() {
    // 数据库连接字符串，根据实际情况修改用户名、密码、数据库名等信息
    dataSourceName := "user:password@tcp(127.0.0.1:3306)/your_database?charset=utf8mb4"
    db, err := sql.Open("mysql", dataSourceName)
    if err!= nil {
        fmt.Printf("数据库连接失败: %v\n", err)
        return
    }
    defer db.Close()

    // 测试连接是否成功
    err = db.Ping()
    if err!= nil {
        fmt.Printf("无法连接到数据库: %v\n", err)
        return
    }

    // 假设在名为 employees 的表中，name 列有索引，执行基于索引的查询
    var name string
    var age int
    query := "SELECT name, age FROM employees WHERE name = '张三'"
    row := db.QueryRow(query)
    err = row.Scan(&name, &age)
    if err == sql.ErrNoRows {
        fmt.Println("未找到符合条件的数据")
    } else if err!= nil {
        fmt.Printf("查询执行出错: %v\n", err)
    } else {
        fmt.Printf("查询结果: 姓名 %s，年龄 %d\n", name, age)
    }
}
```

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
