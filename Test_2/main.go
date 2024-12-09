// 随机地雷
package main

import (
	"fmt"
	"math/rand"
	"time"
)

func generateMineMap(n int, mineCount int) [][]int {
	//初始化矩阵
	mineMap := make([][]int, n)
	for i := range mineMap {
		mineMap[i] = make([]int, n)
	}

	//随机设置种子
	rand.Seed(time.Now().UnixNano())

	//随机设置种子a
	count := 0
	for count < mineCount {
		x := rand.Intn(n)
		y := rand.Intn(n)
		if mineMap[x][y] == 0 {
			mineMap[x][y] = 1
			count++
		}
	}
	return mineMap
}

func main() {
	n := 5
	mineCount := 3
	mineMap := generateMineMap(n, mineCount)
	for _, row := range mineMap {
		for _, cell := range row {
			fmt.Printf("%d ", cell)
		}
		fmt.Println()
	}
}
