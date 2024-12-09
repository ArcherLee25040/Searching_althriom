package main

import "fmt"

//快排
func qucikSort(arr []int) []int {
	if len(arr) <= 1 {
		return arr
	}

	pivot := arr[0]
	var left, right []int
	for _, v := range arr[1:] {
		if v <= pivot {
			left = append(left, v)
		} else {
			right = append(right, v)
		}
	}
	return append(append(qucikSort(left), pivot), qucikSort(qucikSort(right))...)

}

func main() {
	arr := []int{1, 6, 3, 90, 5, 2, 78, 33}
	result := qucikSort(arr)
	fmt.Println(result)
	flieselletr()

}

func flieselletr() {
	fmt.Println("hello")
}
