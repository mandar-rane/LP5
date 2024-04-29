#include <iostream>
#include <omp.h>
#include <thread>

using namespace std;

int minVal(int arr[], int n){
    int minVal = arr[0];

    #pragma omp parallel for reduction(min : minVal)
    for(int i=0; i<n; i++){
        if(arr[i] < minVal){
            minVal = arr[i];
        }
    }

    return minVal;
}

int maxVal(int arr[], int n){
    int maxVal = arr[0];

    #pragma omp parallel for reduction(max: maxVal)
    for(int i=0; i<n; i++){
        if(arr[i] > maxVal){
            maxVal = arr[i];
        }
    }

    return maxVal;
}

int sum(int arr[], int n){
    int sum = 0;

    #pragma omp parallel for reduction(+:sum)
    for(int i=0; i<n; i++){
        sum += arr[i];
    }

    return sum;
}

int average(int arr[], int n){
    return (double)sum(arr, n)/n;
}

int main(){
    double starttime = omp_get_wtime();
    int n = 5;
    int arr[] = {1,2,3,4,5};

    double start_time = omp_get_wtime();
    cout << "Min: " << minVal(arr, n) << endl;
    cout << "Max: " << maxVal(arr, n) << endl;
    cout << "Sum: " << sum(arr, n) << endl;
    cout << "Average: " << average(arr, n) << endl;
    double end_time = omp_get_wtime();

    cout << endl;
    cout << "Time taken: " << end_time - start_time << " seconds" <<endl;
    
    return 0;
}