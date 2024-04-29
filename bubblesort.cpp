#include <iostream>
#include <omp.h>


using namespace std;

void parallelBubbleSort(int arr[], int n){
    int i, j;
    #pragma omp parallel for shared(arr, n) private(i, j)
    for(i = 0; i<n-1; i++){
        for(j = 0; j<n-i-1; j++){
            if(arr[j] > arr[j+1]){
                int temp = arr[j];
                arr[j] = arr[j+1];
                arr[j+1] = temp;
            }
        }
    }

}

void sequentialBubbleSort(int arr[], int n){
    int i, j;
    
    for(i = 0; i<n-1; i++){
        for(j = 0; j<n-i-1; j++){
            if(arr[j] > arr[j+1]){
                int temp = arr[j];
                arr[j] = arr[j+1];
                arr[j+1] = temp;
            }
        }
    }

}

void printArray(int arr[], int size){
    for(int i=0; i<size; i++){
        cout<<arr[i]<<" ";
    }
    cout<<endl;
}

int main(){
    int n = 100;
    int arr[n];

    for (int i = 0; i < n; i++) {
        arr[i] = rand() % 100;
    }

    cout<<"Unsorted Array: ";
    printArray(arr, n);

    cout<<endl;

    double start_time = omp_get_wtime();
    parallelBubbleSort(arr, n);
    double end_time = omp_get_wtime();

    cout<<"Sorted Array: ";
    printArray(arr, n);

    cout << endl;
    cout << "Time taken: " << end_time - start_time << " seconds" <<endl;


    return 0;
}