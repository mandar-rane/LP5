#include <iostream>
#include <omp.h>

using namespace std;

void merge(int arr[], int l, int m, int r){
    int i, j, k;
    int n1 = m - l + 1;
    int n2 = r-m;

    int L[n1], R[n2];

    for(i=0; i<n1; i++){
        L[i] = arr[l + i];
    }

    for(j=0; j<n2; j++){
        R[j] = arr[m + 1 + j];
    }

    i = 0;
    j = 0;
    k = l;

    while(i<n1 && j<n2){
        if(L[i] <= R[j]){
            arr[k] = L[i];
            i++;
        }else{
            arr[k] = R[j];
            j++;
        }
        k++;
    }

    while(i<n1){
        arr[k] = L[i];
        i++;
        k++;
    }

    while(j<n2){
        arr[k] = R[j];
        j++;
        k++;
    }
}

void parallelMergeSort(int arr[], int l, int r){
    if(l<r){
        int m = l+(r-l)/2;
        #pragma omp parallel sections
        {
            #pragma omp section
            parallelMergeSort(arr, l, m);
            #pragma omp section
            parallelMergeSort(arr, m+1, r);
        }
        merge(arr, l, m, r);
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
    parallelMergeSort(arr, 0, n-1);
    double end_time = omp_get_wtime();

    cout<<"Sorted Array: ";
    printArray(arr, n);

    cout << endl;

    cout << "Time taken: " << end_time - start_time << " seconds" <<endl;

    return 0;
}

