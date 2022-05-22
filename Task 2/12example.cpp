#include <omp.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

using namespace std;

int main(){
    const int size = 1000; //(argc > 1 ? atoi(argv[1]) : 1000); int argc, char **argv
    int *a = new int[size];
    #pragma omp parallel for
    for(int i = 0; i < size; i++) {
        a[i] = i;
    }
    int sum = 0;
    
    #pragma omp parallel for num_threads(4) reduction(+ : sum)
    for(int i = 0; i < size; i++){
        sum += a[i];
    }
    
    cout << "Sum = " << sum << endl;
    
    delete[] a;
    return 0;
}

