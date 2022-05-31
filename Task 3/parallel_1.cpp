#include <iostream>
#include <stdio.h>
#include <mpi.h>

#define _USE_MATH_DEFINES
#include <math.h>

#define MAX_ITERATIONS 50
#define SIZE 10000
#define Epsilon 1e-10


double* StateProblemConditions(double* matrixMpi, double* b, double* x, int shift, int num) {
    double* sol = (double*) calloc(SIZE, sizeof(double));
    for (int i = 0; i < SIZE; i++){
        sol[i] = sin(2*M_PI*i/SIZE);
    }
    for (int i = 0; i < num; i++) {
        double loc_sum = 0;
        for (int j = 0; j < SIZE; j++) {
            if (shift + i == j) {
                matrixMpi[i*SIZE + j] = 2.0;
            } else {
                matrixMpi[i*SIZE + j] = 1.0;
            }
            loc_sum += matrixMpi[i*SIZE + j] * sol[j];
        }
        b[i] = loc_sum;
        x[i] = 1.0;
    }
    return sol;
}

double norm(double* vector, int size) {
    double result = 0;
    for (int i = 0; i < size; i++) {
        result += vector[i] * vector[i];
    }
    return sqrt(result);
}

double dotProduct(double* mul1, double* mul2, int size) {
    double result = 0;
    for (int i = 0; i < size; i++) {
        result += mul1[i] * mul2[i];
    }
    return result;
}

void DoMatrixTransform(double* matrix, double* vector, double* result, int num) {
    for (int i = 0; i < num; i++){
        result[i] = 0;
        for (int j = 0; j < SIZE; j++){
            result[i] += matrix[i*SIZE + j] * vector[j];
        }
    }
}

void difference(double* minuend, double* subtrahend, double* output, int size) {
    for (int i = 0; i < size; i++) {
        output[i] = minuend[i] - subtrahend[i];
    }
}

double GetError(double* act, double* expc){
    double asw = 0.0;
    for (int i = 0; i < SIZE; i++){
        asw += pow(act[i] - expc[i], 2);
    }
    return sqrt(asw);
}

int main(int argc, char* argv[]) {
    int size, rank;
    MPI_Init(&argc, &argv);// инициализация MPI
    MPI_Comm_size(MPI_COMM_WORLD, &size);// получение числа процессов
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);// получение номера процесса
    
    int* numberOfElements = (int*) calloc(size, sizeof(int));// количество элементов в каждом процессе
    for (int i = 0; i < size; ++i) {
        numberOfElements[i] = (SIZE / size) + ((i < SIZE % size) ? (1) : (0));
    }
    int* shift = (int*) calloc(size, sizeof(int));// сдвиг по индексам в каждом процессе
    for (int i = 1; i < size; ++i) {
        shift[i] = shift[i - 1] + numberOfElements[i - 1];
    }
    
    double* b = (double*) calloc(SIZE, sizeof(double));
    double* x = (double*) calloc(SIZE, sizeof(double));
    double* y = (double*) calloc(SIZE, sizeof(double));
    double* B_MPI = (double*) calloc(numberOfElements[rank], sizeof(double));
    double* X_MPI = (double*) calloc(numberOfElements[rank], sizeof(double));
    double* A_MPI = (double*) calloc(SIZE * numberOfElements[rank], sizeof(double));
    double* y_MPI = (double*) calloc(numberOfElements[rank], sizeof(double));
    double* AX_MPI = (double*) calloc(numberOfElements[rank], sizeof(double));
    double* AY_MPI = (double*) calloc(numberOfElements[rank], sizeof(double));
    
    double* sol = StateProblemConditions(A_MPI, B_MPI, X_MPI, shift[rank], numberOfElements[rank]);
    
    MPI_Allgatherv(X_MPI, numberOfElements[rank], MPI_DOUBLE, x, numberOfElements, shift, MPI_DOUBLE, MPI_COMM_WORLD);// собираем в-р x по кусочкам из разных процессов в каждом процессе
    MPI_Allgatherv(B_MPI, numberOfElements[rank], MPI_DOUBLE, b, numberOfElements, shift, MPI_DOUBLE, MPI_COMM_WORLD);// собираем в-р b по кусочкам из разных процессов в каждом процессе
    
    double tau = 0;
    double numerator = 0;
    double denominator = 0;
    double denominatorSum = 0;
    double numeratorSum = 0;
    double norm_b = norm(b, SIZE);
    bool StopCond = false;
    
    double procStart = MPI_Wtime();
    int iterations = 0;
    while (!StopCond && iterations < MAX_ITERATIONS) {
        numerator = 0;
        numeratorSum = 0;
        denominator = 0;
        denominatorSum = 0;
        
        DoMatrixTransform(A_MPI, x, AX_MPI, numberOfElements[rank]);
        
        difference(AX_MPI, B_MPI, y_MPI, numberOfElements[rank]);
        
        MPI_Allgatherv(y_MPI, numberOfElements[rank], MPI_DOUBLE, y, numberOfElements, shift, MPI_DOUBLE, MPI_COMM_WORLD);// собираем в-р y по кусочкам из разных процессов в каждом процессе

        DoMatrixTransform(A_MPI, y, AY_MPI, numberOfElements[rank]);

        numerator = dotProduct(y_MPI, AY_MPI, numberOfElements[rank]);
        denominator = dotProduct(AY_MPI, AY_MPI, numberOfElements[rank]);
        
        MPI_Allreduce(&numerator, &numeratorSum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);// суммируем значения "локальных" numerator по всем потокам в "глобальную" numeratorSum
        MPI_Allreduce(&denominator, &denominatorSum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);// суммируем значения "локальных" denominator по всем потокам в "глобальную" denominatorSum
        tau = numeratorSum / denominatorSum;// итоговый шаг tau для полного x
        
        for (int i = 0; i < numberOfElements[rank]; ++i) {// шаг по x
            X_MPI[i] -= tau * y_MPI[i];
        }
        MPI_Allgatherv(X_MPI, numberOfElements[rank], MPI_DOUBLE, x, numberOfElements, shift, MPI_DOUBLE, MPI_COMM_WORLD);// собираем в-р x по кусочкам из разных процессов в каждом процессе
        
        StopCond = norm(y, SIZE)/norm_b <= Epsilon;
        iterations++;
    }
    
    double procFinish = MPI_Wtime();
    double procTime = procFinish - procStart;
    double FullTime;
    MPI_Allreduce(&procTime, &FullTime, 1, MPI_DOUBLE, MPI_MAX,  MPI_COMM_WORLD);// за время выполнения программы возьмём время выполенения самого долгого процесса
    if (rank == 0){
        double error = GetError(sol, x);
        std::cout << "Norm of error vector = " << error << std::endl;
        std::cout << "Iteration number: " << iterations << std::endl;
        std::cout << "Work time: " << FullTime << std::endl;
    }
    
    free(sol);
    free(b);
    free(x);
    free(y);
    free(B_MPI);
    free(X_MPI);
    free(y_MPI);
    free(A_MPI);
    free(AX_MPI);
    free(AY_MPI);
    
    free(numberOfElements);
    free(shift);
    
    MPI_Finalize();// завершение работы MPI
    return 0;
}


























