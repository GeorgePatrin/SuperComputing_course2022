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
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < num; j++) {
            if (shift + j == i) {
                matrixMpi[i*num + j] = 2.0;
            } else {
                matrixMpi[i*num + j] = 1.0;
            }
        }
    }
    for (int i = 0; i < num; i++) {
        double sumStuff = 0;
        for (int j = 0; j < SIZE; j++){
            sumStuff += sol[j] * 1.0;
        }
        b[i] = sumStuff + (2.0 - 1.0)*sol[shift + i];
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

double dotProduct(double* mul1, int shift1, double* mul2, int shift2, int size) {
    double result = 0;
    for (int i = 0; i < size; i++) {
        result += mul1[shift1 + i] * mul2[shift2 + i];
    }
    return result;
}

void DoMatrixTransform(double* matrix, double* vector, double* result, int num) {
    for (int i = 0; i < SIZE; i++){
        result[i] = 0;
        for (int j = 0; j < num; j++){
            result[i] += matrix[i*num + j] * vector[j];
        }
    }
}

void difference(double* minuend, double* subtrahend, double* output, int shift, int size) {
    for (int i = 0; i < size; i++) {
        output[i] = minuend[shift + i] - subtrahend[i];
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
    
    double* B_MPI = (double*) calloc(numberOfElements[rank], sizeof(double));
    double* X_MPI = (double*) calloc(numberOfElements[rank], sizeof(double));
    double* A_MPI = (double*) calloc(SIZE * numberOfElements[rank], sizeof(double));
    double* Y_MPI = (double*) calloc(numberOfElements[rank], sizeof(double));
    double* AX_MPI = (double*) calloc(SIZE, sizeof(double));
    double* AY_MPI = (double*) calloc(SIZE, sizeof(double));
    double* AX = (double*) calloc(SIZE, sizeof(double));
    double* AY = (double*) calloc(SIZE, sizeof(double));
    
    double*  sol = StateProblemConditions(A_MPI, B_MPI, X_MPI, shift[rank], numberOfElements[rank]);
    
    double b_norm_stuff = dotProduct(B_MPI, 0, B_MPI, 0, numberOfElements[rank]);
    double norm_b;
    MPI_Allreduce(&b_norm_stuff, &norm_b, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);// норма в-ра b собирается по кусочкам
    norm_b = sqrt(norm_b);
    
    double tau = 0;
    double numerator = 0;
    double denominator = 0;
    double denominatorSum = 0;
    double numeratorSum = 0;
    bool StopCond = false;
    
    double procStart = MPI_Wtime();
    int iterations = 0;
    while (!StopCond && iterations < MAX_ITERATIONS) {
        numerator = 0;
        numeratorSum = 0;
        denominator = 0;
        denominatorSum = 0;
        
        DoMatrixTransform(A_MPI, X_MPI, AX_MPI, numberOfElements[rank]);
        MPI_Allreduce(AX_MPI, AX, SIZE, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);// собираем полное произведение матрицы на вектор путём суммирования
        
        difference(AX_MPI, B_MPI, Y_MPI, shift[rank], numberOfElements[rank]);

        DoMatrixTransform(A_MPI, Y_MPI, AY_MPI, numberOfElements[rank]);
        MPI_Allreduce(AY_MPI, AY, SIZE, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        numerator = dotProduct(Y_MPI, 0, AY_MPI, shift[rank], numberOfElements[rank]);
        denominator = dotProduct(AY, shift[rank], AY, shift[rank], numberOfElements[rank]);
        
        MPI_Allreduce(&numerator, &numeratorSum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);// суммируем значения "локальных" numerator по всем потокам в "глобальную" numeratorSum
        MPI_Allreduce(&denominator, &denominatorSum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);// суммируем значения "локальных" denominator по всем потокам в "глобальную" denominatorSum
        tau = numeratorSum / denominatorSum;// итоговый шаг tau для полного x
        
        for (int i = 0; i < numberOfElements[rank]; ++i) {// шаг по x
            X_MPI[i] -= tau * Y_MPI[i];
        }
        
        double y_norm_stuff = dotProduct(Y_MPI, 0, Y_MPI, 0, numberOfElements[rank]);
        double norm_y;
        MPI_Allreduce(&y_norm_stuff, &norm_y, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);// норма в-ра y собирается по кусочкам
        norm_y = sqrt(norm_y);
        
        StopCond = norm_y/norm_b <= Epsilon;
        iterations++;
    }
    
    double procFinish = MPI_Wtime();
    double procTime = procFinish - procStart;
    double FullTime;
    MPI_Allreduce(&procTime, &FullTime, 1, MPI_DOUBLE, MPI_MAX,  MPI_COMM_WORLD);// за время выполнения программы возьмём время выполенения самого долгого процесса
    
    double x[SIZE] = {0};
    MPI_Allgatherv(X_MPI, numberOfElements[rank], MPI_DOUBLE, x, numberOfElements, shift, MPI_DOUBLE, MPI_COMM_WORLD);// для подсчёта погрешности
    if (rank == 0){
        double error = GetError(sol, x);
        std::cout << "Norm of error vector = " << error << std::endl;
        std::cout << "Iteration number: " << iterations << std::endl;
        std::cout << "Work time: " << FullTime << std::endl;
    }
    
    free(B_MPI);
    free(X_MPI);
    free(Y_MPI);
    free(A_MPI);
    free(AX_MPI);
    free(AY_MPI);
    free(AY);
    free(AX);
    free(sol);
    
    free(numberOfElements);
    free(shift);
    
    MPI_Finalize();// завершение работы MPI
    return 0;
}


























