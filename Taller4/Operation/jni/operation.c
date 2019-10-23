/********************************************************
This program will numerically compute the operation of vectors Z = (V - W) + (X - Y) 
Raul Arias Quesada
2015061976
*********************************************************/

// -----------------------Includes-----------------------------
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <arm_neon.h> 
#include <time.h>

// -----------------------Constants----------------------------
#define SIZE1 10000000
#define SIZE2 20000000
#define SIZE3 30000000

// -----------------------Variables----------------------------
float32_t vectorX1[SIZE1], vectorX2[SIZE2], vectorX3[SIZE3];
float32_t vectorY1[SIZE1], vectorY2[SIZE2], vectorY3[SIZE3];
float32_t vectorV1[SIZE1], vectorV2[SIZE2], vectorV3[SIZE3];
float32_t vectorW1[SIZE1], vectorW2[SIZE2], vectorW3[SIZE3];
float32_t vectorZ1[SIZE1], vectorZ2[SIZE2], vectorZ3[SIZE3];

double startTime, totalTime;

// -----------------------Methods------------------------------
void operationSerial(float *v, float *w, float *x, float *y, float *z, int size)
{
	startTime = omp_get_wtime();
	for (int i = 0; i < size; ++i)
		z[i] = (v[i] - w[i]) + (x[i] - y[i]);
	totalTime = omp_get_wtime() - startTime;
	printf("Operation serial with %d elements: %f seconds.\n", size, totalTime);
}

void operationParallel(float *v, float *w, float *x, float *y, float *z, int size)
{
	startTime = omp_get_wtime();
	
	#pragma omp parallel  
	{
		#pragma omp for
		for (int i = 0; i < size; i+=4){
			float32x4_t vData = vld1q_f32(v+i); 
			float32x4_t wData = vld1q_f32(w+i);
			float32x4_t yData = vld1q_f32(y+i); 
			float32x4_t xData = vld1q_f32(x+i);
			float32x4_t aData = vsubq_f32(vData, wData); 
			float32x4_t bData = vsubq_f32(xData, yData); 
			float32x4_t zData = vaddq_f32(aData, bData); 
			vst1q_f32(z+i, zData); 
		}
	}
	totalTime = omp_get_wtime() - startTime;
	printf("Operation parallel with %d elements: %f seconds.\n", size, totalTime);
}

// ---------------------MAIN Method----------------------------
int main()
{
	srand(time(NULL));
	omp_set_num_threads(omp_get_num_procs());
	
	//Fill vectors
	#pragma omp parallel  
	{
		#pragma omp for
		for (int i = 0; i < SIZE1; ++i)
		{
			vectorX1[i] = rand()%100;
			vectorY1[i] = rand()%100;
			vectorV1[i] = rand()%100;
			vectorW1[i] = rand()%100;
		}
		#pragma omp for
		for (int i = 0; i < SIZE2; ++i)
		{
			vectorX2[i] = rand()%100;
			vectorY2[i] = rand()%100;
			vectorV2[i] = rand()%100;
			vectorW2[i] = rand()%100;
		}
		#pragma omp for
		for (int i = 0; i < SIZE3; ++i)
		{
			vectorX3[i] = rand()%100;
			vectorY3[i] = rand()%100;
			vectorV3[i] = rand()%100;
			vectorW3[i] = rand()%100;
		}
	}

	printf("\n OPERATION SERIAL\n");
	operationSerial(vectorV1, vectorW1, vectorX1, vectorY1, vectorZ1, SIZE1);
	operationSerial(vectorV2, vectorW2, vectorX2, vectorY2, vectorZ2, SIZE2);
	operationSerial(vectorV3, vectorW3, vectorX3, vectorY3, vectorZ3, SIZE3);


	printf("\n OPERATION PARALLEL\n");
	operationParallel(vectorV1, vectorW1, vectorX1, vectorY1, vectorZ1, SIZE1);
	operationParallel(vectorV2, vectorW2, vectorX2, vectorY2, vectorZ2, SIZE2);
	operationParallel(vectorV3, vectorW3, vectorX3, vectorY3, vectorZ3, SIZE3);

	return 0;
}
