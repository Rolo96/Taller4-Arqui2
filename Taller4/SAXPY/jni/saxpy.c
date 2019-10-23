/********************************************************
This program will numerically compute the SAXPY operation
                  Z = s*X+Y
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
float32_t vectorZ1[SIZE1], vectorZ2[SIZE2], vectorZ3[SIZE3];

float32x4_t scalarParallel;
float scalarSerial;
 
double startTime, totalTime;

// -----------------------Methods------------------------------
void saxpySerial(float *x, float *y, float *z, int size)
{
	startTime = omp_get_wtime();
	for (int i = 0; i < size; ++i)
		z[i] = scalarSerial * x[i] + y[i];
	totalTime = omp_get_wtime() - startTime;
	printf("SAXPY serial with %d elements: %f seconds.\n", size, totalTime);
}

void saxpyParallel(float *x, float *y, float *z, int size)
{
	startTime = omp_get_wtime();
	
	#pragma omp parallel  
	{
		#pragma omp for
		for (int i = 0; i < size; i+=4){
			float32x4_t yData = vld1q_f32(y+i); 
			float32x4_t xData = vld1q_f32(x+i);
			float32x4_t zData = vmlaq_f32(yData, scalarParallel, xData); 
			vst1q_f32(z+i, zData); 
		}
	}
	totalTime = omp_get_wtime() - startTime;
	printf("SAXPY parallel with %d elements: %f seconds.\n", size, totalTime);
}

// ---------------------MAIN Method----------------------------
int main()
{
	srand(time(NULL));
	scalarSerial = rand();
	omp_set_num_threads(omp_get_num_procs());
	float32_t random = rand();
	scalarParallel = vdupq_n_f32(random);

	//Fill vectors
	#pragma omp parallel  
	{
		#pragma omp for
		for (int i = 0; i < SIZE1; ++i)
		{
			vectorX1[i] = rand()%100;
			vectorY1[i] = rand()%100;
		}
		#pragma omp for
		for (int i = 0; i < SIZE2; ++i)
		{
			vectorX2[i] = rand()%100;
			vectorY2[i] = rand()%100;
		}
		#pragma omp for
		for (int i = 0; i < SIZE3; ++i)
		{
			vectorX3[i] = rand()%100;
			vectorY3[i] = rand()%100;
		}
	}

	printf("\n SAXPY SERIAL\n");
	saxpySerial(vectorX1,vectorY1,vectorZ1,SIZE1);
	saxpySerial(vectorX2,vectorY2,vectorZ2,SIZE2);
	saxpySerial(vectorX3,vectorY3,vectorZ3,SIZE3);


	printf("\n SAXPY PARALLEL\n");
	saxpyParallel(vectorX1,vectorY1,vectorZ1,SIZE1);
	saxpyParallel(vectorX2,vectorY2,vectorZ2,SIZE2);
	saxpyParallel(vectorX3,vectorY3,vectorZ3,SIZE3);

	return 0;
}
