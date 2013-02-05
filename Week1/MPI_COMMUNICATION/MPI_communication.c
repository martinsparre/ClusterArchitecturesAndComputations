/* C Example */
#include <mpi.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define N_COLUMNS 1000

int main(int argc, char **argv){
	
	MPI_Status status;
	int MyID, Np,i,j;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD,&MyID);
	MPI_Comm_size(MPI_COMM_WORLD,&Np);
	
	printf("Processor number %d\n",MyID);
	
	//A vector used for communication
	float *Vector = malloc(N_COLUMNS*sizeof(float));		
	
	if (MyID == 0){
		//Initialize array
		float (*Array)[N_COLUMNS] =( float (*)[N_COLUMNS])malloc(Np * sizeof(*Array));
		for(i=0;i<Np;i++)
			for(j=0;j<N_COLUMNS;j++)
				Array[i][j] = i;

		//Send a part of the array (a vector) to each processor
		for(i = 1; i < Np; i++)
			MPI_Send(&Array[i],N_COLUMNS*sizeof(float),MPI_BYTE,i,1,MPI_COMM_WORLD);
	}
	
	
	//Receive vector, multiply by two, send it back
	if (MyID>0)
	{

		MPI_Recv(Vector,N_COLUMNS*sizeof(float),MPI_BYTE ,0,1,MPI_COMM_WORLD,&status);
		for(i=0;i<N_COLUMNS;i++)
			Vector[i] = 1.0*i;

		MPI_Send(Vector,N_COLUMNS*sizeof(float),MPI_BYTE ,0,2,MPI_COMM_WORLD);
	}


	if (MyID == 0){
		float (*ResArray)[N_COLUMNS] =( float (*)[N_COLUMNS])malloc(Np * sizeof(*ResArray));
		
		for(j=0;j<N_COLUMNS;j++)
			ResArray[0][j] = 0.0;		
		
		for(i=1;i<Np;i++)
		{
			MPI_Recv(Vector,N_COLUMNS*sizeof(float),MPI_BYTE ,i,2,MPI_COMM_WORLD,&status);
			for(j=0;j<N_COLUMNS;j++)
				ResArray[i][j] = Vector[j];
		}
		
		//calculate sum of all elements:
		double Sum = 0.0;
		for(i=0;i<Np;i++)
			for(j=0;j<N_COLUMNS;j++)
				Sum += ResArray[i][j];
		
		printf("%G\n",Sum);	
	}

	MPI_Barrier(MPI_COMM_WORLD);

	if (MyID == 0)
	{
		printf("Integral: %d\n",MyID);
	}
	MPI_Finalize();

    return 0;
}


