 /* 
   Sequential Mandelbrot set
 */
// mpirun -n 2 ./MS_Hybrid_static 4 -2 2 -2 2 400 400 enable
#include <X11/Xlib.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <mpi.h>
#include <omp.h>
//enable & disable
#include <string.h>
#include <time.h>//time measure
#include <math.h>//time calculate

typedef struct complextype
{
	double real, imag;
} Compl;

int main(int argc, char *argv[])
{
	Display *display;
	Window window;      //initialization for a window
	int screen;         //which screen 

	int able = strncmp(argv[8], "enable", 6);
	if(able ==0){
		/* open connection with the server */ 
		display = XOpenDisplay(NULL);
		if(display == NULL) {
			fprintf(stderr, "cannot open display\n");
			return 0;
		}

		screen = DefaultScreen(display);
	}
	GC gc;

	//time measure
	struct timespec tt1, tt2;
	clock_gettime(CLOCK_REALTIME, &tt1);
	
	int thread_num = atoi(argv[1]);
	double roffset  = atof(argv[2]);
	double rright = atof(argv[3]);
	double ioffset  = atof(argv[4]);
	double iright = atof(argv[5]);
	/* set window size */
	int width = atoi(argv[6]);
	int height = atoi(argv[7]);
	//char *xin = argv[8];
	
	double rscale = width/(rright - roffset);
	double iscale = height/(iright - ioffset);

	int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
	
	if(rank==0 && able ==0){
		/* set window position */
		int x = 0;
		int y = 0;

		/* border width in pixels */
		int border_width = 0;

		/* create window */
		window = XCreateSimpleWindow(display, RootWindow(display, screen), x, y, width, height, border_width,
						BlackPixel(display, screen), WhitePixel(display, screen));
		
		/* create graph */
		XGCValues values;
		long valuemask = 0;
		
		gc = XCreateGC(display, window, valuemask, &values);
		//XSetBackground (display, gc, WhitePixel (display, screen));
		XSetForeground (display, gc, BlackPixel (display, screen));
		XSetBackground(display, gc, 0X0000FF00);
		XSetLineAttributes (display, gc, 1, LineSolid, CapRound, JoinRound);
		
		/* map(show) the window */
		XMapWindow(display, window);
		XSync(display, 0);
	}
	
	//assign length
	int inii=0;
	//int inij=0;
	int fini=width;
	//int finj=height;
	int coresize=0;
	/*int master=0;
	if(size>1){
		size--;
		master = 1;
	}*/
	//if(width >= height){
		if(width%2 !=0)
			coresize = width/size + 1;
		else
			coresize = width/size;
		
		inii = coresize * rank;
		if(fini > (inii + coresize))
			fini = (inii + coresize);
	/*}
	else{
		if(height%2 !=0)
			coresize = height/size + 1;
		else
			coresize = height/size;
		inij = coresize * rank;
		if(finj > (inij + coresize))
			finj = inij + coresize;
	}*/
	
	//printf("[%d]before for loop \n", rank);
	MPI_Barrier( MPI_COMM_WORLD );
	#pragma omp parallel shared(window, gc , rscale, roffset, iscale, ioffset, inii, fini, height, tt1) private(  tt2) num_threads(thread_num) 
	{
		
		/* draw points */
		int *remote_i, *remote_repeats;
		int *self_repeats = (int *) malloc(sizeof(int) * height);
		
		
		if(rank ==0 ){
			remote_i = (int *)malloc(sizeof(int) * size);
			remote_repeats = (int *)malloc(sizeof(int) * height * size);
			//remote = (struct commtype *)malloc(sizeof(int)*( height + 1 ) * size);
		}
		
		int i, j, k;
		int pt=0;
		Compl z, c;
		int repeats;
		double temp, lengthsq;
		#pragma omp for schedule(static) //collapse(2)
		for(i=inii; i<fini; i++) {
			//if(master==0 || rank > 0)
			for(j=0; j<height; j++) {
				pt+=1;
				z.real = 0.0;
				z.imag = 0.0;
				c.real = (double)i / rscale + roffset; /* Theorem : If c belongs to M(Mandelbrot set), then |c| <= 2 */
				c.imag = (double)j / iscale + ioffset; /* So needs to scale the window */
				repeats = 0;
				lengthsq = 0.0;

				while(repeats < 100000 && lengthsq < 4.0) { /* Theorem : If c belongs to M, then |Zn| <= 2. So Zn^2 <= 4 */
					temp = z.real*z.real - z.imag*z.imag + c.real;
					z.imag = 2*z.real*z.imag + c.imag;
					z.real = temp;
					lengthsq = z.real*z.real + z.imag*z.imag; 
					repeats++;
				}
				self_repeats[j] = repeats;
				//MPI_Barrier( MPI_COMM_WORLD );
			}
			//MPI_Barrier( MPI_COMM_WORLD );
			#pragma omp critical
			{
			MPI_Gather( &i, 1 , MPI_INT,
					   remote_i, 1 , MPI_INT,
					   0, MPI_COMM_WORLD);
			MPI_Gather( self_repeats, height , MPI_INT,
					   remote_repeats, height  , MPI_INT,
					   0, MPI_COMM_WORLD);
			if(rank ==0 && able ==0){
				for(k=0;k<size;k++){
					for(j=0;j<height;j++){
						XSetForeground (display, gc,  1024 * 1024 * (remote_repeats[k*height+j] % 256));	
						XDrawPoint (display, window, gc, remote_i[k], j);
					}
				}
			}
			}
		}
		clock_gettime(CLOCK_REALTIME, &tt2);
		printf("[n%d][t%d]  pt: %d	;comp Time: %.3f sec\n", rank, omp_get_thread_num(), pt, tt2.tv_sec - tt1.tv_sec+ tt2.tv_nsec*pow (10.0, -9.0) - tt1.tv_nsec*pow (10.0, -9.0));
		//printf("[%d]after for loop \n", rank);
		if(rank ==0 && able ==0){
			XFlush(display);
			sleep(5);
		}
		if(rank==0){
			free((void *)remote_i);
			free((void *)remote_repeats);
		}
		free((void *)self_repeats);
	}
	/*
	printf("[%d]total time: %.3f sec\n ", rank, tt2.tv_sec - tt1.tv_sec+ tt2.tv_nsec*pow (10.0, -9.0) - tt1.tv_nsec*pow (10.0, -9.0));*/
	
    MPI_Finalize();
	return 0;
}
