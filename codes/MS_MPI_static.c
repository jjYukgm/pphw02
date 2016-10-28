 /* 
   Sequential Mandelbrot set
 */
// mpirun -n 2 ./MS_MPI_static 4 -2 2 -2 2 400 400 enable
#include <X11/Xlib.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <mpi.h>
//enable & disable
#include <string.h>

typedef struct complextype
{
	double real, imag;
} Compl;
typedef struct commtype
{
	int i, j, repeats;
} Comm;

int main(int argc, char *argv[])
{
	Display *display;
	Window window;      //initialization for a window
	int screen;         //which screen 

	/* open connection with the server */ 
	display = XOpenDisplay(NULL);
	if(display == NULL) {
		fprintf(stderr, "cannot open display\n");
		return 0;
	}

	screen = DefaultScreen(display);

	
	double roffset  = atof(argv[2]);
	double rright = atof(argv[3]);
	double ioffset  = atof(argv[4]);
	double iright = atof(argv[5]);
	/* set window size */
	int width = atoi(argv[6]);
	int height = atoi(argv[7]);
	//char *xin = argv[8];
	int able = strncmp(argv[8], "enable", 6);
	
	double rscale = width/(rright - roffset);
	double iscale = height/(iright - ioffset);

	int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
	
	GC gc;
	if(rank==0){
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
	int inij=0;
	int fini=width;
	int finj=height;
	int coresize=0;
	if(width >= height){
		if(width%2 !=0)
			coresize = width/size + 1;
		else
			coresize = width/size;
		
		inii = coresize * rank;
		if(fini > (inii + coresize))
			fini = (inii + coresize);
	}
	else{
		if(height%2 !=0)
			coresize = height/size + 1;
		else
			coresize = height/size;
		inij = coresize * rank;
		if(finj > (inij + coresize))
			finj = inij + coresize;
	}
	
	/* draw points */
	Compl z, c;
	struct commtype self;
	struct commtype *remote;
	
    if(rank ==0){
		remote = (struct commtype *)malloc(sizeof(struct commtype) * size);
	}
	int repeats;
	double temp, lengthsq;
	int i, j, k;
	
	MPI_Barrier( MPI_COMM_WORLD );
	for(i=inii; i<fini; i++) {
		for(j=inij; j<finj; j++) {
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
			self.i = i;
			self.j = j;
			self.repeats = repeats;
			MPI_Gather( &self, 3, MPI_INT,
					   remote, 3, MPI_INT,
					   0, MPI_COMM_WORLD);
			if(rank==0){
				XSetForeground (display, gc,  1024 * 1024 * (repeats % 256));		
				XDrawPoint (display, window, gc, i, j);
				for(k=1;k<size;k++){
					XSetForeground (display, gc,  1024 * 1024 * (remote[k].repeats % 256));		
					XDrawPoint (display, window, gc, remote[k].i, remote[k].j);
				}
					
			}
			//MPI_Barrier( MPI_COMM_WORLD );
		}
	}
    if(rank ==0 && able ==0){
		XFlush(display);
		sleep(5);
	}
	free((void *)remote);
	
    MPI_Finalize();
	return 0;
}
