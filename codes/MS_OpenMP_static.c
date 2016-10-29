 /* 
   Sequential Mandelbrot set
 */

//./MS_OpenMP_static 4 -2 2 -2 2 400 400 enable
#include <X11/Xlib.h>
#include <stdio.h>
#include <unistd.h>
#include <omp.h>
//atoi
#include <stdlib.h>
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

	/* open connection with the server */ 
	display = XOpenDisplay(NULL);
	if(display == NULL) {
		fprintf(stderr, "cannot open display\n");
		return 0;
	}

	screen = DefaultScreen(display);
	/* create graph */
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
	int able = strncmp(argv[8], "enable", 6);
	double rscale = width/(rright - roffset);
	double iscale = height/(iright - ioffset);


    if(able ==0){
		/* set window position */
		int x = 0;
		int y = 0;

		/* border width in pixels */
		int border_width = 0;

		/* create window */
		window = XCreateSimpleWindow(display, RootWindow(display, screen), x, y, width, height, border_width,
						BlackPixel(display, screen), WhitePixel(display, screen));
		
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
	
	/* draw points */
	int i, j;
	#pragma omp parallel shared(window, gc , rscale, roffset, iscale, ioffset, i) private(  j ) num_threads(thread_num) 
	{
		Compl z, c;
		int repeats;
		double temp, lengthsq;
		//#pragma omp for schedule(static) nowait
		#pragma omp for schedule(static)
		for(i=0; i<width; i++) { 
			for(j=0; j<height; j++) {
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

				#pragma omp critical
				{
				if(able ==0){
					XSetForeground (display, gc,  1024 * 1024 * (repeats % 256));	
					XDrawPoint (display, window, gc, i, j);
				}
				}
				
			}
		}
	}
	
    if(able ==0){
		XFlush(display);
		sleep(5);
	}
	clock_gettime(CLOCK_REALTIME, &tt2);
	printf("total time: %ld sec\n ", tt2.tv_sec - tt1.tv_sec+ tt2.tv_nsec*pow (10.0, -9.0) - tt1.tv_nsec*pow (10.0, -9.0));
	return 0;
}
