//*******************************************************************************************************************************************/
//**********************************************************************
//*
//* Real Time Embedded Systems 5623
//*
//**********************************************************************
//* Professor Sam Siewert
//**********************************************************************
//* Name of the File: capture.cpp
//**********************************************************************
//* Student: Vishvesh 
//* Student: Raghunath 
//*
//* Explanation: The code is written for the extended lab for Real Time Embedded System [5623] class.
//*              The code contains pthreads for real time processing
//*              The code captures frames at two rates, 1 Hz and 10 Hz. The detail analysis is present in the report submitted with the code
//*              The code saved the images in two formats, ppm and jpg. The ppm images header is modified to include header information in them.
//*              The code also performs Motion Detection in the beginning before starting the real time processing.
//* ----------------------------------------------------------------------------------------------------------------------------------------
//*******************************************************************************************************************************************/



// Header files
#include <pthread.h>
#include <stdio.h>
#include <sched.h>
#include <stdlib.h>
#include <semaphore.h>
#include <errno.h>
#include <unistd.h>
#include <errno.h>
#include <time.h>

#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/contrib/contrib.hpp"


using namespace cv;
using namespace std;


#define NUM_THREADS	5
#define SEQ_SERVICE 	0 //SEQUENCER SERVICE
#define PPM_ONE_SERVICE 	 1
#define JPG_ONE_SERVICE    2
#define JPG_TEN_SERVICE 4
#define NSEC_PER_SEC (1000000000)
#define DELAY_TICKS (1)
#define ERROR (-1)
#define OK (0)
#define AVG_COUNT 200
#define HRES_STR "640"
#define VRES_STR "480"


// For the images stored in ppm and jpg format
char ppm_header[]="P6\n#9999999999 sec 9999999999 msec \n"HRES_STR" "VRES_STR"\n255 \n";
char ppm_dumpname[]="test00000000.ppm";



// Global Variable 
void end_delay_test(void);
unsigned long a = 0;
unsigned long n = 1;

unsigned long b = 0;
unsigned long m = 1;

double c = 0;
double o = 1;

unsigned int tt = 1;
unsigned int uu = 1;
double kk = 1;


int dumpfd =0;
int written=0;

// For Frame Rate
static struct timespec delta_time={0,0};
static struct timespec frame_start_time={0,0};
static struct timespec jpeg_time={0,0};
static struct timespec ppm_time={0,0};
static struct timespec ten_time={0,0};
static struct timespec frame_stop_time ={0,0};
struct timespec frame_time;
struct timespec frame_time_jpeg;
struct timespec frame_time_ten;
double curr_frame_time;
double curr_frame_time_jpeg;
double curr_frame_time_ten;

// For Pthreads
pthread_t threads[NUM_THREADS];
pthread_attr_t rt_sched_attr[NUM_THREADS];
int rt_max_prio, rt_min_prio;
struct sched_param rt_param[NUM_THREADS];
struct sched_param nrt_param;
struct sched_param main_param;
struct timespec timeNow;
int abortTest = 0;
static unsigned int sleep_count = 0;
char snapshotname[80]="snapshot_xxx.ppm";
char snapname[80]="tenhertz_xxx.ppm";




// Mutex to protect the shared resources
pthread_mutex_t frame_mutex;


int rt_protocol;
volatile int runInterference=0, CScount=0;
volatile unsigned long long idleCount[NUM_THREADS];
int intfTime=0;

//CPU Set for affinity
cpu_set_t cpuset;	


void *startService(void *threadid);


// For timing calculation
// To find Delta value between start and stop time 
int delta_t(struct timespec *stop, struct timespec *start, struct timespec *delta_t)
{
  int dt_sec=stop->tv_sec - start->tv_sec;
  int dt_nsec=stop->tv_nsec - start->tv_nsec;

  if(dt_sec >= 0)
  {
    if(dt_nsec >= 0)
    {
      delta_t->tv_sec=dt_sec;
      delta_t->tv_nsec=dt_nsec;
    }
    else
    {
      delta_t->tv_sec=dt_sec-1;
      delta_t->tv_nsec=NSEC_PER_SEC+dt_nsec;
    }
  }
  else
  {
    if(dt_nsec >= 0)
    {
      delta_t->tv_sec=dt_sec;
      delta_t->tv_nsec=dt_nsec;
    }
    else
    {
      delta_t->tv_sec=dt_sec-1;
      delta_t->tv_nsec=NSEC_PER_SEC+dt_nsec;
    }
  }

  return(OK);
}

static struct timespec delay_error = {0, 0};


//Global variables to all threads 
IplImage* frame;
CvCapture* capture;

// This function performs Motion Detection
// If the camera detects a round object, then it will start the real time processing, otherwise it will wait for the round object
int motion_detection(void)
{

 Mat src, src_gray;

  /// Read the image
  
  namedWindow( "Hough Circle Transform Demo", CV_WINDOW_AUTOSIZE );
  int start_flag=0;
  
  while(start_flag==0)
 {
  if(cvGrabFrame(capture)) frame=cvRetrieveFrame(capture);
     
  if(!frame) break;
  cvSaveImage("circle.jpg", frame);
  src = imread( "circle.jpg", 1 );

  if( !src.data )
    { return -1; }
  

  /// Convert it to gray
  cvtColor( src, src_gray, CV_BGR2GRAY );

  /// Reduce the noise so we avoid false circle detection
  GaussianBlur( src_gray, src_gray, Size(9, 9), 2, 2 );

  vector<Vec3f> circles;
  

  /// Apply the Hough Transform to find the circles
  HoughCircles( src_gray, circles, CV_HOUGH_GRADIENT, 1, src_gray.rows/8, 200, 100, 0, 0 );
  

  /// Draw the circles detected
  for( size_t i = 0; i < circles.size(); i++ )
  {
      Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
      int radius = cvRound(circles[i][2]);
      // circle center
      circle( src, center, 3, Scalar(0,255,0), -1, 8, 0 );
      // circle outline
      circle( src, center, radius, Scalar(255,0,0), 3, 8, 0 );
      start_flag=1;
   }

  /// Show your results
  imshow( "Circle detection demo", src );
  waitKey(1);
  }


}



// This functions stores the frame in the jpg format
// The frames are captured at 1 frame per second rates
void * onejpg(void *)
{
    	
    pthread_mutex_lock(&frame_mutex);			
    clock_gettime(CLOCK_REALTIME, &frame_time_jpeg);
    curr_frame_time_jpeg=((double)frame_time_jpeg.tv_sec * 1000.0) + ((double)((double)frame_time_jpeg.tv_nsec / 1000000.0));	
    printf("\nCurrent frame time is seconds=%ld,nano_seconds=%ld\n",frame_start_time.tv_sec,((double)frame_start_time.tv_nsec) );
 
   if(b==0)
   {
	b = (unsigned)frame_time_jpeg.tv_sec;
	printf("\n Value of A is %u", a);
   }
   else
   {
	m = (unsigned)frame_time_jpeg.tv_sec;
	printf("\n Value of B is %lu", b);
	printf("\n Value of M is %lu", m);
	printf("\n Value of B-M is %lu", (m-b));

	if((m-b)==1)
	{
           printf("Taking snapshot number\n");
           sprintf(&snapshotname[9], "%8.4lf.jpg",kk);
	  
           cvSaveImage(snapshotname, frame);
           b = (unsigned)frame_time_jpeg.tv_sec;
           m = 0;
	   kk++;
         
	}
   }
    
	pthread_mutex_unlock(&frame_mutex);					
    clock_gettime(CLOCK_REALTIME, &jpeg_time);
    delta_t(&jpeg_time, &frame_start_time, &delta_time);
    printf("onejpg took DT seconds = %ld, nanoseconds = %ld\n", delta_time.tv_sec, delta_time.tv_nsec);
    //Printing difference in time for execution of onejpg 
    
}




void tenhz(void)
{
	
	tt = 0;
	
	while(tt<5401)
	{
	//frame=cvQueryFrame(capture);
	if(cvGrabFrame(capture)) frame=cvRetrieveFrame(capture);
     
        if(!frame) break;

	clock_gettime(CLOCK_REALTIME, &frame_time);
                curr_frame_time=((double)frame_time.tv_sec * 1000.0) + 
                                ((double)((double)frame_time.tv_nsec / 1000000.0));
	frame_count++;

	if(frame_count > 2)
	{
	fc=(double)frame_count;
        ave_framedt=((fc-1.0)*ave_framedt + framedt)/fc;
        ave_frame_rate=1.0/(ave_framedt/1000.0);
        }

        cvShowImage("Capture Example", frame);

	//printf("Frame %u sec and %lu nsec %d sec  %ld nsec\n", (unsigned)frame_time.tv_sec, (unsigned long)frame_time.tv_nsec,(unsigned)frame_time.tv_sec, (unsigned long)frame_time.tv_nsec);

            printf("Frame %u sec, %lu nsec, dt=%5.2lf msec, avedt=%5.2lf msec, rate=%5.2lf fps, frame count %d \n", 
                   (unsigned)frame_time.tv_sec, 
                   (unsigned long)frame_time.tv_nsec,
                   framedt, ave_framedt, ave_frame_rate,frame_count);

if(a==0)
{
a = (unsigned)frame_time.tv_sec;
}

if(a!=0)
{
n = (unsigned)frame_time.tv_sec;

if((n-a)==1)
{
                printf("Taking snapshot number %d\n",tt);
		dumpfd = open(ppm_dumpname, O_WRONLY | O_NONBLOCK | O_CREAT, 00666);
		snprintf(&ppm_dumpname[4], 9, "%08d", tt);
    		strncat(&ppm_dumpname[12], ".ppm", 5);

		snprintf(&ppm_header[4], 11, "%010d", (int)frame_time.tv_sec);
    		strncat(&ppm_header[14], " sec ", 5);
    		snprintf(&ppm_header[19], 11, "%010d", (int)((frame_time.tv_nsec)/1000000));
    		strncat(&ppm_header[29], " msec \n"HRES_STR" "VRES_STR"\n255\n", 19);
    		written=write(dumpfd, ppm_header, sizeof(ppm_header));

		close(dumpfd);

		//written=write(dumpfd,ppm_dumpname,0);

                //sprintf(&ppm_dumpname[9], "%d.ppm", tt);
                cvSaveImage(ppm_dumpname, frame);
                tt++;
                a = 0;
                n = 0;
}

}

	}



        char c = cvWaitKey(33);
        if( c == 27 ) break;

	if((c == 'q') || (c == 'Q'))
            {
                printf("Exiting ...\n\n\n\n");
                printf("Number of pictures taken %d\n",(tt));



		dumpfd = open(ppm_dumpname, O_WRONLY | O_NONBLOCK | O_CREAT, 00666);

		snprintf(&ppm_header[4], 11, "%010d", ((int)frame_time.tv_sec + 1));

    		strncat(&ppm_header[14], " sec ", 5);
    		snprintf(&ppm_header[19], 11, "%010d", (int)((frame_time.tv_nsec)/1000000));
    		strncat(&ppm_header[29], " msec \n"HRES_STR" "VRES_STR"\n255\n", 19);
    		written=write(dumpfd, ppm_header, sizeof(ppm_header));

		close(dumpfd);

                cvReleaseCapture(&capture);
                exit(0);
	
	
}


// This function stores the images in ppm format
// The header information are also added in this function
// The frames are captured at 1 frame per second rate
void * oneppm(void *)
{ 
  
   pthread_mutex_lock(&frame_mutex);				
   //Mat mat_frame(frame);
   //frame.copyTo(houghTempframe);  
   clock_gettime(CLOCK_REALTIME, &frame_time);
   curr_frame_time=((double)frame_time.tv_sec * 1000.0) + ((double)((double)frame_time.tv_nsec / 1000000.0));	
   printf("\nCurrent frame time is seconds=%ld,nano_seconds=%ld\n",frame_start_time.tv_sec,((double)frame_start_time.tv_nsec) );
 
   if(a==0)
   {
	a = (unsigned)frame_time.tv_sec;
	printf("\n Value of A is %u", a);
   }
   else
   {
	n = (unsigned)frame_time.tv_sec;
	printf("\n Value of A is %lu", a);
	printf("\n Value of N is %lu", n);
	printf("\n Value of N-A is %lu", (n-a));

	if((n-a)==1)
	{
        printf("Taking snapshot number %d\n",tt);
	dumpfd = open(ppm_dumpname, O_WRONLY | O_NONBLOCK | O_CREAT, 00666);
	snprintf(&ppm_dumpname[4], 9, "%08d", tt);
	strncat(&ppm_dumpname[12], ".ppm", 5);
	snprintf(&ppm_header[4], 11, "%010d", (int)frame_time_jpeg.tv_sec);
	strncat(&ppm_header[14], " sec ", 5);
	snprintf(&ppm_header[19], 11, "%010d", (int)((frame_time_jpeg.tv_nsec)/1000000));
	strncat(&ppm_header[29], " msec \n"HRES_STR" "VRES_STR"\n255\n", 19);
	written=write(dumpfd, ppm_header, sizeof(ppm_header));
	close(dumpfd);
        cvSaveImage(ppm_dumpname, frame);
	a = (unsigned)frame_time.tv_sec;
        n = 0;
	tt++;
	}
   }
   pthread_mutex_unlock(&frame_mutex);			
   clock_gettime(CLOCK_REALTIME, &ppm_time);
   delta_t(&ppm_time, &frame_start_time, &delta_time);
   printf("Frame completed entry DT seconds = %ld, nanoseconds = %ld\n", delta_time.tv_sec, delta_time.tv_nsec);
   
  
}


    Mat gray;
    vector<Vec3f> circles;



//Function to Print the scheduler used 
int CScnt=0;
void print_scheduler(void)
{
   int schedType;
   schedType = sched_getscheduler(getpid());
   switch(schedType)
   {
     case SCHED_FIFO:
	   printf("Pthread Policy is SCHED_FIFO\n");
	   break;
     case SCHED_OTHER:
	   printf("Pthread Policy is SCHED_OTHER\n");
       break;
     case SCHED_RR:
	   printf("Pthread Policy is SCHED_OTHER\n");
	   break;
     default:
       printf("Pthread Policy is UNKNOWN\n");
   }
}




// Main Program
// Performs the motion detection and captures frames from the camera at either 1 Hz or 10 Hz.
// Stores the frame in either ppm or jpg format 
int main (int argc, char *argv[])
{
	
	// Initialize the variables
   int rc, invSafe=0, i, scope;
   struct timespec sleepTime, dTime;
   double FrameAvgTime=0,FrameSumTime=0;
   int iterations=0,Passed=0,Missed=0;
   int dev=0;                                 // Default value for camera device 
   int resVal=0;                              //default value of resoultion
   double Jitter=0;             //

   //configuration values for resolution 
   // The code make use of the resolution 640x480
   int HRESArray[5]={80,160,320,640,1280};
   int VRESArray[5]={60,120,240,480,960};

   //configuration value for Deadline wrt to resolution 
   //Value set after calulating average for different resolution 
   int DeadlineArray[5]={0,0,0,140,0};

  int HRES;
  int VRES;
  double deadline;


  if (argc ==1)
  {
    printf("No Inputs using deafaults values\n");
  }
  else if (argc > 1)
  {
    if (argc >=2)
    {
      sscanf(argv[1] ,"%d",&dev);
      {
        if (argc ==3)
        {
          sscanf(argv[2] ,"%d",&resVal);   
        }
      }

    }
  }

  printf("%d %d %d \n",argv[1],argv[2]);
  if (resVal<5)
  {
   HRES=HRESArray[3];
   VRES=VRESArray[3];
   deadline =DeadlineArray[3];
  }
  else
   {
    HRES=HRESArray[3];
    VRES=VRESArray[3];
    deadline =DeadlineArray[3];
   } 

  printf("HRES %d VRES %d deadline %lf\n",HRES,VRES,deadline);




   printf("Main Started ");
   print_scheduler();

   //Setting attributtes of Services   
   pthread_attr_setinheritsched(&rt_sched_attr[JPG_ONE_SERVICE], PTHREAD_EXPLICIT_SCHED);
   pthread_attr_init(&rt_sched_attr[JPG_ONE_SERVICE]);
   pthread_attr_setschedpolicy(&rt_sched_attr[JPG_ONE_SERVICE], SCHED_FIFO);
  
   pthread_attr_init(&rt_sched_attr[PPM_ONE_SERVICE]);
   pthread_attr_setinheritsched(&rt_sched_attr[PPM_ONE_SERVICE], PTHREAD_EXPLICIT_SCHED);
   pthread_attr_setschedpolicy(&rt_sched_attr[PPM_ONE_SERVICE], SCHED_FIFO);

   
   //Find the max and min priority 
   rt_max_prio = sched_get_priority_max(SCHED_FIFO);
   rt_min_prio = sched_get_priority_min(SCHED_FIFO);
   rc=sched_getparam(getpid(), &nrt_param);
   main_param.sched_priority = rt_max_prio;

   //creating the main scheduler thread 
   rc=sched_setscheduler(getpid(), SCHED_FIFO, &main_param);
   print_scheduler();
   
   
   Mat gray, canny_frame, cdst;
  
   //Capture a Camera Device
   capture = (CvCapture*)cvCreateCameraCapture(dev);
   cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_WIDTH, HRES);
   cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_HEIGHT, VRES);

  //Initiate Mutex 
  pthread_mutex_init(&frame_mutex,NULL);

  //Set Priorities 
  // Assigning priorities to different threads
  rt_param[JPG_ONE_SERVICE].sched_priority = rt_max_prio-20;                                   //Set priority of Thread
  pthread_attr_setschedparam(&rt_sched_attr[JPG_ONE_SERVICE], &rt_param[JPG_ONE_SERVICE]);     //Set Schedule paramater of this thread

  rt_param[PPM_ONE_SERVICE].sched_priority = rt_max_prio-30;                                   //Set priority of Thread
  pthread_attr_setschedparam(&rt_sched_attr[PPM_ONE_SERVICE], &rt_param[PPM_ONE_SERVICE]);     //Set Schedule paramater of this thread
  
  rt_param[JPG_TEN_SERVICE].sched_priority = rt_max_prio-10;                                   //Set priority of Thread
  pthread_attr_setschedparam(&rt_sched_attr[JPG_TEN_SERVICE], &rt_param[JPG_TEN_SERVICE]);     //Set Schedule paramater of this thread

  // First the camera will detect the round object and then proceeds with capturing of the frames for real time processing
  motion_detection();

  while (1) 
  {
	//capture the frame
    clock_gettime(CLOCK_REALTIME, &frame_start_time);
	if(cvGrabFrame(capture)) frame=cvRetrieveFrame(capture);

	if(!frame) break;
  else
  {
    //Acquire start of frame 
                  
  }                                 



//Creating Threads 
    printf("Creating thread jpeg_one Thread\n\r");		
rc = pthread_create(&threads[JPG_ONE_SERVICE], &rt_sched_attr[JPG_ONE_SERVICE], oneppm, (void *)JPG_ONE_SERVICE);
	if (rc)
	{
	     printf("ERROR: pthread_create() rc is jpeg_one service rc value %d" , rc);
	     perror(NULL);
	     exit(-1);
	}

        printf("Creating thread ppm_one Thread\n\r");		
        rc = pthread_create(&threads[PPM_ONE_SERVICE], &rt_sched_attr[PPM_ONE_SERVICE], onejpg, (void *)PPM_ONE_SERVICE);
        
	if (rc)
	{
	     printf("ERROR: pthread_create() rc is ppm_one service rc value %d" , rc);
	     perror(NULL);
	     exit(-1);
	}
  
 
	//Wating for thread executions to complete and joining them
	//oneppm
    
	if(pthread_join(threads[JPG_ONE_SERVICE], NULL) == 0)
	 {
	   //printf(" JPG_ONE_SERVICE Thread completed\n");
	 }	
	 else
	 {
	   perror("ppm_thread Join");
	 }
       
       
	if(pthread_join(threads[PPM_ONE_SERVICE], NULL) == 0)
	 {
	  //printf("PPM_ONE_SERVICE Thread completed\n");
	 }	
	 else
	 {
	   perror("jpeg_thread Join");
	 }
   

	 //char q = cvWaitKey(100);
	 if( kk ==  20 )
	 {
      	printf("Quit resources and application \n"); 
	   //Exiting capture 	

	   break;

	   //exit(-1);	
	 }	 
    //printf("\n Recreate onejpg and oneppm\n\r");
	
	
	// This function captures the frames at 10 Hz
	tenhz();
	
    
    clock_gettime(CLOCK_REALTIME,&frame_stop_time);
    
  
// Jitter and Accumulated Latency calculation  
    Jitter = (delta_time.tv_nsec/1000000) - deadline;
    if (Jitter >0 )
    {
      Missed++;
      printf("Deadline Missed\n");
    }
    else
    {
      Passed++;
      printf("Deadline Not Missed\n");
    }
    printf("*****Frame No: %d Jitter :%lf\n",iterations,Jitter);
    //To find average execution time 
    
      FrameSumTime +=delta_time.tv_nsec;
      iterations++;

    
    printf("\nstart frame time is seconds=%ld,nano_seconds=%lf\n",frame_start_time.tv_sec,((double)frame_start_time.tv_nsec) );
    printf("\nstop frame time is seconds=%ld,nano_seconds=%lf\n",frame_stop_time.tv_sec,((double)frame_stop_time.tv_nsec) );
    delta_t(&frame_stop_time, &frame_start_time, &delta_time);
    printf("completed all transformation seconds = %ld, nanoseconds = %ld\n", delta_time.tv_sec, delta_time.tv_nsec);

    
      
    
 } 

   //Calculate Average Time 
   FrameAvgTime=FrameSumTime/iterations;
      
   //Print Summary	  
   printf("************************************Summary************************\n\r");
   printf("Resolution %d*%d \n\r",HRES,VRES);
   printf("Average time for all Transformation %f ms\n\r",FrameAvgTime/1000000);
   printf("Frame Rate for all Transformation %f Hz\n\r",1000/(FrameAvgTime/1000000));
   printf("Total Frames : %d\n",iterations);
   printf("Deadline Missed : %d\nDeadline Not Missed: %d\n\r",Missed,Passed);
   printf("Number of time looped into the ten hertz frames is %d\n",uu);
   printf("*********************************END Summary************************\n\r");
   cvReleaseCapture(&capture);  	
   //Exiting Thread 
   
   return 0;
}



