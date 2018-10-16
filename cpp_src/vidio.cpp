#include "stdafx.h"
#include <cv.h>
#include <cvaux.h>
#include <highgui.h>

#pragma comment(lib, "ml.lib")
#pragma comment(lib, "cv.lib")
#pragma comment(lib, "cvaux.lib")
#pragma comment(lib, "cvcam.lib")
#pragma comment(lib, "cxcore.lib")
#pragma comment(lib, "cxts.lib")
#pragma comment(lib, "highgui.lib")
#pragma comment(lib, "cvhaartraining.lib")

int _tmain(int argc, _TCHAR* argv[])
{
	IplImage *newFrame=NULL;
	IplImage *frame1=NULL;
	IplImage *frame2=NULL;
	CvCapture * pCapture1 = cvCaptureFromAVI(argv[1]);
	CvCapture * pCapture2 = cvCaptureFromAVI(argv[2]);
	CvRect rect;
	CvVideoWriter *pWriter=NULL;
	//get the frame number of two videos
	int frameNo1 = (int) cvGetCaptureProperty(pCapture1, CV_CAP_PROP_FRAME_COUNT);
	int frameNo2 = (int) cvGetCaptureProperty(pCapture2, CV_CAP_PROP_FRAME_COUNT);
	if(frameNo1!=frameNo2)
		printf("video length 1 != video length 2\n");
	else
		printf("Total frame numbers: %d\n",frameNo1);
	//get the frame width of two videos
	int frameWidth1 = (int) cvGetCaptureProperty(pCapture1, CV_CAP_PROP_FRAME_WIDTH);
	int frameWidth2 = (int) cvGetCaptureProperty(pCapture2, CV_CAP_PROP_FRAME_WIDTH);
	if(frameWidth1!=frameWidth2)
		printf("video width 1 != video width 2\n");
	else
		printf("frame width: %d\n",frameWidth1);
	//get the frame height of two videos
	int frameHeight1 = (int) cvGetCaptureProperty(pCapture1, CV_CAP_PROP_FRAME_HEIGHT );
	int frameHeight2 = (int) cvGetCaptureProperty(pCapture2, CV_CAP_PROP_FRAME_HEIGHT );
	if(frameHeight1!=frameHeight2)
		printf("video height 1 != video height 2\n");
	else
		printf("frame height: %d\n",frameHeight1);
	//get the video fps
	int fps1 = (int) cvGetCaptureProperty(pCapture1, CV_CAP_PROP_FPS  );
	int fps2 = (int) cvGetCaptureProperty(pCapture2, CV_CAP_PROP_FPS  );
	if(fps1!=fps2)
		printf("video fps 1 != video fps 2\n");
	else
		printf("frame fps: %d\n",fps1);

	int initFlag=0;
	int counter=0;
	while((frame1=cvQueryFrame(pCapture1))!=NULL && (frame2=cvQueryFrame(pCapture2))!=NULL )
	{
		printf("%d\n",counter++);
		if(initFlag==0)
		{
			newFrame = cvCreateImage(cvSize(frame1->width*2,frame1->height),frame1->depth,frame1->nChannels);
			pWriter = cvCreateVideoWriter("re.avi",CV_FOURCC('X','V','I','D'),fps1,cvSize(frame1->width*2,frame1->height),1);
			initFlag=1;
		}
		rect.x=0;
		rect.y=0;
		rect.height=frameHeight1;
		rect.width=frameWidth1;
		//use ROI to implement the video split joint
		cvSetImageROI(newFrame, rect);
		cvCopyImage(frame1, newFrame);
		cvResetImageROI(newFrame);
		rect.x = frameWidth1;
		rect.y=0;
		rect.height = frameHeight1;
		rect.width = frameWidth1;
		cvSetImageROI(newFrame, rect);
		cvCopyImage(frame2, newFrame);
		cvResetImageROI(newFrame);
		cvWriteFrame(pWriter, newFrame);
	}
	cvReleaseImage(&frame1);
	cvReleaseImage(&frame2);
	cvReleaseImage(&newFrame);
	cvReleaseVideoWriter(&pWriter)
	return 0;}