#include"lbp.h"




int main()
{
	Mat image1 = imread("lena.jpg", 0);
	Mat image2 = imread("2.jpg", 0);

	Mat lbpImage1;
	getUniformPatternLBPFeature(image1, lbpImage1, 3, 8);
	imshow("lena", lbpImage1);
	
	Mat lbpImage2;
	getUniformPatternLBPFeature(image2, lbpImage2, 3, 8);
	imshow("2", lbpImage2);

	Mat hist1 = spatial_histogram(lbpImage1,59,8,8, true);
	Mat hist2 = spatial_histogram(lbpImage2, 59, 8, 8, true);

	double dis = chiSquare(hist1, hist2);
	cout << dis << endl;
	

	waitKey(0);
	return 0;
}
