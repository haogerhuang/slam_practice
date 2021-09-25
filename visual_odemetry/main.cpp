#include <iostream>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <sophus/se3.hpp>

using namespace std;
//using namespace cv;


void find_feature_matches(
	const cv::Mat &img_1, const cv::Mat &img_2,
	std::vector<cv::KeyPoint> &keypoints_1,
	std::vector<cv::KeyPoint> &keypoints_2,
	std::vector<cv::DMatch> &matches);

cv::Point2d pixel2cam(const cv::Point2d &p, const cv::Mat &K);


typedef vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> VecVector2d;
typedef vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> VecVector3d;

void bundleAdjustmentG2O(
	const VecVector3d &points_3d,
	const VecVector2d &points_2d,
	const cv::Mat &K,
	Sophus::SE3d &pose
);

// BA by gauss-newton
void BAGaussNewton(
	const VecVector3d &points_3d,
	const VecVector2d &points_2d,
	const cv::Mat &K,
	Sophus::SE3d &pose
);
int main(int argc, char *argv[]){

	cv::Mat img1 = cv::imread("1.png", cv::IMREAD_COLOR);
	cv::Mat img2 = cv::imread("2.png", cv::IMREAD_COLOR);
	cv::Mat d1 = cv::imread("1_depth.png", cv::IMREAD_UNCHANGED); 

  	vector<cv::KeyPoint> keypoints1, keypoints2;
  	vector<cv::DMatch> matches;
 	find_feature_matches(img1, img2, keypoints1, keypoints2, matches);
  	cout << "Found " << matches.size() << " matchs" << endl;


  	cv::Mat K = (cv::Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);


	vector<cv::Point3f> pts_3d;
	vector<cv::Point2f> pts_2d;
	for (cv::DMatch m:matches) {
		ushort d = d1.ptr<unsigned short>(int(keypoints1[m.queryIdx].pt.y))[int(keypoints1[m.queryIdx].pt.x)];
		if (d == 0) continue;
		float dd = d / 5000.0;
		cv::Point2d p1 = pixel2cam(keypoints1[m.queryIdx].pt, K);
		pts_3d.push_back(cv::Point3f(p1.x * dd, p1.y * dd, dd));
		pts_2d.push_back(keypoints2[m.trainIdx].pt);
  	}

	cout << "3d-2d pairs: " << pts_3d.size() << endl;

	VecVector3d pts_3d_eigen;
	VecVector2d pts_2d_eigen;
	for (size_t i = 0; i < pts_3d.size(); ++i) {
	  pts_3d_eigen.push_back(Eigen::Vector3d(pts_3d[i].x, pts_3d[i].y, pts_3d[i].z));
	  pts_2d_eigen.push_back(Eigen::Vector2d(pts_2d[i].x, pts_2d[i].y));
	}

	cout << "calling bundle adjustment by gauss newton" << endl;
	Sophus::SE3d pose_gn;
	BAGaussNewton(pts_3d_eigen, pts_2d_eigen, K, pose_gn);

	return 0;

}


void find_feature_matches(const cv::Mat &img_1, const cv::Mat &img_2,
                          std::vector<cv::KeyPoint> &keypoints_1,
                          std::vector<cv::KeyPoint> &keypoints_2,
                          std::vector<cv::DMatch> &matches) {

	cv::Mat descriptors_1, descriptors_2;
	
	cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create();
	cv::Ptr<cv::DescriptorExtractor> descriptor = cv::ORB::create();
	// use this if you are in OpenCV2
	// Ptr<FeatureDetector> detector = FeatureDetector::create ( "ORB" );
	// Ptr<DescriptorExtractor> descriptor = DescriptorExtractor::create ( "ORB" );
	cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");
	
	
	detector->detect(img_1, keypoints_1);
	detector->detect(img_2, keypoints_2);
	
	descriptor->compute(img_1, keypoints_1, descriptors_1);
	descriptor->compute(img_2, keypoints_2, descriptors_2);
	
	vector<cv::DMatch> match;
	matcher->match(descriptors_1, descriptors_2, match);
	
	double min_dist = 10000, max_dist = 0;
	
	for (int i = 0; i < descriptors_1.rows; i++) {
		double dist = match[i].distance;
		if (dist < min_dist) min_dist = dist;
		if (dist > max_dist) max_dist = dist;
	}
	
	printf("-- Max dist : %f \n", max_dist);
	printf("-- Min dist : %f \n", min_dist);
	
	for (int i = 0; i < descriptors_1.rows; i++) {
		if (match[i].distance <= max(2 * min_dist, 30.0)) {
		  matches.push_back(match[i]);
		}
	}
}


cv::Point2d pixel2cam(const cv::Point2d &p, const cv::Mat &K) {
  return cv::Point2d
    (
      (p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
      (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1)
    );
}

void BAGaussNewton(const VecVector3d &points_3d, 
					const VecVector2d &points_2d, 
					const cv::Mat &K,
					Sophus::SE3d &pose){
	cout<<pose.matrix()<<endl;	
	double fx = K.at<double>(0,0);
	double fy = K.at<double>(1,1);
	double cx = K.at<double>(0,2);
	double cy = K.at<double>(1,2);
	double cost;
	double lastcost = 0;
	for(int iter = 0; iter < 10; iter++){
		cost = 0;
		Eigen::Matrix<double,6,6> H = Eigen::Matrix<double, 6, 6>::Zero();
		Eigen::Matrix<double,6,1> g = Eigen::Matrix<double, 6, 1>::Zero();
		for(int i = 0; i < points_3d.size(); i++){
			Eigen::Vector3d P_ = pose * points_3d[i];
			double x_ = P_[0];
			double y_ = P_[1];
			double z_ = P_[2];
			Eigen::Vector2d reproj = Eigen::Vector2d(fx*x_/z_+cx, fy*y_/z_+cy);
			Eigen::Vector2d e = points_2d[i] - reproj;
			cost += e.squaredNorm();

			Eigen::Matrix<double, 2, 6> Jacobian;
			Jacobian(0,0) = fx/z_;	
			Jacobian(0,2) = -fx*x_/(z_*z_);
			Jacobian(0,3) = -fx*x_*y_/(z_*z_);
			Jacobian(0,4) = fx + fx*x_*x_/(z_*z_);
			Jacobian(0,5) = -fx*y_/z_;
			Jacobian(1,1) = fy/z_;
			Jacobian(1,2) = -fy*y_/(z_*z_);
			Jacobian(1,3) = -fy - fy*y_*y_/(z_*z_);
			Jacobian(1,4) = fy*x_*y_/(z_*z_);
			Jacobian(1,5) = fy*x_/z_;
			Jacobian = -Jacobian;
			H += Jacobian.transpose()*Jacobian;
			
			g += -Jacobian.transpose() * e;
		}
		Eigen::VectorXd dx = H.ldlt().solve(g);
		pose = Sophus::SE3d::exp(dx) * pose;
		lastcost = cost;
		cout<<cost<<endl;	
	}
} 
