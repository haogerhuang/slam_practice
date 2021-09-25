#include <iostream>
#include <opencv2/opencv.hpp>

#include <fstream>
#include <boost/format.hpp>
#include <sophus/se3.hpp>
//#include <pangolin/pangolin.h>

#include <pcl/common/angles.h> // for pcl::deg2rad
#include <pcl/features/normal_3d.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/console/parse.h>

using namespace std;
typedef vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>> TrajectoryType;
typedef Eigen::Matrix<double, 6, 1> Vector6d;

pcl::visualization::PCLVisualizer::Ptr rgbVis (pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud)
{
  // --------------------------------------------
  // -----Open 3D viewer and add point cloud-----
  // --------------------------------------------
  pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
  viewer->setBackgroundColor (0, 0, 0);
  pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);
  viewer->addPointCloud<pcl::PointXYZRGB> (cloud, rgb, "sample cloud");
  viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud");
  viewer->addCoordinateSystem (1.0);
  viewer->initCameraParameters ();
  return (viewer);
}

int main(int argc, char *argv[])
{
	vector<cv::Mat> colorImgs, depthImgs;
	TrajectoryType poses;

	ifstream fin("./pose.txt");

	cv::Mat img;
	for(int i = 0; i < 5; i++){
		//boost::format fmt("./%s/%d.%s");
		string color_name = "color/" + to_string(i+1) + ".png";
		string depth_name = "depth/" + to_string(i+1) + ".pgm";
		colorImgs.push_back(cv::imread(color_name));
		depthImgs.push_back(cv::imread(depth_name, -1));

		double data[7] = {0};
		for(auto &d:data) fin >> d;

        Sophus::SE3d pose(Eigen::Quaterniond(data[6], data[3], data[4], data[5]),
                          Eigen::Vector3d(data[0], data[1], data[2]));
        poses.push_back(pose);
	}

	double cx = 325.5;
	double cy = 253.5;
    double fx = 518.0;
    double fy = 519.0;
	double depthScale = 1000.0;
    //Sophus::SE3d pose(Eigen::Quaterniond(0,0,0,0), Eigen::Vector3d(0,0,0));

	
    vector<Vector6d, Eigen::aligned_allocator<Vector6d>> pointcloud;
    pointcloud.reserve(1000000);
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud_ptr (new pcl::PointCloud<pcl::PointXYZRGB>);

    for (int i = 0; i < 5; i++) {
        cout << "transforming" << i + 1 << endl;
        cv::Mat color = colorImgs[i];
        cv::Mat depth = depthImgs[i];
        Sophus::SE3d T = poses[i];
        for (int v = 0; v < color.rows; v++)
            for (int u = 0; u < color.cols; u++) {
                unsigned int d = depth.ptr<unsigned short>(v)[u]; 
                if (d == 0) continue;
                Eigen::Vector3d point;
                point[2] = double(d) / depthScale;
                point[0] = (u - cx) * point[2] / fx;
                point[1] = (v - cy) * point[2] / fy;
                Eigen::Vector3d pointWorld = T * point;

                Vector6d p;
                p.head<3>() = pointWorld;
                p[5] = color.data[v * color.step + u * color.channels()];   // blue
                p[4] = color.data[v * color.step + u * color.channels() + 1]; // green
                p[3] = color.data[v * color.step + u * color.channels() + 2]; // red
                pointcloud.push_back(p);
				pcl::PointXYZRGB pnt(p[0],p[1],p[2],p[3],p[4],p[5]);
				point_cloud_ptr->points.push_back(pnt);
			
            }
    }

    cout << "point cloud numbers:" << pointcloud.size() << endl;

	//pcl::visualization::PCLVisualizer::Ptr viewer = rgbVis(point_cloud_ptr);
	pcl::visualization::CloudViewer viewer("Cloud Viewer");
	viewer.showCloud (point_cloud_ptr);
	while (!viewer.wasStopped())
  	{
  	}

	return 0;

}
