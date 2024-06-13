## 3D Face Reconstruction from Stereo Images

The repository contains the source codes for generating a binary to reconstruct 3d points on face from stereo images.

- OS & build tools : Windows 10, Visual Studio 2017
- Dependencies : [our custom calibration library](https://bitbucket.org/mgfacescan/calibr/src/master/), 
  [OpenCV 4.1.0](https://github.com/opencv/opencv/tree/4.1.0)


---

## How to perform a test

1. Move to the "test" folder
2. Execute the binary with the input file included : e.g. face-reconst.exe input.txt
3. After execution, you have point cloud, depthmap and disparity files.
