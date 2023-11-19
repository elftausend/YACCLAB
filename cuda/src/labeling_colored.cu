
#include <opencv2/cudafeatures2d.hpp>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "labeling_algorithms.h"
#include "register.h"

#define BLOCK_SIZE 32   // this must be multiple of the warp size (leave it to 32)
#define PATCH_SIZE (BLOCK_SIZE + 2)

using namespace cv;
using namespace std;

namespace {
__global__ void Init(const cuda::PtrStepSzb img, cuda::PtrStepSzi labels) {
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (!(x < img.cols && y < img.rows)) {
        return;
    }

    const unsigned r = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    const unsigned c = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    const unsigned labels_index = r * (labels.step / labels.elem_size) + c;
    
    labels[labels_index] = labels_index+1;
}

// labels do not contain connection information (rasmusson)
__global__ void Propagate(cuda::PtrStepSzb img, cuda::PtrStepSzi globalLabels, int* changed, unsigned char offsetY, unsigned char offsetX) {
    int bloatedBlockIdxX = blockIdx.x * 2 + offsetX;
    int bloatedBlockIdxY = blockIdx.y * 2 + offsetY;

    bool thread_changed = false;

    int x = bloatedBlockIdxX * blockDim.x + threadIdx.x;
    int y = bloatedBlockIdxY * blockDim.y + threadIdx.y;

    if (x >= img.cols) {
        return;
    }

    if (y >= img.rows) {
        return;
    }

    __shared__ unsigned char pixels[34][34];
    __shared__ int labels[34][34];
    
    __shared__ bool something_changed[1];

    int pixelIdx = y * img.cols + x;

    if (threadIdx.y == 0) {
        int upperOverlap = (bloatedBlockIdxY * blockDim.y -1);
        int upperOverlapIdx = upperOverlap * img.cols + x;
        if (upperOverlap >= 0) {

            labels[0][threadIdx.x+1] = globalLabels[upperOverlapIdx];
            pixels[0][threadIdx.x+1] = img[upperOverlapIdx];
            
            //maybe not needed, diagonals
            if (threadIdx.x == 0) {
                labels[0][0] = globalLabels[upperOverlapIdx - 1];
                pixels[0][0] = img[upperOverlapIdx - 1];
            }
            if (threadIdx.x == 31) {
                labels[0][33] = globalLabels[upperOverlapIdx + 1];
                pixels[0][0] = img[upperOverlapIdx + 1];

            }
        }
    }

    if (threadIdx.y == 31) {
        int lowerOverlap =  (bloatedBlockIdxY * blockDim.y + 32);
        int lowerOverlapIdx = lowerOverlap * img.cols + x;
        if (lowerOverlap < img.rows) {                
            labels[33][threadIdx.x+1] = globalLabels[lowerOverlapIdx];
            pixels[33][threadIdx.x+1] = img[lowerOverlapIdx];

            //maybe not needed, diagonals 8way?
            if (threadIdx.x == 0) {
                labels[33][0] = globalLabels[lowerOverlapIdx - 1];
                pixels[33][0] = img[lowerOverlapIdx - 1];
            }
            if (threadIdx.x == 31) {
                labels[33][33] = globalLabels[lowerOverlapIdx + 1];
                pixels[33][33] = img[lowerOverlapIdx + 1];
            }
        }
    }

    if (threadIdx.x == 0) {
        int leftOverlap =  (bloatedBlockIdxX * blockDim.x -1);
        int leftOverlapIdx = y * img.cols + leftOverlap;
        if (leftOverlap >= 0) {
            labels[threadIdx.y+1][0] = globalLabels[leftOverlapIdx];
            pixels[threadIdx.y+1][0] = img[leftOverlapIdx];
        }
    }

    if (threadIdx.x == 31) {
        int rightOverlap =  (bloatedBlockIdxX * blockDim.x + 32);
        int rightOverlapIdx = y * img.cols + rightOverlap;
        if (rightOverlap < img.cols) {
            labels[threadIdx.y+1][33] = globalLabels[rightOverlapIdx];
            pixels[threadIdx.y+1][33] = img[rightOverlapIdx];
        }
    }

    pixels[threadIdx.y+1][threadIdx.x+1] = img[pixelIdx];
    labels[threadIdx.y+1][threadIdx.x+1] = globalLabels[y * img.cols + x];
    __syncthreads();

    int threshold = 10;
    
    {
    int currentLabel = labels[threadIdx.y+1][32-threadIdx.x];
    int currentPixel = (int) pixels[threadIdx.y+1][32-threadIdx.x];
    int pixel = (int) pixels[threadIdx.y+1][33-threadIdx.x];
    int label = labels[threadIdx.y+1][33-threadIdx.x];
    if (abs(pixel - currentPixel) < threshold) {
        if (currentLabel < label) {
            labels[threadIdx.y+1][32-threadIdx.x] = label;
            // printf("thread changed");
            // *changed = 1;
            atomicOr(changed, 1);
            // thread_changed = true;  
        }
    }
    }
    
    // if (thread_changed) {
        // something_changed[0] = true;
    // }

    __syncthreads();

    {
        int currentLabel = labels[32-threadIdx.y][threadIdx.x+1];
        int currentPixel = (int) pixels[32-threadIdx.y][threadIdx.x+1];

        int pixel = (int) pixels[33-threadIdx.y][threadIdx.x+1];
        int label = labels[33-threadIdx.y][threadIdx.x+1];
        if (abs(pixel - currentPixel) < threshold) {
            if (currentLabel < label) {
                labels[32-threadIdx.y][threadIdx.x+1] = label;
                atomicOr(changed, 1);
            }
        }
    }
    __syncthreads();

    //left
    {
        int currentLabel = labels[threadIdx.y+1][threadIdx.x+1];
        int currentPixel = (int) pixels[threadIdx.y+1][threadIdx.x+1];
        int pixel = (int) pixels[threadIdx.y+1][threadIdx.x];
        int label = labels[threadIdx.y+1][threadIdx.x];

        if (abs(pixel - currentPixel) < threshold) {
            if (currentLabel < label) {
                labels[threadIdx.y+1][threadIdx.x+1] = label;
                atomicOr(changed, 1);
            }
        }
    }
    __syncthreads();

    // up
     {
        int currentLabel = labels[threadIdx.y+1][threadIdx.x+1];
        int currentPixel = (int) pixels[threadIdx.y+1][threadIdx.x+1];

        int pixel = (int) pixels[threadIdx.y][threadIdx.x+1];
        int label = labels[threadIdx.y][threadIdx.x+1];
        if (abs(pixel - currentPixel) < threshold) {
            if (currentLabel < label) {
                labels[threadIdx.y+1][threadIdx.x+1] = label;
                atomicOr(changed, 1);
            }
        }
    }
    __syncthreads();



    // if (threadIdx.x == 0 && threadIdx.y == 0 && something_changed[0]) {
        // *changed = 1;
    // }

    globalLabels[y * img.cols + x] = labels[threadIdx.y + 1][threadIdx.x + 1];
}

__global__ void End(cuda::PtrStepSzi labels) {

    unsigned global_row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    unsigned global_col = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    unsigned labels_index = global_row * (labels.step / labels.elem_size) + global_col;

    if (global_row < labels.rows && global_col < labels.cols) {
        labels.data[labels_index] &= 0x0FFFFFFF;
    }
}
}


class COLORED: public GpuLabeling2D<Connectivity2D::CONN_8> {
private:
    dim3 grid_size_;
    dim3 block_size_;
    char* d_changed_ptr_;
public:
    COLORED() {}

    void PerformLabeling() {
        // printf("start");
        d_img_labels_.create(d_img_.size(), CV_32SC1);
        grid_size_ = dim3((d_img_.cols + BLOCK_SIZE - 1) / BLOCK_SIZE, (d_img_.rows + BLOCK_SIZE - 1) / BLOCK_SIZE, 1);
        block_size_ = dim3(BLOCK_SIZE, BLOCK_SIZE, 1);
 
        Init << <grid_size_, block_size_ >> > (d_img_, d_img_labels_);

        char yOffsets[] = {0, 0, 1, 1};
        char xOffsets[] = {0, 1, 0, 1};

        char changed = 1;
        int* d_changed_ptr;
        cudaMalloc(&d_changed_ptr, 1);

        while (changed) {     
            changed = 0;
            // changed += 1;
            cudaMemset(d_changed_ptr, 0, 1);

            for (int i = 0; i < 4; i++) {
                Propagate << <grid_size_, block_size_ >> > (d_img_, d_img_labels_, d_changed_ptr, yOffsets[i], xOffsets[i]);
                cudaDeviceSynchronize();
            }

            cudaMemcpy(&changed, d_changed_ptr, 1, cudaMemcpyDeviceToHost);
        }
        // printf("finished");
        End << <grid_size_, block_size_ >> > (d_img_labels_);
        
        cudaFree(d_changed_ptr);
        cudaDeviceSynchronize();
    }
};

REGISTER_LABELING(COLORED);
