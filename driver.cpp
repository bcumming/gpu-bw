#include <iostream>
#include <limits>
#include <vector>

#include <cuda_stream.hpp>
#include <cuda_event.hpp>

enum class cmode { d2d, h2d, d2h };

int num_devices();
void set_device(int);

void* malloc_device_bytes(size_t n);
void* malloc_pinned_bytes(size_t n);
void* malloc_bytes(size_t n);
void enable_peer_access(int peer);
void enable_peer_access();

template <typename T>
T* malloc_device(size_t n) {
    return static_cast<T*>(malloc_device_bytes(n*sizeof(T)));
}

template <typename T>
T* malloc_host(size_t n, T value=T(), bool use_pinned=true) {
    auto ptr = use_pinned?
        static_cast<T*>(malloc_pinned_bytes(n*sizeof(T))):
        static_cast<T*>(malloc_bytes(n*sizeof(T)));
    std::fill(ptr, ptr+n, value);
    return ptr;
}

template <typename T>
void copy(T* src, T* dst, size_t n, cmode mode, cuda_stream& stream) {
    copy_bytes(src, dst, n*sizeof(T), mode, stream);
}

int main(int argc, const char** argv) {
    // size of buffers
    const size_t min_n = 1000;        // 1 kB
    const size_t max_n = min_n<<22;
    const bool use_pinned = true;
    const cmode mode = cmode::d2d;

    int ngpu = num_devices();
    if (ngpu==0) {
        std::cerr << "error -- no gpus available" << std::endl;
        exit(1);
    }

    std::cout << "\n";
    enable_peer_access();
    std::cout << "\n";

    std::vector<char*> srcbuf(ngpu);
    std::vector<char*> dstbuf(ngpu);

    for (int i=0; i<ngpu; ++i) {
        set_device(i);
        srcbuf[i] = mode==cmode::h2d?
            malloc_host<char>(max_n, '\0', use_pinned):
            malloc_device<char>(max_n);
        dstbuf[i] = mode==cmode::d2h?
            malloc_host<char>(max_n, '\0', use_pinned):
            malloc_device<char>(max_n);
    }

    // create cuda stream for timing of memory copies.
    set_device(0);
    cuda_stream stream;

    // Do a warm up run
    for (auto dst=0; dst<ngpu; dst++) {
        copy<char>(srcbuf[0], dstbuf[dst], max_n, mode,  stream);
    }
    stream.enqueue_event().wait();

    std::printf("\n----------------------------------------------------------------------\n");
    std::printf("%16s ", "msg size (kB)");
    for (auto i=0; i<ngpu; ++i) {
        std::printf("      gpu %d", i);
    }
    std::printf("\n----------------------------------------------------------------------\n");
    for (auto n=min_n; n<=max_n; n*=2) {
        const unsigned kb = n/1000;
        const double gb = n*1.e-9;
        std::printf( "%16u ", kb);

        for (auto dst=0; dst<ngpu; dst++) {
            double time = std::numeric_limits<double>::max();
            // run test multiple times and take the best
            for (int i=0; i<8; ++i) {
                auto start = stream.enqueue_event();
                copy<char>(srcbuf[0], dstbuf[dst], n, mode,  stream);
                auto stop = stream.enqueue_event();
                stop.wait();
                time = std::min(stop.time_since(start), time);
            }

            float bw = gb/time;
            std::printf(" %10.2f", bw);
        }

        std::printf("\n");
    }
    std::printf("----------------------------------------------------------------------\n");
}

void check_api_call(cudaError_t status) {
    if(status != cudaSuccess) {
        std::cerr << "error: CUDA API call : "
                  << cudaGetErrorString(status) << std::endl;
        exit(1);
    }
}

cudaMemcpyKind copy_mode(cmode m) {
    switch(m) {
        case cmode::d2d: return cudaMemcpyDeviceToDevice;
        case cmode::h2d: return cudaMemcpyHostToDevice;
        case cmode::d2h: return cudaMemcpyDeviceToHost;
    }
    std::cerr << "error: invalid copy direction" << std::endl;
    exit(1);
}

void copy_bytes(void* src, void* dst, size_t n, cmode mode, cuda_stream& stream) {
    check_api_call(cudaMemcpyAsync(dst, src, n, copy_mode(mode), stream.stream()));
}

int num_devices() {
    int n;
    check_api_call(cudaGetDeviceCount(&n));
    return n;
}

void set_device(int i) {
    check_api_call(cudaSetDevice(i));
}

void* malloc_device_bytes(size_t n) {
    void* ptr = nullptr;
    check_api_call(cudaMalloc(&ptr, n));
    return ptr;
}

void* malloc_pinned_bytes(size_t n) {
    void* ptr;
    check_api_call(cudaHostAlloc(&ptr, n, 0));
    return ptr;
}

void* malloc_bytes(size_t n) {
    auto ptr = malloc(n);
    if (!ptr) {
        std::cerr << "error: unable to allocate " << n << " bytes on host" << std::endl;
        exit(1);
    }
    return ptr;
}

void enable_peer_access() {
    int ngpu = num_devices();
    // first check that peer-peer access is possible for each device
    std::cout << ngpu << " gpus with peer-connectivity matrix:\n\n";
    std::cout << "  ";
    for (int i=0; i<ngpu; ++i) {
        std::cout << " " << i;
    }
    std::cout << "\n";
    bool good = true;
    for (int device=0; device<ngpu; ++device) {
        std::cout << " " << device;
        for (int peer=0; peer<ngpu; ++peer) {
            int can;
            check_api_call(cudaDeviceCanAccessPeer(&can, device, peer));
            std::cout << " " << (can? '*': '.');
            if (peer!=device && can==0) good = false;
        }
        std::cout << "\n";
    }
    if (!good) {
        std::cerr<< "Error: peer-peer gpu access is not enabled for all GPUs" << std::endl;
        exit(1);
    }

    for (int device=0; device<ngpu; ++device) {
        set_device(device);
        for (int peer=0; peer<ngpu; ++peer) {
            if (peer!=device) enable_peer_access(peer);
        }
    }
}

void enable_peer_access(int peer) {
    check_api_call(cudaDeviceEnablePeerAccess(peer, 0));
}
