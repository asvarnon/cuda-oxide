#![allow(clippy::needless_range_loop)]

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{DisjointSlice, SharedArray, kernel, thread};
use cuda_host::cuda_module;

#[cuda_module]
mod kernals {
    use super::*;

    #[kernel]
    pub fn reduction(data: &[f32], mut out: DisjointSlice<f32>) {
        // allocate a tile for shared memory
        static mut TILE: SharedArray<f32, 1024> = SharedArray::UNINIT;
        const TILE_SIZE: usize = 1024;

        // get threads for that block
        let tid = thread::threadIdx_x() as usize;
        let idx = thread::index_1d();
        let gid = idx.get();

        // copy data to shared memory
        unsafe {
            TILE[tid] = data[gid];
        }

        thread::sync_threads();

        // read data from neighbor:
        let mut multiple = 1;

        while multiple <= TILE_SIZE / 2 {
            if tid % (multiple * 2) == 0 {
                let neighbor_id = tid + multiple;
                unsafe { TILE[tid] += TILE[neighbor_id] };
            }
            thread::sync_threads();
            multiple *= 2;
        }

        // write result to global memory
        if tid == 0
            && let Some(out_elem) = out.get_mut(idx)
        {
            *out_elem = unsafe { TILE[0] };
        }
    }
}

pub fn main() {
    println!("=== Reduction Example ===\n");

    //initiallize cuda
    let ctx = CudaContext::new(0).unwrap();
    let stream = ctx.default_stream();

    //test Data
    const N: usize = 1024;
    let data_host: Vec<f32> = (0..N).map(|i| i as f32).collect();

    // allocate Device memory
    let data_dev = DeviceBuffer::from_host(&stream, &data_host).unwrap();
    //only 1 result, so we only need 1 element
    let mut out: DeviceBuffer<f32> = DeviceBuffer::<f32>::zeroed(&stream, 1).unwrap();

    // Launch config for shared memory kernels.
    // for_num_elems is a convenience function that divides your N across multiple blocks. It doesn't guarantee one block.
    // need custom config for shared memory kernels and need to ensure on one block.
    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (1024, 1, 1),
        shared_mem_bytes: 0,
    };

    // Load the embedded PTX bundle and launch through the typed module API.
    println!("Load the embedded PTX..");
    let module = kernals::load(&ctx).expect("Failed to load embedded CUDA module");
    module
        .reduction(&stream, cfg, &data_dev, &mut out)
        .expect("Kernal Launch failed..");

    // copy results from VRAM to host.
    let out_host = out.to_host_vec(&stream).unwrap();

    println!("Result: {:?}", out_host);
}
