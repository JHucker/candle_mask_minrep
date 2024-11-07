use std::collections::HashMap;
use std::time::Instant;
use tch::IndexOp;

fn tch_get_mask(seq_len: i64, device: &tch::Device) -> tch::Tensor {
    tch::Tensor::ones([seq_len, seq_len], (tch::Kind::Uint8, *device))
        .tril(0)
        .view([1, 1, seq_len, seq_len])
        .requires_grad_(false)
        .to_kind(tch::Kind::Bool)
}

fn bench_tch(
    seq_len: i64,
    num_head: i64,
    batch_sizes: &[usize],
    repeats: usize,
) -> Vec<(usize, u128)> {
    let kind = tch::Kind::Float;
    let tch_dev = tch::Device::Cuda(0);

    // warmup
    let warmup_batch_size = 16;
    let mask = tch_get_mask(seq_len, &tch_dev);
    for _ in 0..repeats {
        let input = tch::Tensor::randn(
            [warmup_batch_size, num_head, seq_len, seq_len],
            (kind, tch_dev),
        );
        let _masked = input.masked_fill(
            &mask.i((.., .., ..seq_len, ..seq_len)).eq(0.),
            f64::NEG_INFINITY,
        );
    }
    tch::Cuda::synchronize(0);

    // start recording
    let mut results = Vec::new();
    for batch_size in batch_sizes {
        let mask = tch_get_mask(seq_len, &tch_dev);

        let mut timings = Vec::new();
        for _ in 0..repeats {
            let input = tch::Tensor::randn(
                [
                    i64::try_from(*batch_size).unwrap(),
                    num_head,
                    seq_len,
                    seq_len,
                ],
                (kind, tch_dev),
            );
            tch::Cuda::synchronize(0);

            let tic = Instant::now();
            let _masked = input.masked_fill(
                &mask.i((.., .., ..seq_len, ..seq_len)).eq(0.),
                f64::NEG_INFINITY,
            );
            tch::Cuda::synchronize(0);
            let elapsed = tic.elapsed().as_micros();
            timings.push(elapsed);
        }
        let average = timings.iter().sum::<u128>() / timings.len() as u128;
        results.push((*batch_size, average));
    }

    results
}

// below two functions per: https://github.com/huggingface/candle/blob/e2b6b367fa852ed30ac532f8d77cd8479c7ed092/candle-transformers/src/models/mpt.rs#L278
fn cdl_get_mask(
    size: usize,
    device: &candle_core::Device,
) -> candle_core::Result<candle_core::Tensor> {
    let mask: Vec<_> = (0..size)
        .flat_map(|i| (0..size).map(move |j| u8::from(j > i)))
        .collect();
    candle_core::Tensor::from_slice(&mask, (size, size), device)
}

fn cdl_masked_fill(
    on_false: &candle_core::Tensor,
    mask: &candle_core::Tensor,
    on_true: f32,
) -> candle_core::Result<candle_core::Tensor> {
    let shape = mask.shape();
    let on_true =
        candle_core::Tensor::new(on_true, on_false.device())?.broadcast_as(shape.dims())?;
    let m = mask.where_cond(&on_true, on_false)?;
    Ok(m)
}

fn bench_candle(
    seq_len: usize,
    num_head: usize,
    batch_sizes: &[usize],
    repeats: usize,
) -> Vec<(usize, u128)> {
    let dtype = candle_core::DType::F32;
    let cdl_dev = candle_core::Device::cuda_if_available(0).unwrap();

    let backend_inner: HashMap<_, _> = HashMap::new();
    let cdl_vb = candle_nn::VarBuilder::new_with_args(Box::new(backend_inner), dtype, &cdl_dev);

    // warmup round
    let warmup_batch_size = 16;
    let mask = cdl_get_mask(seq_len, cdl_vb.device()).unwrap();
    for _ in 0..repeats {
        let input = candle_core::Tensor::randn(
            0.0,
            1.0,
            &[warmup_batch_size, num_head, seq_len, seq_len],
            &cdl_dev,
        )
            .unwrap()
            .to_dtype(dtype)
            .unwrap();
        let _masked = cdl_masked_fill(
            &input,
            &mask.broadcast_as(input.shape()).unwrap(),
            f32::NEG_INFINITY,
        )
            .unwrap();
    }
    cdl_dev.synchronize().unwrap();

    // start recording
    let mut results = Vec::new();
    for batch_size in batch_sizes {
        let mask = cdl_get_mask(seq_len, cdl_vb.device()).unwrap();

        let mut timings = Vec::new();
        for _ in 0..repeats {
            let input = candle_core::Tensor::randn(
                0.0,
                1.0,
                &[*batch_size, num_head, seq_len, seq_len],
                &cdl_dev,
            )
                .unwrap()
                .to_dtype(dtype)
                .unwrap();
            cdl_dev.synchronize().unwrap();

            let tic = Instant::now();
            let _masked = cdl_masked_fill(
                &input,
                &mask.broadcast_as(input.shape()).unwrap(),
                f32::NEG_INFINITY,
            )
                .unwrap();
            cdl_dev.synchronize().unwrap();
            let elapsed = tic.elapsed().as_micros();
            timings.push(elapsed);
        }
        let average = timings.iter().sum::<u128>() / timings.len() as u128;
        results.push((*batch_size, average));
    }

    results
}

fn main() {
    let seq_len: usize = 76;
    let num_head: usize = 5;
    let repeats: usize = 50;
    let batch_sizes: Vec<_> = (0..12).map(|x| 2_usize.pow(x)).collect();

    let tch_results = bench_tch(
        i64::try_from(seq_len).unwrap(),
        i64::try_from(num_head).unwrap(),
        &batch_sizes,
        repeats,
    );
    let cdl_results = bench_candle(seq_len, num_head, &batch_sizes, repeats);

    println!("batch_size,tch_μs_average,cdl_μs_average");
    for (tch_res, cdl_res) in tch_results.iter().zip(cdl_results.iter()) {
        let (batch_size, tch_us) = tch_res;
        let (_, cdl_us) = cdl_res;
        println!("{batch_size},{tch_us},{cdl_us}");
    }
}