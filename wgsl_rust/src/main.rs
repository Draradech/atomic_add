use std::time::Instant;
use std::{num::NonZeroU64, sync::mpsc};

const WORKGROUP_SIZE: u64 = 256;
const MIN_BUFFER_EXPONENT: u32 = 18; // 256 KiB
const MAX_BUFFER_EXPONENT: u32 = 26; // 64 MiB
const STEPS_PER_OCTAVE: u32 = 10;

const SHADER: &str = r#"
struct Params {
    adds_per_thread: u32,
    slot_count: u32,
    thread_count: u32,
    seed: u32,
}

@group(0) @binding(0) var<storage, read_write> counters: array<atomic<u32>>;
@group(0) @binding(1) var<uniform> params: Params;
@group(0) @binding(2) var<storage, read_write> checksums: array<u32>;

fn hash(input: u32) -> u32 {
    var value = input;
    value ^= value >> 16u;
    value *= 0x7feb352du;
    value ^= value >> 15u;
    value *= 0x846ca68bu;
    value ^= value >> 16u;
    return value;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    var checksum = 0u;
    for (var i = 0u; i < params.adds_per_thread; i++) {
        let key = id.x + i * params.thread_count + params.seed;
        let index = hash(key) % params.slot_count;
        checksum ^= atomicAdd(&counters[index], 1u);
    }
    checksums[id.x] = checksum;
}
"#;

#[derive(Clone, Copy)]
enum Backend {
    Vulkan,
    Dx12,
}

impl Backend {
    fn parse(value: &str) -> Result<Self, String> {
        match value.to_ascii_lowercase().as_str() {
            "vulkan" => Ok(Self::Vulkan),
            "dx12" => Ok(Self::Dx12),
            _ => Err(format!(
                "invalid backend: {value} (expected vulkan or dx12)"
            )),
        }
    }

    fn as_wgpu(self) -> wgpu::Backends {
        match self {
            Self::Vulkan => wgpu::Backends::VULKAN,
            Self::Dx12 => wgpu::Backends::DX12,
        }
    }

    fn api(self) -> &'static str {
        match self {
            Self::Vulkan => "Vulkan",
            Self::Dx12 => "Direct3D",
        }
    }

    fn api_version(self, adapter: &wgpu::Adapter) -> String {
        match self {
            Self::Vulkan => {
                // SAFETY: The adapter was requested from a Vulkan-only instance, and the
                // returned guard is used only to read immutable physical-device properties.
                let version = unsafe { adapter.as_hal::<wgpu::hal::api::Vulkan>() }
                    .map(|adapter| {
                        adapter
                            .physical_device_capabilities()
                            .properties()
                            .api_version
                    })
                    .unwrap_or_default();
                if version == 0 {
                    "unknown".into()
                } else {
                    format!(
                        "{}.{}.{}",
                        (version >> 22) & 0x7f,
                        (version >> 12) & 0x3ff,
                        version & 0xfff
                    )
                }
            }
            Self::Dx12 => "12".into(),
        }
    }
}

fn os_name() -> &'static str {
    match std::env::consts::OS {
        "linux" => "Linux",
        "windows" => "Windows",
        "macos" => "macOS",
        other => other,
    }
}

fn csv_field(value: &str) -> String {
    if value.contains([',', '"', '\n', '\r']) {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.to_owned()
    }
}

#[derive(Clone, Copy)]
struct Config {
    backend: Backend,
    workgroups: u32,
    adds_per_thread: u32,
    samples: u32,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            backend: Backend::Vulkan,
            workgroups: 4096,
            adds_per_thread: 1024,
            samples: 5,
        }
    }
}

fn step_relation(step: u32) -> &'static str {
    match step {
        0 => "power_of_two",
        1 => "step_1_of_10",
        2 => "step_2_of_10",
        3 => "step_3_of_10",
        4 => "step_4_of_10",
        5 => "step_5_of_10",
        6 => "step_6_of_10",
        7 => "step_7_of_10",
        8 => "step_8_of_10",
        9 => "step_9_of_10",
        _ => unreachable!("step must be within one octave"),
    }
}

fn make_cases() -> Vec<(u64, &'static str, u32)> {
    let mut cases = Vec::with_capacity(
        ((MAX_BUFFER_EXPONENT - MIN_BUFFER_EXPONENT) * STEPS_PER_OCTAVE + 1) as usize,
    );
    for exponent in MIN_BUFFER_EXPONENT..MAX_BUFFER_EXPONENT {
        for step in 0..STEPS_PER_OCTAVE {
            let log2_bytes = exponent as f64 + step as f64 / STEPS_PER_OCTAVE as f64;
            let slots = (2f64.powf(log2_bytes) / size_of::<u32>() as f64).round() as u64;
            cases.push((
                slots * size_of::<u32>() as u64,
                step_relation(step),
                exponent,
            ));
        }
    }
    cases.push((
        1u64 << MAX_BUFFER_EXPONENT,
        "power_of_two",
        MAX_BUFFER_EXPONENT,
    ));
    cases
}

fn usage(program: &str) {
    eprintln!(
        "Usage: {program} [--backend vulkan|dx12] [--workgroups N] [--adds N] [--samples N]\n\
         Defaults: --backend vulkan --workgroups 4096 --adds 1024 --samples 5"
    );
}

fn parse_args() -> Result<Config, String> {
    let mut config = Config::default();
    let mut args = std::env::args();
    let program = args
        .next()
        .unwrap_or_else(|| "wgsl-atomic-add-bench".into());
    while let Some(flag) = args.next() {
        if flag == "-h" || flag == "--help" {
            usage(&program);
            std::process::exit(0);
        }
        let value = args
            .next()
            .ok_or_else(|| format!("missing value after {flag}"))?;
        if flag == "--backend" {
            config.backend = Backend::parse(&value)?;
            continue;
        }
        let value: u32 = value
            .parse()
            .map_err(|_| format!("invalid integer for {flag}: {value}"))?;
        if value == 0 {
            return Err(format!("{flag} must be greater than zero"));
        }
        match flag.as_str() {
            "--workgroups" => config.workgroups = value,
            "--adds" => config.adds_per_thread = value,
            "--samples" => config.samples = value,
            _ => return Err(format!("unknown option: {flag}")),
        }
    }
    Ok(config)
}

fn main() {
    let program = std::env::args().next().unwrap_or_default();
    let config = parse_args().unwrap_or_else(|error| {
        eprintln!("error: {error}");
        usage(&program);
        std::process::exit(2);
    });

    if let Err(error) = pollster::block_on(run(config)) {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

async fn run(config: Config) -> Result<(), Box<dyn std::error::Error>> {
    let mut instance_descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
    instance_descriptor.backends = config.backend.as_wgpu();
    let instance = wgpu::Instance::new(instance_descriptor);
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        })
        .await?;
    let info = adapter.get_info();
    let api_version = config.backend.api_version(&adapter);
    let driver_version = if info.driver_info.trim().is_empty() {
        info.driver.as_str()
    } else {
        info.driver_info.as_str()
    };
    let adapter_limits = adapter.limits();
    let required_limits = wgpu::Limits {
        max_buffer_size: adapter_limits.max_buffer_size,
        max_storage_buffer_binding_size: adapter_limits.max_storage_buffer_binding_size,
        ..Default::default()
    };
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("atomic-add benchmark device"),
            required_features: wgpu::Features::empty(),
            required_limits,
            ..Default::default()
        })
        .await?;

    let limits = device.limits();
    if config.workgroups > limits.max_compute_workgroups_per_dimension {
        return Err(format!(
            "--workgroups {} exceeds this device's limit ({})",
            config.workgroups, limits.max_compute_workgroups_per_dimension
        )
        .into());
    }

    let cases = make_cases();
    let max_counter_size = cases.iter().map(|case| case.0).max().unwrap();
    let checksum_size = config.workgroups as u64 * WORKGROUP_SIZE * 4;
    for (name, size) in [("atomic", max_counter_size), ("checksum", checksum_size)] {
        if size > limits.max_storage_buffer_binding_size {
            return Err(format!(
                "{name} buffer requires {size} bytes, exceeding this device's storage binding limit ({})",
                limits.max_storage_buffer_binding_size
            )
            .into());
        }
    }

    let counters = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("random-access atomic counters"),
        size: max_counter_size,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let params = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("parameters"),
        size: 16,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let thread_count = config.workgroups * WORKGROUP_SIZE as u32;
    let checksums = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("per-thread checksums"),
        size: checksum_size,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("atomic buffer readback"),
        size: max_counter_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("atomic add WGSL"),
        source: wgpu::ShaderSource::Wgsl(SHADER.into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("atomic add pipeline"),
        layout: None,
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let make_bind_group = |buffer_size| {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("benchmark bindings"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &counters,
                        offset: 0,
                        size: NonZeroU64::new(buffer_size),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: checksums.as_entire_binding(),
                },
            ],
        })
    };
    let encode_clear = |buffer_size| {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("clear atomic buffer"),
        });
        encoder.clear_buffer(&counters, 0, Some(buffer_size));
        encoder.finish()
    };
    let encode_dispatch = |bind_group: &wgpu::BindGroup| {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("random atomic add commands"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(config.workgroups, 1, 1);
        }
        encoder.finish()
    };

    let write_params = |slot_count| {
        let mut bytes = [0u8; 16];
        for (chunk, value) in bytes.chunks_exact_mut(4).zip([
            config.adds_per_thread,
            slot_count,
            thread_count,
            0x1234_5678,
        ]) {
            chunk.copy_from_slice(&value.to_le_bytes());
        }
        queue.write_buffer(&params, 0, &bytes);
    };

    let total = config.workgroups as u64 * WORKGROUP_SIZE * config.adds_per_thread as u64;
    let minimum_slots = cases.iter().map(|case| case.0 / 4).min().unwrap();
    if total > minimum_slots * u32::MAX as u64 {
        return Err("workload can overflow individual counters at the smallest buffer size".into());
    }

    eprintln!("GPU: {} ({:?})", info.name, info.backend);
    eprintln!(
        "OS/API: {} / {} {}",
        os_name(),
        config.backend.api(),
        api_version
    );
    eprintln!("Driver: {} {}", info.driver, info.driver_info);
    eprintln!(
        "Sweep: {} cases, {} samples each, 256 KiB through 64 MiB (ten log2 steps per octave), {} atomic adds per sample",
        cases.len(),
        config.samples,
        total
    );

    // Force lazy pipeline compilation and warm up before collecting the sweep.
    let warmup_slots = (max_counter_size / 4) as u32;
    write_params(warmup_slots);
    let warmup_bind_group = make_bind_group(max_counter_size);
    queue.submit([
        encode_clear(max_counter_size),
        encode_dispatch(&warmup_bind_group),
    ]);
    device.poll(wgpu::PollType::wait_indefinitely())?;

    let mut results = Vec::with_capacity(cases.len());
    for (index, &(buffer_size, relation, exponent)) in cases.iter().enumerate() {
        let slot_count = (buffer_size / 4) as u32;
        eprintln!(
            "[{}/{}] 2^{} B {}: {} B ({:.6} MiB, {} u32 slots)",
            index + 1,
            cases.len(),
            exponent,
            relation,
            buffer_size,
            buffer_size as f64 / (1024.0 * 1024.0),
            slot_count
        );

        write_params(slot_count);
        let bind_group = make_bind_group(buffer_size);
        let mut throughputs = Vec::with_capacity(config.samples as usize);
        for sample in 1..=config.samples {
            // Clearing is deliberately outside the timed region.
            queue.submit([encode_clear(buffer_size)]);
            device.poll(wgpu::PollType::wait_indefinitely())?;
            let commands = encode_dispatch(&bind_group);
            let start = Instant::now();
            queue.submit([commands]);
            device.poll(wgpu::PollType::wait_indefinitely())?;
            let seconds = start.elapsed().as_secs_f64();
            let throughput = total as f64 / seconds / 1e9;
            eprintln!(
                "         sample {}/{}: {:.3} ms, {:.3} Gatomic/s",
                sample,
                config.samples,
                seconds * 1e3,
                throughput
            );
            throughputs.push(throughput);
        }

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("copy atomic buffer for validation"),
        });
        encoder.copy_buffer_to_buffer(&counters, 0, &readback, 0, buffer_size);
        queue.submit([encoder.finish()]);
        device.poll(wgpu::PollType::wait_indefinitely())?;

        let (sender, receiver) = mpsc::channel();
        readback.map_async(wgpu::MapMode::Read, 0..buffer_size, move |result| {
            let _ = sender.send(result);
        });
        device.poll(wgpu::PollType::wait_indefinitely())?;
        receiver.recv()??;
        let mapped = readback.get_mapped_range(0..buffer_size)?;
        let actual: u64 = mapped
            .chunks_exact(4)
            .map(|bytes| u32::from_le_bytes(bytes.try_into().unwrap()) as u64)
            .sum();
        drop(mapped);
        readback.unmap();
        if actual != total {
            return Err(format!(
                "validation failed for {buffer_size} bytes: got sum {actual}, expected {total}"
            )
            .into());
        }

        throughputs.sort_by(f64::total_cmp);
        let average = throughputs.iter().sum::<f64>() / throughputs.len() as f64;
        let median = throughputs[throughputs.len() / 2];
        eprintln!(
            "         average {:.3}, median {:.3} Gatomic/s; validation passed",
            average, median
        );
        results.push((buffer_size, relation, exponent, average, median));
    }

    eprintln!("Sweep complete; CSV follows on stdout.");
    println!(
        "buffer_bytes,buffer_mib,relation,power_of_two_exponent,average_gatomic_per_s,median_gatomic_per_s,os,api,api_version,driver_version"
    );
    for (buffer_size, relation, exponent, average, median) in results {
        println!(
            "{},{:.9},{},{},{:.6},{:.6},{},{},{},{}",
            buffer_size,
            buffer_size as f64 / (1024.0 * 1024.0),
            relation,
            exponent,
            average,
            median,
            csv_field(os_name()),
            csv_field(config.backend.api()),
            csv_field(&api_version),
            csv_field(driver_version)
        );
    }
    Ok(())
}
