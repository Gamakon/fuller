//! The device every WGSL host part runs on ([`device::Device`]: the adapter's own limits, the features the kernels
//! use, the installed validation target) and the error scopes that turn a kernel the backend cannot compile into an
//! error instead of a panic ([`device::ErrorScopes`]). The work guard is re-exported here ([`device::Fuel`]).

#[cfg(feature = "gpu")]
pub mod device {
    //! The device and the error scopes every device creation runs in.

    /// The work guard (`work_guard`): `work_bound` is `H * 2^(6K) * invocations`; every submit of a kernel that is
    /// not proven goes through [`Fuel::check`] first.
    pub use crate::wgsl::work_guard::{work_bound, Fuel, CAP_M3_MAX_NODE_EXECUTIONS_PER_SUBMIT};

    /// The error scopes a block of device creation runs in: a kernel the backend cannot compile (Metal rejecting what
    /// naga wrote for a `modf` of a vector bitcast to integers, 2026-10-21) is reported by wgpu as a validation
    /// error or as an internal one, and an error with no scope of its kind is a panic that ends the process. Both
    /// kinds are caught; the first is returned as text, the kernel scored as one that does not compile.
    pub struct ErrorScopes {
        internal: wgpu::ErrorScopeGuard,
        validation: wgpu::ErrorScopeGuard,
    }

    impl ErrorScopes {
        pub fn push(device: &wgpu::Device) -> ErrorScopes {
            let internal = device.push_error_scope(wgpu::ErrorFilter::Internal);
            let validation = device.push_error_scope(wgpu::ErrorFilter::Validation);
            ErrorScopes { internal, validation }
        }

        /// The first error the block raised, if any.
        pub fn pop(self) -> Option<String> {
            let validation = pollster::block_on(self.validation.pop());
            let internal = pollster::block_on(self.internal.pop());
            validation.or(internal).map(|e| e.to_string())
        }
    }

    pub struct Device {
        pub device: wgpu::Device,
        pub queue: wgpu::Queue,
        pub name: String,
        /// The test binary's GPU turn (`gpu_test_lock`).
        #[cfg(test)]
        _gpu: crate::gpu_test_lock::Token,
    }

    impl Device {
        pub fn new() -> Result<Device, String> {
            #[cfg(test)]
            let gpu = crate::gpu_test_lock::acquire();
            let instance = wgpu::Instance::default();
            let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default())).map_err(|e| format!("no adapter: {e}"))?;
            let name = format!("{} ({:?})", adapter.get_info().name, adapter.get_info().backend);
            // Subgroups when the adapter has them (the M3 Max does): a kernel
            // that uses a subgroup built-in compiles only on a device that
            // asked for the feature (2026-10-12). Every later validation is
            // for this device's features and limits.
            // SHADER_F16 and TEXTURE_FORMAT_16BIT_NORM too, as `GpuEvaluator` asks and as the recorded
            // target names: the installed target is every later validation in the process, so a device
            // without them made `enable f16;` kernels fail validation in whatever test ran after it.
            let features = adapter.features()
                & (wgpu::Features::TIMESTAMP_QUERY
                    | wgpu::Features::SUBGROUP
                    | wgpu::Features::SUBGROUP_BARRIER
                    | wgpu::Features::SHADER_F16
                    | wgpu::Features::TEXTURE_FORMAT_16BIT_NORM);
            let limits = adapter.limits();
            // The adapter's own limits, requested whole below: the
            // binding limit a problem may plan by (`gram::adapter_buffer_limit`).
            eprintln!("device {name}: max_storage_buffer_binding_size {} bytes, max_buffer_size {} bytes (the adapter's)", limits.max_storage_buffer_binding_size, limits.max_buffer_size);
            crate::wgsl::target::install(crate::wgsl::target::Target::of_device(&adapter, features, &limits));
            let (device, queue) = pollster::block_on(adapter.request_device(
                &wgpu::DeviceDescriptor { label: Some("wgsl-oracle"), required_features: features, required_limits: limits, ..Default::default() },
            ))
            .map_err(|e| format!("request_device: {e}"))?;
            Ok(Device {
                device,
                queue,
                name,
                #[cfg(test)]
                _gpu: gpu,
            })
        }
    }
}

/// A kernel the backend cannot compile is reported, not fatal (2026-10-21, run 6g: a wgsl_fit run died at generation 12
/// on a kernel whose `bitcast<vec3<i32>>(modf(vec3<f32>(..)).whole)` naga validates and writes as an MSL `as_type` between
/// a packed_float3 and an int3, which Metal rejects). Needs a GPU, run on whatever backend the machine has; on one that
/// compiles the kernel the test says so and checks nothing.
#[cfg(all(test, feature = "gpu"))]
mod device_compile_failure_tests {
    use super::device::{Device, ErrorScopes};

    const MODF_BITCAST: &str = "@group(0) @binding(0) var<storage, read_write> out: array<i32>;
@compute @workgroup_size(1)
fn main() {
    let x = f32(out[1]);
    let v = bitcast<vec3<i32>>(modf(vec3<f32>(x, x, x)).whole);
    out[0] = v.x;
}
";

    #[test]
    fn a_kernel_the_backend_cannot_compile_is_an_error_not_a_panic() {
        let dev = Device::new().expect("device");
        let scope = ErrorScopes::push(&dev.device);
        let shader = dev.device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("modf"), source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(MODF_BITCAST)) });
        let _pipeline = dev.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some("main"), layout: None, module: &shader, entry_point: Some("main"), compilation_options: Default::default(), cache: None });
        match scope.pop() {
            Some(e) => assert!(e.contains("as_type") || e.contains("ipeline") || e.contains("Metal"), "an error that does not name the compile: {e}"),
            None => eprintln!("this backend ({}) compiles the kernel; nothing to check", dev.name),
        }
    }
}

/// THE APPLE METAL HANG'S CONSTRUCT, MEASURED WITHOUT A DISPATCH (docs/reports/2026-10-21_metal_hang_k2): the
/// reproducer is never run on the device (a hung dispatch starves the display's WindowServer; 2026-10-21 needed a hard
/// reset). Its loops in each spelling go through the Metal source wgpu 30 hands Apple's compiler
/// (`target::metal_source`): the hang needs naga's continuing-step form, `loop_init` in the MSL.
#[cfg(all(test, feature = "gpu", feature = "wgsl"))]
mod metal_hang_msl_tests {
    fn reproducer() -> String {
        std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/docs/reports/2026-10-21_metal_hang_k2/gene583_reduced.wgsl")).expect("the reproducer")
    }

    /// `src` with `from` replaced once (exactly one occurrence).
    fn once(src: &str, from: &str, to: &str) -> String {
        assert_eq!(src.matches(from).count(), 1, "{from}");
        src.replacen(from, to, 1)
    }

    /// The reproducer's two `for` loops spelled `for` (as written), `shadow` (the kingdom's `loops.py` `loop_text`) or
    /// `tail` (the step at the end of the body).
    fn spelled(form: &str) -> String {
        let src = reproducer();
        let (first, head, end) = ("k1 = 0u; for (; k1 < n1; k1 = k1 + 1u) { {} }", "kx1 = 0u; for (; kx1 < nx1; kx1 = kx1 + 1u) {", "if (lit_bool(true)) { break; } }");
        match form {
            "for" => src,
            "shadow" => {
                let s = once(&src, "var k1: u32 = 0u;", "var k1: u32 = 0u; var kc1: u32 = 0u; var kxc1: u32 = 0u;");
                let s = once(&s, first, "kc1 = 0u; loop { k1 = kc1; if (k1 >= n1) { break; } kc1 = k1 + 1u; {} }");
                once(&s, head, "kxc1 = 0u; loop { kx1 = kxc1; if (kx1 >= nx1) { break; } kxc1 = kx1 + 1u;")
            }
            "tail" => {
                let s = once(&src, first, "k1 = 0u; loop { if (k1 >= n1) { break; } {} k1 = k1 + 1u; }");
                let s = once(&s, head, "kx1 = 0u; loop { if (kx1 >= nx1) { break; }");
                once(&s, end, "if (lit_bool(true)) { break; } kx1 = kx1 + 1u; }")
            }
            other => panic!("no spelling {other}"),
        }
    }

    fn msl(text: &str) -> String {
        let module = crate::wgsl::naga_shape::parse_unscanned(text).expect("parses");
        let info = crate::wgsl::target::Target::current().validate(&module, text).expect("validates");
        crate::wgsl::target::metal_source(&module, &info, "main", &[]).expect("the Metal source")
    }

    /// The plain `for` gives Apple's compiler naga's `loop_init` form (the hang's construct); the kingdom's shadow
    /// counter and the step at the end of the body give a plain `while (true)` with no continuing step.
    #[test]
    fn the_metal_hangs_loop_init_comes_from_the_for_loop_only() {
        assert!(msl(&spelled("for")).contains("loop_init"), "the for spelling writes no loop_init");
        for form in ["shadow", "tail"] {
            let m = msl(&spelled(form));
            assert!(!m.contains("loop_init"), "the {form} spelling writes loop_init");
        }
    }
}
