//! THE TWO WGPUS the runtime builds against: 30.0.1 (feature `wgpu30`, the default: fuller's and the shim's wgpu) and
//! 27.0.1 (feature `wgpu27`: UltraDim's, so a generated module runs on UltraDim's own device). Generated code never
//! names wgpu, so the same module compiles against either. The three calls whose API differs between them:
//!
//! | call | 27.0.1 | 30.0.1 |
//! |---|---|---|
//! | error scopes | `push_error_scope(filter)`, then `pop_error_scope()` | `push_error_scope(filter)` returns a guard; `guard.pop()` |
//! | `PipelineLayoutDescriptor` | `bind_group_layouts: &[&BindGroupLayout]`, `push_constant_ranges` | `bind_group_layouts: &[Option<&BindGroupLayout>]`, `immediate_size` |
//! | `Buffer::get_mapped_range` | the `BufferView` | a `Result<BufferView, _>` |

use crate::Error;

/// An open error scope.
pub(crate) struct Scope {
    #[cfg(feature = "wgpu30")]
    guard: wgpu::ErrorScopeGuard,
    #[cfg(feature = "wgpu27")]
    device: wgpu::Device,
}

pub(crate) fn push_scope(device: &wgpu::Device, filter: wgpu::ErrorFilter) -> Scope {
    #[cfg(feature = "wgpu30")]
    {
        Scope { guard: device.push_error_scope(filter) }
    }
    #[cfg(feature = "wgpu27")]
    {
        device.push_error_scope(filter);
        Scope { device: device.clone() }
    }
}

impl Scope {
    /// Close the scope: the first error it caught, if any.
    pub(crate) fn pop(self) -> Option<wgpu::Error> {
        #[cfg(feature = "wgpu30")]
        {
            pollster::block_on(self.guard.pop())
        }
        #[cfg(feature = "wgpu27")]
        {
            pollster::block_on(self.device.pop_error_scope())
        }
    }
}

/// A pipeline layout over one bind group layout.
pub(crate) fn pipeline_layout(device: &wgpu::Device, bind_group: &wgpu::BindGroupLayout, label: &str) -> wgpu::PipelineLayout {
    #[cfg(feature = "wgpu30")]
    {
        device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some(label), bind_group_layouts: &[Some(bind_group)], immediate_size: 0 })
    }
    #[cfg(feature = "wgpu27")]
    {
        device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some(label), bind_group_layouts: &[bind_group], push_constant_ranges: &[] })
    }
}

/// The mapped view of `range` of a mapped buffer.
pub(crate) fn mapped(buffer: &wgpu::Buffer, range: std::ops::Range<u64>) -> Result<wgpu::BufferView, Error> {
    #[cfg(feature = "wgpu30")]
    {
        buffer.get_mapped_range(range).map_err(|e| Error::new(format!("a mapped buffer's view: {e}")))
    }
    #[cfg(feature = "wgpu27")]
    {
        Ok(buffer.get_mapped_range(range))
    }
}
