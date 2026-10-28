//! MEMORY: device blocks with reference counts and Futhark's free list, and host blocks.
//!
//! A memory variable of the generated code ([`DevMem`], [`HostMem`]) holds a reference to a block, or nothing.
//! `Allocate` gives it a new block, `SetMem` makes it share another variable's block, `Free` drops its reference; a
//! block whose last reference goes returns its buffer to the context's free list, as `gpu_free` does in
//! `rts/c/gpu.h`. A block made from a buffer the caller owns ([`DevMem::from_buffer`]) is never put in the free list:
//! the caller's buffer is only ever the caller's.

use std::cell::RefCell;
use std::rc::Rc;

use crate::values::Elem;
use crate::Error;

/// One entry of the free list: a buffer, its size, and the tag of the variable that allocated it.
struct FreeEntry {
    buffer: wgpu::Buffer,
    size: u64,
    tag: &'static str,
}

/// Futhark's free list (`rts/c/free_list.h`): freed buffers kept for reuse.
#[derive(Default)]
pub struct FreeList {
    entries: Vec<FreeEntry>,
}

impl FreeList {
    /// `free_list_acceptable`: big enough, and either small (< 4096 bytes), of the same tag, or wasting less than half.
    fn acceptable(size: u64, tag: &str, e: &FreeEntry) -> bool {
        size <= e.size && (e.size < 4096 || e.tag == tag || e.size < size * 2)
    }

    /// `free_list_find`: the smallest acceptable entry, removed.
    pub fn find(&mut self, size: u64, tag: &str) -> Option<(wgpu::Buffer, u64)> {
        let mut best: Option<usize> = None;
        for (i, e) in self.entries.iter().enumerate() {
            if Self::acceptable(size, tag, e) && best.is_none_or(|b| e.size < self.entries[b].size) {
                best = Some(i);
            }
        }
        best.map(|i| {
            let e = self.entries.remove(i);
            (e.buffer, e.size)
        })
    }

    pub fn insert(&mut self, buffer: wgpu::Buffer, size: u64, tag: &'static str) {
        self.entries.push(FreeEntry { buffer, size, tag });
    }

    /// Every buffer released to the driver (`gpu_free_all`).
    pub fn clear(&mut self) {
        for e in self.entries.drain(..) {
            e.buffer.destroy();
        }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// A device block.
pub struct Block {
    buffer: wgpu::Buffer,
    size: u64,
    tag: &'static str,
    /// Where the buffer goes when the block's last reference is dropped; `None` for a caller's buffer.
    pool: Option<Rc<RefCell<FreeList>>>,
}

impl Block {
    pub(crate) fn new(buffer: wgpu::Buffer, size: u64, tag: &'static str, pool: Option<Rc<RefCell<FreeList>>>) -> Block {
        Block { buffer, size, tag, pool }
    }
    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }
    pub fn size(&self) -> u64 {
        self.size
    }
}

impl Drop for Block {
    fn drop(&mut self) {
        if let Some(pool) = &self.pool {
            pool.borrow_mut().insert(self.buffer.clone(), self.size, self.tag);
        }
    }
}

/// A device memory variable.
#[derive(Default)]
pub struct DevMem {
    cell: RefCell<Option<Rc<Block>>>,
}

impl Clone for DevMem {
    /// Another variable referring to the same block (a C `memblock` passed by value).
    fn clone(&self) -> DevMem {
        DevMem { cell: RefCell::new(self.cell.borrow().clone()) }
    }
}

impl DevMem {
    pub fn new() -> DevMem {
        DevMem::default()
    }

    /// A variable holding a block over a buffer the caller owns. The buffer must allow STORAGE, COPY_SRC and COPY_DST
    /// and its size must be a multiple of 4, as every buffer the runtime makes.
    pub fn from_buffer(buffer: wgpu::Buffer) -> Result<DevMem, Error> {
        let want = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;
        if !buffer.usage().contains(want) {
            return Err(Error::new(format!(
                "a caller's buffer needs the usages STORAGE | COPY_SRC | COPY_DST, it has {:?}",
                buffer.usage()
            )));
        }
        if !buffer.size().is_multiple_of(4) {
            return Err(Error::new(format!("a caller's buffer of {} bytes (wants a multiple of 4)", buffer.size())));
        }
        let size = buffer.size();
        Ok(DevMem { cell: RefCell::new(Some(Rc::new(Block::new(buffer, size, "caller", None)))) })
    }

    /// `Allocate`: a new block of at least `size` bytes (from the free list when one fits).
    pub fn alloc(&self, ctx: &crate::FutharkCtx, size: i64, tag: &'static str) -> Result<(), Error> {
        // The old block is released first, as memblock_alloc unreferences it before allocating.
        self.cell.borrow_mut().take();
        let b = ctx.alloc_block(size, tag)?;
        *self.cell.borrow_mut() = Some(Rc::new(b));
        Ok(())
    }

    /// `SetMem`: refer to the other variable's block.
    pub fn set_from(&self, other: &DevMem) {
        let b = other.cell.borrow().clone();
        *self.cell.borrow_mut() = b;
    }

    /// `Free`: drop this variable's reference.
    pub fn free(&self) {
        self.cell.borrow_mut().take();
    }

    /// The block, or an error naming the unset variable.
    pub fn block(&self) -> Result<Rc<Block>, Error> {
        self.cell.borrow().clone().ok_or_else(|| Error::new("our software has a bug: a device memory variable used before it was set"))
    }

    /// The buffer the variable refers to.
    pub fn buffer(&self) -> Result<wgpu::Buffer, Error> {
        Ok(self.block()?.buffer.clone())
    }

    /// Whether two variables refer to the same block.
    pub fn same_block(&self, other: &DevMem) -> bool {
        match (&*self.cell.borrow(), &*other.cell.borrow()) {
            (Some(a), Some(b)) => Rc::ptr_eq(a, b),
            _ => false,
        }
    }
}

/// A host memory variable (Futhark's default space): bytes.
#[derive(Default)]
pub struct HostMem {
    cell: RefCell<Option<Rc<RefCell<Vec<u8>>>>>,
}

impl Clone for HostMem {
    fn clone(&self) -> HostMem {
        HostMem { cell: RefCell::new(self.cell.borrow().clone()) }
    }
}

impl HostMem {
    pub fn new() -> HostMem {
        HostMem::default()
    }

    /// A constant array (`DeclareArray`).
    pub fn from_values<T: Elem>(vs: &[T]) -> HostMem {
        let mut bytes = vec![0u8; vs.len() * T::SIZE];
        for (i, v) in vs.iter().enumerate() {
            v.write_le(&mut bytes[i * T::SIZE..(i + 1) * T::SIZE]);
        }
        HostMem { cell: RefCell::new(Some(Rc::new(RefCell::new(bytes)))) }
    }

    pub fn zeros(bytes: usize) -> HostMem {
        HostMem { cell: RefCell::new(Some(Rc::new(RefCell::new(vec![0u8; bytes])))) }
    }

    pub fn alloc(&self, size: i64) {
        *self.cell.borrow_mut() = Some(Rc::new(RefCell::new(vec![0u8; size.max(0) as usize])));
    }

    pub fn set_from(&self, other: &HostMem) {
        let b = other.cell.borrow().clone();
        *self.cell.borrow_mut() = b;
    }

    pub fn free(&self) {
        self.cell.borrow_mut().take();
    }

    pub fn bytes(&self) -> Result<Rc<RefCell<Vec<u8>>>, Error> {
        self.cell.borrow().clone().ok_or_else(|| Error::new("our software has a bug: a host memory variable used before it was set"))
    }

    pub fn read<T: Elem>(&self, i: i64) -> Result<T, Error> {
        let b = self.bytes()?;
        let b = b.borrow();
        let o = i as usize * T::SIZE;
        b.get(o..o + T::SIZE).map(T::read_le).ok_or_else(|| Error::new(format!("a host read at element {i} past {} bytes", b.len())))
    }

    pub fn write<T: Elem>(&self, i: i64, v: T) -> Result<(), Error> {
        let b = self.bytes()?;
        let mut b = b.borrow_mut();
        let o = i as usize * T::SIZE;
        let n = b.len();
        let dst = b.get_mut(o..o + T::SIZE).ok_or_else(|| Error::new(format!("a host write at element {i} past {n} bytes")))?;
        v.write_le(dst);
        Ok(())
    }
}
