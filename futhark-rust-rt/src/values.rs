//! VALUES: typed array handles over device buffers ([`Array`]), the dynamic values of a host that loads a module it
//! was not compiled with ([`Value`], [`EntryDesc`]), and Futhark's binary value format ([`HostValue`]).

use std::marker::PhantomData;

use crate::{DevMem, Error, FutharkCtx};

/// An element type of the API, with signedness (Futhark's `u32` is held as `i32` inside a program).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ElemType {
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    F32,
    F64,
    Bool,
}

impl ElemType {
    pub fn size(self) -> usize {
        match self {
            ElemType::I8 | ElemType::U8 | ElemType::Bool => 1,
            ElemType::I16 | ElemType::U16 => 2,
            ElemType::I32 | ElemType::U32 | ElemType::F32 => 4,
            ElemType::I64 | ElemType::U64 | ElemType::F64 => 8,
        }
    }
    /// The four-character name of Futhark's binary format.
    pub fn binary_name(self) -> &'static str {
        match self {
            ElemType::I8 => "  i8",
            ElemType::I16 => " i16",
            ElemType::I32 => " i32",
            ElemType::I64 => " i64",
            ElemType::U8 => "  u8",
            ElemType::U16 => " u16",
            ElemType::U32 => " u32",
            ElemType::U64 => " u64",
            ElemType::F32 => " f32",
            ElemType::F64 => " f64",
            ElemType::Bool => "bool",
        }
    }
    pub fn from_binary_name(s: &[u8]) -> Option<ElemType> {
        [
            ElemType::I8,
            ElemType::I16,
            ElemType::I32,
            ElemType::I64,
            ElemType::U8,
            ElemType::U16,
            ElemType::U32,
            ElemType::U64,
            ElemType::F32,
            ElemType::F64,
            ElemType::Bool,
        ]
        .into_iter()
        .find(|t| t.binary_name().as_bytes() == s)
    }
}

/// A primitive element: its type, size and little-endian bytes.
pub trait Elem: Copy + 'static {
    const TYPE: ElemType;
    const SIZE: usize;
    fn write_le(self, out: &mut [u8]);
    fn read_le(b: &[u8]) -> Self;
}

macro_rules! elem {
    ($($t:ty, $tag:ident);*) => {
        $(
            impl Elem for $t {
                const TYPE: ElemType = ElemType::$tag;
                const SIZE: usize = std::mem::size_of::<$t>();
                fn write_le(self, out: &mut [u8]) {
                    out.copy_from_slice(&self.to_le_bytes());
                }
                fn read_le(b: &[u8]) -> Self {
                    let mut a = [0u8; std::mem::size_of::<$t>()];
                    a.copy_from_slice(b);
                    <$t>::from_le_bytes(a)
                }
            }
        )*
    };
}

elem!(i8, I8; i16, I16; i32, I32; i64, I64; u8, U8; u16, U16; u32, U32; u64, U64; f32, F32; f64, F64);

impl Elem for bool {
    const TYPE: ElemType = ElemType::Bool;
    const SIZE: usize = 1;
    fn write_le(self, out: &mut [u8]) {
        out[0] = self as u8;
    }
    fn read_le(b: &[u8]) -> Self {
        b[0] != 0
    }
}

/// A rank-`R` array of `T` in device memory: a handle (the memory is shared, never copied, by a clone).
pub struct Array<T: Elem, const R: usize> {
    mem: DevMem,
    shape: [i64; R],
    _t: PhantomData<T>,
}

impl<T: Elem, const R: usize> Clone for Array<T, R> {
    fn clone(&self) -> Self {
        Array { mem: self.mem.clone(), shape: self.shape, _t: PhantomData }
    }
}

impl<T: Elem, const R: usize> Array<T, R> {
    fn elements(shape: &[i64; R]) -> Result<u64, Error> {
        let mut n: u64 = 1;
        for &d in shape {
            if d < 0 {
                return Err(Error::new(format!("a negative dimension {d}")));
            }
            n = n.checked_mul(d as u64).ok_or_else(|| Error::new("an array too large to address"))?;
        }
        Ok(n)
    }

    /// An array over a buffer the caller owns: no copy; the program reads it in place (and writes it, for an entry
    /// point that consumes its argument). The buffer must hold the shape's elements, row-major, from offset 0.
    pub fn from_buffer(buffer: wgpu::Buffer, shape: [i64; R]) -> Result<Self, Error> {
        let need = Self::elements(&shape)? * T::SIZE as u64;
        if buffer.size() < need {
            return Err(Error::new(format!("a buffer of {} bytes for an array of {need}", buffer.size())));
        }
        Ok(Array { mem: DevMem::from_buffer(buffer)?, shape, _t: PhantomData })
    }

    /// An array holding a copy of host data (the runtime's memory, from the context's free list).
    pub fn from_slice(ctx: &FutharkCtx, data: &[T], shape: [i64; R]) -> Result<Self, Error> {
        let n = Self::elements(&shape)?;
        if n != data.len() as u64 {
            return Err(Error::new(format!("{} elements for the shape {shape:?}", data.len())));
        }
        let mem = DevMem::new();
        mem.alloc(ctx, n as i64 * T::SIZE as i64, "array")?;
        let mut bytes = vec![0u8; data.len() * T::SIZE];
        for (i, v) in data.iter().enumerate() {
            v.write_le(&mut bytes[i * T::SIZE..(i + 1) * T::SIZE]);
        }
        if !bytes.is_empty() {
            bytes.resize(bytes.len().div_ceil(4) * 4, 0);
            ctx.queue().write_buffer(&mem.buffer()?, 0, &bytes);
        }
        Ok(Array { mem, shape, _t: PhantomData })
    }

    /// An array over memory a program returned.
    pub fn from_mem(mem: DevMem, shape: [i64; R]) -> Self {
        Array { mem, shape, _t: PhantomData }
    }

    pub fn mem(&self) -> &DevMem {
        &self.mem
    }
    pub fn shape(&self) -> &[i64; R] {
        &self.shape
    }
    /// The device buffer holding the elements, row-major, from offset 0 (it may be larger than the array).
    pub fn buffer(&self) -> Result<wgpu::Buffer, Error> {
        self.mem.buffer()
    }

    /// The elements, read back to the host.
    pub fn to_vec(&self, ctx: &FutharkCtx) -> Result<Vec<T>, Error> {
        let n = Self::elements(&self.shape)? as usize;
        let bytes = read_bytes(ctx, &self.mem, n * T::SIZE)?;
        Ok((0..n).map(|i| T::read_le(&bytes[i * T::SIZE..(i + 1) * T::SIZE])).collect())
    }
}

/// The first `n` bytes of a device block.
fn read_bytes(ctx: &FutharkCtx, mem: &DevMem, n: usize) -> Result<Vec<u8>, Error> {
    let host = crate::HostMem::zeros(n);
    if n > 0 {
        crate::lmad_copy_dev2host(ctx, 1, crate::Lmad::new(&host, 0, &[1]), crate::Lmad::new(mem, 0, &[1]), &[n as i64])?;
    }
    let b = host.bytes()?;
    let v = b.borrow().clone();
    Ok(v)
}

/// A scalar of any element type.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Scalar {
    I8(i8),
    I16(i16),
    I32(i32),
    I64(i64),
    U8(u8),
    U16(u16),
    U32(u32),
    U64(u64),
    F32(f32),
    F64(f64),
    Bool(bool),
}

impl Scalar {
    pub fn elem(&self) -> ElemType {
        match self {
            Scalar::I8(_) => ElemType::I8,
            Scalar::I16(_) => ElemType::I16,
            Scalar::I32(_) => ElemType::I32,
            Scalar::I64(_) => ElemType::I64,
            Scalar::U8(_) => ElemType::U8,
            Scalar::U16(_) => ElemType::U16,
            Scalar::U32(_) => ElemType::U32,
            Scalar::U64(_) => ElemType::U64,
            Scalar::F32(_) => ElemType::F32,
            Scalar::F64(_) => ElemType::F64,
            Scalar::Bool(_) => ElemType::Bool,
        }
    }
    fn bytes(&self) -> Vec<u8> {
        match *self {
            Scalar::I8(x) => x.to_le_bytes().to_vec(),
            Scalar::I16(x) => x.to_le_bytes().to_vec(),
            Scalar::I32(x) => x.to_le_bytes().to_vec(),
            Scalar::I64(x) => x.to_le_bytes().to_vec(),
            Scalar::U8(x) => x.to_le_bytes().to_vec(),
            Scalar::U16(x) => x.to_le_bytes().to_vec(),
            Scalar::U32(x) => x.to_le_bytes().to_vec(),
            Scalar::U64(x) => x.to_le_bytes().to_vec(),
            Scalar::F32(x) => x.to_le_bytes().to_vec(),
            Scalar::F64(x) => x.to_le_bytes().to_vec(),
            Scalar::Bool(x) => vec![x as u8],
        }
    }
    fn from_bytes(t: ElemType, b: &[u8]) -> Scalar {
        match t {
            ElemType::I8 => Scalar::I8(i8::read_le(b)),
            ElemType::I16 => Scalar::I16(i16::read_le(b)),
            ElemType::I32 => Scalar::I32(i32::read_le(b)),
            ElemType::I64 => Scalar::I64(i64::read_le(b)),
            ElemType::U8 => Scalar::U8(u8::read_le(b)),
            ElemType::U16 => Scalar::U16(u16::read_le(b)),
            ElemType::U32 => Scalar::U32(u32::read_le(b)),
            ElemType::U64 => Scalar::U64(u64::read_le(b)),
            ElemType::F32 => Scalar::F32(f32::read_le(b)),
            ElemType::F64 => Scalar::F64(f64::read_le(b)),
            ElemType::Bool => Scalar::Bool(bool::read_le(b)),
        }
    }
}

/// An array of any element type and rank.
#[derive(Clone)]
pub struct AnyArray {
    pub elem: ElemType,
    pub shape: Vec<i64>,
    pub mem: DevMem,
}

/// A value of an entry point, dynamically typed.
#[derive(Clone)]
pub enum Value {
    Scalar(Scalar),
    Array(AnyArray),
}

impl Value {
    pub fn type_desc(&self) -> TypeDesc {
        match self {
            Value::Scalar(s) => TypeDesc { elem: s.elem(), rank: 0 },
            Value::Array(a) => TypeDesc { elem: a.elem, rank: a.shape.len() },
        }
    }

    pub fn as_array<T: Elem, const R: usize>(&self) -> Result<Array<T, R>, Error> {
        match self {
            Value::Array(a) if a.elem == T::TYPE && a.shape.len() == R => {
                let mut shape = [0i64; R];
                shape.copy_from_slice(&a.shape);
                Ok(Array::from_mem(a.mem.clone(), shape))
            }
            other => Err(Error::new(format!("an argument of type {:?}, wanted {:?} of rank {R}", other.type_desc(), T::TYPE))),
        }
    }

    pub fn as_scalar<T: Elem>(&self) -> Result<T, Error> {
        match self {
            Value::Scalar(s) if s.elem() == T::TYPE => Ok(T::read_le(&s.bytes())),
            other => Err(Error::new(format!("an argument of type {:?}, wanted the scalar {:?}", other.type_desc(), T::TYPE))),
        }
    }

    /// The value uploaded from its host form.
    pub fn from_host(ctx: &FutharkCtx, h: &HostValue) -> Result<Value, Error> {
        if h.shape.is_empty() {
            if h.bytes.len() != h.elem.size() {
                return Err(Error::new("a scalar of the wrong size"));
            }
            return Ok(Value::Scalar(Scalar::from_bytes(h.elem, &h.bytes)));
        }
        let mem = DevMem::new();
        mem.alloc(ctx, h.bytes.len() as i64, "array")?;
        if !h.bytes.is_empty() {
            let mut bytes = h.bytes.clone();
            bytes.resize(bytes.len().div_ceil(4) * 4, 0);
            ctx.queue().write_buffer(&mem.buffer()?, 0, &bytes);
        }
        Ok(Value::Array(AnyArray { elem: h.elem, shape: h.shape.clone(), mem }))
    }

    /// The value read back to its host form.
    pub fn to_host(&self, ctx: &FutharkCtx) -> Result<HostValue, Error> {
        match self {
            Value::Scalar(s) => Ok(HostValue { elem: s.elem(), shape: vec![], bytes: s.bytes() }),
            Value::Array(a) => {
                let n: i64 = a.shape.iter().product();
                let bytes = read_bytes(ctx, &a.mem, n as usize * a.elem.size())?;
                Ok(HostValue { elem: a.elem, shape: a.shape.clone(), bytes })
            }
        }
    }
}

impl<T: Elem, const R: usize> From<Array<T, R>> for Value {
    fn from(a: Array<T, R>) -> Value {
        Value::Array(AnyArray { elem: T::TYPE, shape: a.shape.to_vec(), mem: a.mem })
    }
}

macro_rules! scalar_value {
    ($($t:ty, $tag:ident);*) => {
        $(
            impl From<$t> for Value {
                fn from(x: $t) -> Value {
                    Value::Scalar(Scalar::$tag(x))
                }
            }
            impl IntoValues for $t {
                fn into_values(self) -> Vec<Value> {
                    vec![Value::from(self)]
                }
            }
        )*
    };
}

scalar_value!(i8, I8; i16, I16; i32, I32; i64, I64; u8, U8; u16, U16; u32, U32; u64, U64; f32, F32; f64, F64; bool, Bool);

/// An opaque value of an entry point (a tuple or record of the program): its components, in Futhark's order.
pub struct Opaque {
    pub name: String,
    pub values: Vec<Value>,
}

impl Opaque {
    pub fn new(name: &str, values: Vec<Value>) -> Opaque {
        Opaque { name: name.to_string(), values }
    }
}

/// A result as dynamic values: one, or an opaque value's components.
pub trait IntoValues {
    fn into_values(self) -> Vec<Value>;
}

impl<T: Elem, const R: usize> IntoValues for Array<T, R> {
    fn into_values(self) -> Vec<Value> {
        vec![Value::from(self)]
    }
}

impl IntoValues for Opaque {
    fn into_values(self) -> Vec<Value> {
        self.values
    }
}

/// The type of a value: element type and rank (0: a scalar).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TypeDesc {
    pub elem: ElemType,
    pub rank: usize,
}

/// An entry point over dynamic values: its name, its inputs' and outputs' types (an opaque output's components), and
/// the call.
pub struct EntryDesc {
    pub name: &'static str,
    pub inputs: &'static [TypeDesc],
    pub outputs: &'static [TypeDesc],
    pub call: fn(&FutharkCtx, &[Value]) -> Result<Vec<Value>, Error>,
}

pub fn check_arity(entry: &str, args: &[Value], n: usize) -> Result<(), Error> {
    if args.len() != n {
        return Err(Error::new(format!("the entry point {entry} takes {n} arguments, given {}", args.len())));
    }
    Ok(())
}

/// Two dimensions an entry point's type says are equal (or a dimension and its constant).
pub fn check_dim(entry: &str, got: i64, want: i64) -> Result<(), Error> {
    if got != want {
        return Err(Error::new(format!("Error: entry point arguments have invalid sizes ({entry}: a dimension {got} where {want} is required).")));
    }
    Ok(())
}

// ---- Futhark's binary value format ----------------------------------------------------------------------------

/// A value on the host: element type, shape (empty: a scalar), little-endian bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HostValue {
    pub elem: ElemType,
    pub shape: Vec<i64>,
    pub bytes: Vec<u8>,
}

impl HostValue {
    /// Futhark's binary format: `b`, version 2, the rank, the type's four characters, the dimensions, the data.
    pub fn write_binary(&self, out: &mut Vec<u8>) {
        out.push(b'b');
        out.push(2);
        out.push(self.shape.len() as u8);
        out.extend_from_slice(self.elem.binary_name().as_bytes());
        for d in &self.shape {
            out.extend_from_slice(&d.to_le_bytes());
        }
        out.extend_from_slice(&self.bytes);
    }

    /// Every value of a byte string in Futhark's binary format (whitespace between values allowed).
    pub fn read_all(raw: &[u8]) -> Result<Vec<HostValue>, Error> {
        let mut out = Vec::new();
        let mut i = 0;
        let bad = |what: &str| Error::new(format!("not Futhark's binary value format: {what}"));
        loop {
            while i < raw.len() && raw[i].is_ascii_whitespace() {
                i += 1;
            }
            if i == raw.len() {
                return Ok(out);
            }
            if raw.len() < i + 7 || raw[i] != b'b' || raw[i + 1] != 2 {
                return Err(bad("a value header"));
            }
            let rank = raw[i + 2] as usize;
            let elem = ElemType::from_binary_name(&raw[i + 3..i + 7]).ok_or_else(|| bad("an element type"))?;
            i += 7;
            let mut shape = Vec::with_capacity(rank);
            for _ in 0..rank {
                let b = raw.get(i..i + 8).ok_or_else(|| bad("a dimension"))?;
                shape.push(i64::read_le(b));
                i += 8;
            }
            let n: i64 = shape.iter().product();
            let len = n as usize * elem.size();
            let bytes = raw.get(i..i + len).ok_or_else(|| bad("the data"))?.to_vec();
            i += len;
            out.push(HostValue { elem, shape, bytes });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_binary_format_round_trips() {
        let vs = vec![
            HostValue { elem: ElemType::I32, shape: vec![3], bytes: [1i32, -2, 3].iter().flat_map(|x| x.to_le_bytes()).collect() },
            HostValue { elem: ElemType::I32, shape: vec![], bytes: 7i32.to_le_bytes().to_vec() },
            HostValue { elem: ElemType::F32, shape: vec![2, 0], bytes: vec![] },
        ];
        let mut raw = Vec::new();
        for v in &vs {
            v.write_binary(&mut raw);
        }
        assert_eq!(HostValue::read_all(&raw).expect("the values"), vs);
    }
}
