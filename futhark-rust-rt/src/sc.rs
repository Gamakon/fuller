//! THE SCALAR FUNCTIONS the generated host code calls, by the names Futhark gives its operators (`prettyString` of a
//! `BinOp`, `CmpOp`, `ConvOp` or `UnOp`, as Futhark's C backends call them), with the semantics of the host branch of
//! Futhark's `rts/c/scalar.h` (no OpenCL, CUDA or ISPC): floor division and modulo for `sdiv`/`smod`, truncating
//! `squot`/`srem`, `-1` divisors defined, the `_safe` forms 0 on a zero divisor, unsigned operations on the
//! two's-complement bits of the signed storage type, wrapping arithmetic. Every integer is held in its SIGNED Rust type,
//! as Futhark's host holds it.
//!
//! The integer addition, subtraction and multiplication Futhark marks "no wrap" (`add_nw64`) are written by C as plain
//! `+`, whose overflow is undefined; here they wrap, which agrees with C on every program whose arithmetic does not
//! overflow and is defined on the ones that do.
//!
//! A builtin function of the language (`FunExp`, e.g. `sqrt32`) is `fun_<name>`.

// The functions, written out per width by `int_fns!`: no proc-macro dependency, so the names are spelled in full.
macro_rules! int_fns {
    ($s:ty, $u:ty,
     $add:ident, $add_nw:ident, $sub:ident, $sub_nw:ident, $mul:ident, $mul_nw:ident,
     $udiv:ident, $udiv_safe:ident, $uceil_div:ident, $uceil_div_safe:ident, $umod:ident, $umod_safe:ident,
     $sdiv:ident, $sdiv_safe:ident, $sceil_div:ident, $sceil_div_safe:ident, $smod:ident, $smod_safe:ident,
     $squot:ident, $squot_safe:ident, $srem:ident, $srem_safe:ident,
     $smin:ident, $umin:ident, $smax:ident, $umax:ident, $shl:ident, $lshr:ident, $ashr:ident,
     $and:ident, $or:ident, $xor:ident, $pow:ident, $ult:ident, $ule:ident, $slt:ident, $sle:ident,
     $abs:ident, $ssignum:ident, $usignum:ident, $complement:ident,
     $popc:ident, $clzz:ident, $ctzz:ident, $umul_hi:ident, $smul_hi:ident, $wide:ty, $uwide:ty, $bits:literal) => {
        pub fn $add(x: $s, y: $s) -> $s { x.wrapping_add(y) }
        pub fn $add_nw(x: $s, y: $s) -> $s { x.wrapping_add(y) }
        pub fn $sub(x: $s, y: $s) -> $s { x.wrapping_sub(y) }
        pub fn $sub_nw(x: $s, y: $s) -> $s { x.wrapping_sub(y) }
        pub fn $mul(x: $s, y: $s) -> $s { x.wrapping_mul(y) }
        pub fn $mul_nw(x: $s, y: $s) -> $s { x.wrapping_mul(y) }
        pub fn $udiv(x: $s, y: $s) -> $s { ((x as $u) / (y as $u)) as $s }
        pub fn $udiv_safe(x: $s, y: $s) -> $s { if y == 0 { 0 } else { $udiv(x, y) } }
        pub fn $uceil_div(x: $s, y: $s) -> $s {
            let (x, y) = (x as $u, y as $u);
            (x / y + if x % y != 0 { 1 } else { 0 }) as $s
        }
        pub fn $uceil_div_safe(x: $s, y: $s) -> $s { if y == 0 { 0 } else { $uceil_div(x, y) } }
        pub fn $umod(x: $s, y: $s) -> $s { ((x as $u) % (y as $u)) as $s }
        pub fn $umod_safe(x: $s, y: $s) -> $s { if y == 0 { 0 } else { $umod(x, y) } }
        pub fn $sdiv(x: $s, y: $s) -> $s {
            if y == -1 {
                return x.wrapping_neg();
            }
            let q = x / y;
            let r = x % y;
            q - if r != 0 && ((r < 0) != (y < 0)) { 1 } else { 0 }
        }
        pub fn $sdiv_safe(x: $s, y: $s) -> $s { if y == 0 { 0 } else { $sdiv(x, y) } }
        pub fn $sceil_div(x: $s, y: $s) -> $s {
            if y == -1 {
                return x.wrapping_neg();
            }
            let q = x / y;
            let r = x % y;
            q + if r != 0 && ((r > 0) == (y > 0)) { 1 } else { 0 }
        }
        pub fn $sceil_div_safe(x: $s, y: $s) -> $s { if y == 0 { 0 } else { $sceil_div(x, y) } }
        pub fn $smod(x: $s, y: $s) -> $s {
            if y == -1 {
                return 0;
            }
            let r = x % y;
            r + if r == 0 || (x > 0 && y > 0) || (x < 0 && y < 0) { 0 } else { y }
        }
        pub fn $smod_safe(x: $s, y: $s) -> $s { if y == 0 { 0 } else { $smod(x, y) } }
        pub fn $squot(x: $s, y: $s) -> $s { if y == -1 { x.wrapping_neg() } else { x / y } }
        pub fn $squot_safe(x: $s, y: $s) -> $s { if y == 0 { 0 } else { $squot(x, y) } }
        pub fn $srem(x: $s, y: $s) -> $s { if y == -1 { 0 } else { x % y } }
        pub fn $srem_safe(x: $s, y: $s) -> $s { if y == 0 { 0 } else { $srem(x, y) } }
        pub fn $smin(x: $s, y: $s) -> $s { if x < y { x } else { y } }
        pub fn $umin(x: $s, y: $s) -> $s { if (x as $u) < (y as $u) { x } else { y } }
        pub fn $smax(x: $s, y: $s) -> $s { if x < y { y } else { x } }
        pub fn $umax(x: $s, y: $s) -> $s { if (x as $u) < (y as $u) { y } else { x } }
        pub fn $shl(x: $s, y: $s) -> $s { (x as $u).wrapping_shl(y as u32) as $s }
        pub fn $lshr(x: $s, y: $s) -> $s { (x as $u).wrapping_shr(y as u32) as $s }
        pub fn $ashr(x: $s, y: $s) -> $s { x.wrapping_shr(y as u32) }
        pub fn $and(x: $s, y: $s) -> $s { x & y }
        pub fn $or(x: $s, y: $s) -> $s { x | y }
        pub fn $xor(x: $s, y: $s) -> $s { x ^ y }
        pub fn $pow(x: $s, y: $s) -> $s {
            let (mut x, mut rem, mut res) = (x as $u, y as $u, 1 as $u);
            while rem != 0 {
                if rem & 1 != 0 {
                    res = res.wrapping_mul(x);
                }
                rem >>= 1;
                x = x.wrapping_mul(x);
            }
            res as $s
        }
        pub fn $ult(x: $s, y: $s) -> bool { (x as $u) < (y as $u) }
        pub fn $ule(x: $s, y: $s) -> bool { (x as $u) <= (y as $u) }
        pub fn $slt(x: $s, y: $s) -> bool { x < y }
        pub fn $sle(x: $s, y: $s) -> bool { x <= y }
        pub fn $abs(x: $s) -> $s { x.wrapping_abs() }
        pub fn $ssignum(x: $s) -> $s { (x > 0) as $s - (x < 0) as $s }
        pub fn $usignum(x: $s) -> $s { (x != 0) as $s }
        pub fn $complement(x: $s) -> $s { !x }
        pub fn $popc(x: $s) -> i32 { x.count_ones() as i32 }
        pub fn $clzz(x: $s) -> i32 { x.leading_zeros() as i32 }
        pub fn $ctzz(x: $s) -> i32 { if x == 0 { $bits } else { x.trailing_zeros() as i32 } }
        pub fn $umul_hi(x: $s, y: $s) -> $s { (((x as $u as $uwide) * (y as $u as $uwide)) >> $bits) as $s }
        pub fn $smul_hi(x: $s, y: $s) -> $s { (((x as $wide) * (y as $wide)) >> $bits) as $s }
    };
}

int_fns!(i8, u8, add8, add_nw8, sub8, sub_nw8, mul8, mul_nw8, udiv8, udiv_safe8, uceil_div8, uceil_div_safe8, umod8,
    umod_safe8, sdiv8, sdiv_safe8, sceil_div8, sceil_div_safe8, smod8, smod_safe8, squot8, squot_safe8, srem8,
    srem_safe8, smin8, umin8, smax8, umax8, shl8, lshr8, ashr8, and8, or8, xor8, pow8, ult8, ule8, slt8, sle8, abs8,
    ssignum8, usignum8, complement8, fun_popc8, fun_clz8, fun_ctz8, fun_umul_hi8, fun_smul_hi8, i16, u16, 8);
int_fns!(i16, u16, add16, add_nw16, sub16, sub_nw16, mul16, mul_nw16, udiv16, udiv_safe16, uceil_div16,
    uceil_div_safe16, umod16, umod_safe16, sdiv16, sdiv_safe16, sceil_div16, sceil_div_safe16, smod16, smod_safe16,
    squot16, squot_safe16, srem16, srem_safe16, smin16, umin16, smax16, umax16, shl16, lshr16, ashr16, and16, or16,
    xor16, pow16, ult16, ule16, slt16, sle16, abs16, ssignum16, usignum16, complement16, fun_popc16, fun_clz16,
    fun_ctz16, fun_umul_hi16, fun_smul_hi16, i32, u32, 16);
int_fns!(i32, u32, add32, add_nw32, sub32, sub_nw32, mul32, mul_nw32, udiv32, udiv_safe32, uceil_div32,
    uceil_div_safe32, umod32, umod_safe32, sdiv32, sdiv_safe32, sceil_div32, sceil_div_safe32, smod32, smod_safe32,
    squot32, squot_safe32, srem32, srem_safe32, smin32, umin32, smax32, umax32, shl32, lshr32, ashr32, and32, or32,
    xor32, pow32, ult32, ule32, slt32, sle32, abs32, ssignum32, usignum32, complement32, fun_popc32, fun_clz32,
    fun_ctz32, fun_umul_hi32, fun_smul_hi32, i64, u64, 32);
int_fns!(i64, u64, add64, add_nw64, sub64, sub_nw64, mul64, mul_nw64, udiv64, udiv_safe64, uceil_div64,
    uceil_div_safe64, umod64, umod_safe64, sdiv64, sdiv_safe64, sceil_div64, sceil_div_safe64, smod64, smod_safe64,
    squot64, squot_safe64, srem64, srem_safe64, smin64, umin64, smax64, umax64, shl64, lshr64, ashr64, and64, or64,
    xor64, pow64, ult64, ule64, slt64, sle64, abs64, ssignum64, usignum64, complement64, fun_popc64, fun_clz64,
    fun_ctz64, fun_umul_hi64, fun_smul_hi64, i128, u128, 64);

// ---- conversions: `<op>_<from>_<to>` --------------------------------------------------------------------------------

/// `zext`/`sext` between every pair of widths: C's `((to) (from_unsigned_or_signed) (x))`.
macro_rules! int_convs {
    ($($sext:ident, $zext:ident, $from:ty, $ufrom:ty, $to:ty);* $(;)?) => {
        $(
            pub fn $sext(x: $from) -> $to { x as $to }
            pub fn $zext(x: $from) -> $to { x as $ufrom as $to }
        )*
    };
}

int_convs!(
    sext_i8_i8, zext_i8_i8, i8, u8, i8; sext_i8_i16, zext_i8_i16, i8, u8, i16; sext_i8_i32, zext_i8_i32, i8, u8, i32;
    sext_i8_i64, zext_i8_i64, i8, u8, i64; sext_i16_i8, zext_i16_i8, i16, u16, i8;
    sext_i16_i16, zext_i16_i16, i16, u16, i16; sext_i16_i32, zext_i16_i32, i16, u16, i32;
    sext_i16_i64, zext_i16_i64, i16, u16, i64; sext_i32_i8, zext_i32_i8, i32, u32, i8;
    sext_i32_i16, zext_i32_i16, i32, u32, i16; sext_i32_i32, zext_i32_i32, i32, u32, i32;
    sext_i32_i64, zext_i32_i64, i32, u32, i64; sext_i64_i8, zext_i64_i8, i64, u64, i8;
    sext_i64_i16, zext_i64_i16, i64, u64, i16; sext_i64_i32, zext_i64_i32, i64, u64, i32;
    sext_i64_i64, zext_i64_i64, i64, u64, i64;
);

/// Integer and boolean conversions.
macro_rules! bool_convs {
    ($($itob:ident, $btoi:ident, $t:ty);* $(;)?) => {
        $(
            pub fn $itob(x: $t) -> bool { x != 0 }
            pub fn $btoi(x: bool) -> $t { x as $t }
        )*
    };
}

bool_convs!(itob_i8_bool, btoi_bool_i8, i8; itob_i16_bool, btoi_bool_i16, i16; itob_i32_bool, btoi_bool_i32, i32;
    itob_i64_bool, btoi_bool_i64, i64);

/// Integers to floats (`sitofp`, `uitofp`) and floats to integers (`fptosi`, `fptoui`: 0 outside the range and on NaN,
/// as `scalar.h` writes them).
macro_rules! float_int_convs {
    ($($sitofp:ident, $uitofp:ident, $fptosi:ident, $fptoui:ident, $i:ty, $u:ty, $f:ty, $lo:expr, $hi:expr, $uhi:expr);* $(;)?) => {
        $(
            pub fn $sitofp(x: $i) -> $f { x as $f }
            pub fn $uitofp(x: $i) -> $f { x as $u as $f }
            pub fn $fptosi(x: $f) -> $i { if x.is_nan() || !($lo..$hi).contains(&x) { 0 } else { x as $i } }
            pub fn $fptoui(x: $f) -> $i { if x.is_nan() || x <= -1.0 || x >= $uhi { 0 } else { x as $u as $i } }
        )*
    };
}

float_int_convs!(
    sitofp_i8_f32, uitofp_i8_f32, fptosi_f32_i8, fptoui_f32_i8, i8, u8, f32, -128.0, 128.0, 256.0;
    sitofp_i16_f32, uitofp_i16_f32, fptosi_f32_i16, fptoui_f32_i16, i16, u16, f32, -32768.0, 32768.0, 65536.0;
    sitofp_i32_f32, uitofp_i32_f32, fptosi_f32_i32, fptoui_f32_i32, i32, u32, f32, -2147483648.0, 2147483648.0, 4294967296.0;
    sitofp_i64_f32, uitofp_i64_f32, fptosi_f32_i64, fptoui_f32_i64, i64, u64, f32, -9223372036854775808.0, 9223372036854775808.0, 18446744073709551616.0;
    sitofp_i8_f64, uitofp_i8_f64, fptosi_f64_i8, fptoui_f64_i8, i8, u8, f64, -128.0, 128.0, 256.0;
    sitofp_i16_f64, uitofp_i16_f64, fptosi_f64_i16, fptoui_f64_i16, i16, u16, f64, -32768.0, 32768.0, 65536.0;
    sitofp_i32_f64, uitofp_i32_f64, fptosi_f64_i32, fptoui_f64_i32, i32, u32, f64, -2147483648.0, 2147483648.0, 4294967296.0;
    sitofp_i64_f64, uitofp_i64_f64, fptosi_f64_i64, fptoui_f64_i64, i64, u64, f64, -9223372036854775808.0, 9223372036854775808.0, 18446744073709551616.0;
);

pub fn fpconv_f32_f32(x: f32) -> f32 {
    x
}
pub fn fpconv_f32_f64(x: f32) -> f64 {
    x as f64
}
pub fn fpconv_f64_f32(x: f64) -> f32 {
    x as f32
}
pub fn fpconv_f64_f64(x: f64) -> f64 {
    x
}
pub fn fptobits_f32_i32(x: f32) -> i32 {
    x.to_bits() as i32
}
pub fn bitstofp_i32_f32(x: i32) -> f32 {
    f32::from_bits(x as u32)
}
pub fn fptobits_f64_i64(x: f64) -> i64 {
    x.to_bits() as i64
}
pub fn bitstofp_i64_f64(x: i64) -> f64 {
    f64::from_bits(x as u64)
}
pub fn ftob_f32_bool(x: f32) -> bool {
    x != 0.0
}
pub fn btof_bool_f32(x: bool) -> f32 {
    if x {
        1.0
    } else {
        0.0
    }
}
pub fn ftob_f64_bool(x: f64) -> bool {
    x != 0.0
}
pub fn btof_bool_f64(x: bool) -> f64 {
    if x {
        1.0
    } else {
        0.0
    }
}

// ---- floats -----------------------------------------------------------------------------------------------------

/// The float operators that are functions in C, and the language's float builtins, per width.
macro_rules! float_fns {
    ($f:ty, $bits:literal, $fabs:ident, $fmin:ident, $fmax:ident, $fpow:ident, $fmod:ident, $fsignum:ident,
     $sqrt:ident, $cbrt:ident, $exp:ident, $log:ident, $log2:ident, $log10:ident, $log1p:ident, $sin:ident,
     $cos:ident, $tan:ident, $asin:ident, $acos:ident, $atan:ident, $sinh:ident, $cosh:ident, $tanh:ident,
     $asinh:ident, $acosh:ident, $atanh:ident, $atan2:ident, $hypot:ident, $round:ident, $floor:ident,
     $ceil:ident, $trunc:ident, $isnan:ident, $isinf:ident, $fma:ident, $mad:ident, $copysign:ident,
     $to_bits:ident, $from_bits:ident, $int:ty, $uint:ty) => {
        pub fn $fabs(x: $f) -> $f { x.abs() }
        /// C's `fmin`: the other operand when one is NaN.
        pub fn $fmin(x: $f, y: $f) -> $f { x.min(y) }
        /// C's `fmax`: the other operand when one is NaN.
        pub fn $fmax(x: $f, y: $f) -> $f { x.max(y) }
        pub fn $fpow(x: $f, y: $f) -> $f { x.powf(y) }
        /// C's `fmod`: the remainder of truncating division, the sign of `x`.
        pub fn $fmod(x: $f, y: $f) -> $f { x % y }
        pub fn $fsignum(x: $f) -> $f { if x.is_nan() { x } else { ((x > 0.0) as i32 - (x < 0.0) as i32) as $f } }
        pub fn $sqrt(x: $f) -> $f { x.sqrt() }
        pub fn $cbrt(x: $f) -> $f { x.cbrt() }
        pub fn $exp(x: $f) -> $f { x.exp() }
        pub fn $log(x: $f) -> $f { x.ln() }
        pub fn $log2(x: $f) -> $f { x.log2() }
        pub fn $log10(x: $f) -> $f { x.log10() }
        pub fn $log1p(x: $f) -> $f { x.ln_1p() }
        pub fn $sin(x: $f) -> $f { x.sin() }
        pub fn $cos(x: $f) -> $f { x.cos() }
        pub fn $tan(x: $f) -> $f { x.tan() }
        pub fn $asin(x: $f) -> $f { x.asin() }
        pub fn $acos(x: $f) -> $f { x.acos() }
        pub fn $atan(x: $f) -> $f { x.atan() }
        pub fn $sinh(x: $f) -> $f { x.sinh() }
        pub fn $cosh(x: $f) -> $f { x.cosh() }
        pub fn $tanh(x: $f) -> $f { x.tanh() }
        pub fn $asinh(x: $f) -> $f { x.asinh() }
        pub fn $acosh(x: $f) -> $f { x.acosh() }
        pub fn $atanh(x: $f) -> $f { x.atanh() }
        pub fn $atan2(x: $f, y: $f) -> $f { x.atan2(y) }
        pub fn $hypot(x: $f, y: $f) -> $f { x.hypot(y) }
        /// C's `rint`: to the nearest, ties to even.
        pub fn $round(x: $f) -> $f { x.round_ties_even() }
        pub fn $floor(x: $f) -> $f { x.floor() }
        pub fn $ceil(x: $f) -> $f { x.ceil() }
        pub fn $trunc(x: $f) -> $f { x.trunc() }
        pub fn $isnan(x: $f) -> bool { x.is_nan() }
        pub fn $isinf(x: $f) -> bool { x.is_infinite() }
        pub fn $fma(a: $f, b: $f, c: $f) -> $f { a.mul_add(b, c) }
        pub fn $mad(a: $f, b: $f, c: $f) -> $f { a * b + c }
        pub fn $copysign(x: $f, y: $f) -> $f { x.copysign(y) }
        pub fn $to_bits(x: $f) -> $int { x.to_bits() as $int }
        pub fn $from_bits(x: $int) -> $f { <$f>::from_bits(x as $uint) }
    };
}

float_fns!(f32, 32, fabs32, fmin32, fmax32, fpow32, fmod32, fsignum32, fun_sqrt32, fun_cbrt32, fun_exp32, fun_log32,
    fun_log2_32, fun_log10_32, fun_log1p_32, fun_sin32, fun_cos32, fun_tan32, fun_asin32, fun_acos32, fun_atan32,
    fun_sinh32, fun_cosh32, fun_tanh32, fun_asinh32, fun_acosh32, fun_atanh32, fun_atan2_32, fun_hypot32,
    fun_round32, fun_floor32, fun_ceil32, fun_trunc32, fun_isnan32, fun_isinf32, fun_fma32, fun_mad32,
    fun_copysign32, fun_to_bits32, fun_from_bits32, i32, u32);
float_fns!(f64, 64, fabs64, fmin64, fmax64, fpow64, fmod64, fsignum64, fun_sqrt64, fun_cbrt64, fun_exp64, fun_log64,
    fun_log2_64, fun_log10_64, fun_log1p_64, fun_sin64, fun_cos64, fun_tan64, fun_asin64, fun_acos64, fun_atan64,
    fun_sinh64, fun_cosh64, fun_tanh64, fun_asinh64, fun_acosh64, fun_atanh64, fun_atan2_64, fun_hypot64,
    fun_round64, fun_floor64, fun_ceil64, fun_trunc64, fun_isnan64, fun_isinf64, fun_fma64, fun_mad64,
    fun_copysign64, fun_to_bits64, fun_from_bits64, i64, u64);

#[cfg(test)]
mod tests {
    use super::*;

    /// The division family against the C definitions' worked cases: floor division and modulo, truncation, the
    /// `-1` divisor, the safe forms.
    #[test]
    fn division_is_scalar_h_s() {
        assert_eq!(sdiv32(7, 2), 3);
        assert_eq!(sdiv32(-7, 2), -4);
        assert_eq!(sdiv32(7, -2), -4);
        assert_eq!(sdiv32(i32::MIN, -1), i32::MIN);
        assert_eq!(smod32(-7, 2), 1);
        assert_eq!(smod32(7, -2), -1);
        assert_eq!(smod32(5, -1), 0);
        assert_eq!(squot32(-7, 2), -3);
        assert_eq!(srem32(-7, 2), -1);
        assert_eq!(sceil_div64(7, 2), 4);
        assert_eq!(sceil_div64(-7, 2), -3);
        assert_eq!(sceil_div64(7, -2), -3);
        assert_eq!(sdiv_safe32(5, 0), 0);
        assert_eq!(umod32(-1, 10), (u32::MAX % 10) as i32);
        assert_eq!(udiv64(-2, 2), (u64::MAX / 2) as i64);
        assert_eq!(uceil_div32(7, 2), 4);
    }

    #[test]
    fn bit_operations_are_on_the_unsigned_bits() {
        assert_eq!(lshr32(-1, 28), 15);
        assert_eq!(ashr32(-16, 2), -4);
        assert_eq!(shl8(1, 7), i8::MIN);
        assert!(ult32(1, -1));
        assert!(!slt32(1, -1));
        assert_eq!(umax32(1, -1), -1);
        assert_eq!(pow32(3, 4), 81);
        assert_eq!(pow32(2, 32), 0);
        assert_eq!(fun_umul_hi32(-1, -1), -2);
        assert_eq!(fun_smul_hi32(-1, -1), 0);
    }

    #[test]
    fn conversions_follow_c() {
        assert_eq!(sext_i64_i32(0x1_0000_0005), 5);
        assert_eq!(zext_i32_i64(-1), 0xffff_ffff);
        assert_eq!(sext_i32_i64(-1), -1);
        assert_eq!(fptosi_f32_i32(f32::NAN), 0);
        assert_eq!(fptosi_f32_i32(3e9), 0);
        assert_eq!(fptosi_f32_i32(-2.7), -2);
        assert_eq!(fptoui_f32_i32(3e9), 3_000_000_000u32 as i32);
        assert_eq!(uitofp_i32_f32(-1), 4294967296.0);
        assert_eq!(fun_round32(2.5), 2.0);
        assert_eq!(fun_round32(3.5), 4.0);
        assert_eq!(fmod32(-7.0, 2.0), -1.0);
        assert_eq!(fmin32(f32::NAN, 1.0), 1.0);
    }
}
