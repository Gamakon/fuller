//! The WGSL kingdom's host side. The kingdom itself is rows of the gene frame's fixed schema (`crate::wgsl_rows`,
//! `kingdoms/master/`, variants `wgsl.gram` and `wgsl.image`, written by `kingdoms/wgsl_true/gen_rows.py`);
//! [`true_kingdom::TrueKingdom::from_rows`] is the bridge phylu's runners read it through. Here: naga's front end and the
//! installed target ([`naga_shape`], [`target`]), the work guard every device run goes through ([`work_guard`]), the
//! device and its error scopes ([`oracle::device`]), the Gram problem's host ([`gram`]) and the image host
//! ([`image`], [`cuneus_originals`]: the cuneus shaders' bindings), and [`snap`] over a kernel tree (the share pass is
//! the rows' own, `crate::wgsl_rows::share`).
//!
//! Compiled only under the `wgsl` feature (naga in and out).

pub mod cuneus_originals;
pub mod gram;
pub mod image;
pub mod naga_shape;
pub mod oracle;
pub mod snap;
pub mod target;
pub mod true_kingdom;
pub mod work_guard;
