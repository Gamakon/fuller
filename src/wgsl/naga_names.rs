//! The names the WGSL kingdom gives naga's operator enums, and their inverse.
//!
//! The reader ([`super::reader`]) spells a naga node as the Debug name of its
//! operator (`Binary::Add`, `Unary::Negate`, `Relational::All`, `Math::Sin`)
//! and a builtin argument as the snake case of the `BuiltIn` Debug name
//! (`builtin.global_invocation_id`). The rebuild needs the inverse, from
//! those names back to the enum values. Both directions are explicit, total
//! match tables: every `*_name` is an exhaustive match with no wildcard arm,
//! so a variant added or renamed in a future naga fails to compile here, and
//! the tests compare each name against naga's own Debug spelling.
//!
//! `BuiltIn::Position { invariant }` is the one variant carrying data: it
//! names to `"position"` for either flag, and `"position"` reads back with
//! `invariant: false`. The flag is lost by design; compute kernels never
//! carry the builtin.

use naga::{BinaryOperator, BuiltIn, MathFunction, RelationalFunction, UnaryOperator};

/// `"Add"` → `BinaryOperator::Add`, and so on; `None` for an unknown name.
pub fn binary_operator(name: &str) -> Option<BinaryOperator> {
    Some(match name {
        "Add" => BinaryOperator::Add,
        "Subtract" => BinaryOperator::Subtract,
        "Multiply" => BinaryOperator::Multiply,
        "Divide" => BinaryOperator::Divide,
        "Modulo" => BinaryOperator::Modulo,
        "Equal" => BinaryOperator::Equal,
        "NotEqual" => BinaryOperator::NotEqual,
        "Less" => BinaryOperator::Less,
        "LessEqual" => BinaryOperator::LessEqual,
        "Greater" => BinaryOperator::Greater,
        "GreaterEqual" => BinaryOperator::GreaterEqual,
        "And" => BinaryOperator::And,
        "ExclusiveOr" => BinaryOperator::ExclusiveOr,
        "InclusiveOr" => BinaryOperator::InclusiveOr,
        "LogicalAnd" => BinaryOperator::LogicalAnd,
        "LogicalOr" => BinaryOperator::LogicalOr,
        "ShiftLeft" => BinaryOperator::ShiftLeft,
        "ShiftRight" => BinaryOperator::ShiftRight,
        _ => return None,
    })
}

/// The Debug name of a binary operator.
pub fn binary_operator_name(op: BinaryOperator) -> &'static str {
    match op {
        BinaryOperator::Add => "Add",
        BinaryOperator::Subtract => "Subtract",
        BinaryOperator::Multiply => "Multiply",
        BinaryOperator::Divide => "Divide",
        BinaryOperator::Modulo => "Modulo",
        BinaryOperator::Equal => "Equal",
        BinaryOperator::NotEqual => "NotEqual",
        BinaryOperator::Less => "Less",
        BinaryOperator::LessEqual => "LessEqual",
        BinaryOperator::Greater => "Greater",
        BinaryOperator::GreaterEqual => "GreaterEqual",
        BinaryOperator::And => "And",
        BinaryOperator::ExclusiveOr => "ExclusiveOr",
        BinaryOperator::InclusiveOr => "InclusiveOr",
        BinaryOperator::LogicalAnd => "LogicalAnd",
        BinaryOperator::LogicalOr => "LogicalOr",
        BinaryOperator::ShiftLeft => "ShiftLeft",
        BinaryOperator::ShiftRight => "ShiftRight",
    }
}

/// `"Negate"` → `UnaryOperator::Negate`, and so on; `None` for an unknown name.
pub fn unary_operator(name: &str) -> Option<UnaryOperator> {
    Some(match name {
        "Negate" => UnaryOperator::Negate,
        "LogicalNot" => UnaryOperator::LogicalNot,
        "BitwiseNot" => UnaryOperator::BitwiseNot,
        _ => return None,
    })
}

/// The Debug name of a unary operator.
pub fn unary_operator_name(op: UnaryOperator) -> &'static str {
    match op {
        UnaryOperator::Negate => "Negate",
        UnaryOperator::LogicalNot => "LogicalNot",
        UnaryOperator::BitwiseNot => "BitwiseNot",
    }
}

/// `"All"` → `RelationalFunction::All`, and so on; `None` for an unknown name.
pub fn relational_function(name: &str) -> Option<RelationalFunction> {
    Some(match name {
        "All" => RelationalFunction::All,
        "Any" => RelationalFunction::Any,
        "IsNan" => RelationalFunction::IsNan,
        "IsInf" => RelationalFunction::IsInf,
        _ => return None,
    })
}

/// The Debug name of a relational function.
pub fn relational_function_name(f: RelationalFunction) -> &'static str {
    match f {
        RelationalFunction::All => "All",
        RelationalFunction::Any => "Any",
        RelationalFunction::IsNan => "IsNan",
        RelationalFunction::IsInf => "IsInf",
    }
}

/// `"Sin"` → `MathFunction::Sin`, and so on; `None` for an unknown name.
pub fn math_function(name: &str) -> Option<MathFunction> {
    Some(match name {
        "Abs" => MathFunction::Abs,
        "Min" => MathFunction::Min,
        "Max" => MathFunction::Max,
        "Clamp" => MathFunction::Clamp,
        "Saturate" => MathFunction::Saturate,
        "Cos" => MathFunction::Cos,
        "Cosh" => MathFunction::Cosh,
        "Sin" => MathFunction::Sin,
        "Sinh" => MathFunction::Sinh,
        "Tan" => MathFunction::Tan,
        "Tanh" => MathFunction::Tanh,
        "Acos" => MathFunction::Acos,
        "Asin" => MathFunction::Asin,
        "Atan" => MathFunction::Atan,
        "Atan2" => MathFunction::Atan2,
        "Asinh" => MathFunction::Asinh,
        "Acosh" => MathFunction::Acosh,
        "Atanh" => MathFunction::Atanh,
        "Radians" => MathFunction::Radians,
        "Degrees" => MathFunction::Degrees,
        "Ceil" => MathFunction::Ceil,
        "Floor" => MathFunction::Floor,
        "Round" => MathFunction::Round,
        "Fract" => MathFunction::Fract,
        "Trunc" => MathFunction::Trunc,
        "Modf" => MathFunction::Modf,
        "Frexp" => MathFunction::Frexp,
        "Ldexp" => MathFunction::Ldexp,
        "Exp" => MathFunction::Exp,
        "Exp2" => MathFunction::Exp2,
        "Log" => MathFunction::Log,
        "Log2" => MathFunction::Log2,
        "Pow" => MathFunction::Pow,
        "Dot" => MathFunction::Dot,
        "Outer" => MathFunction::Outer,
        "Cross" => MathFunction::Cross,
        "Distance" => MathFunction::Distance,
        "Length" => MathFunction::Length,
        "Normalize" => MathFunction::Normalize,
        "FaceForward" => MathFunction::FaceForward,
        "Reflect" => MathFunction::Reflect,
        "Refract" => MathFunction::Refract,
        "Sign" => MathFunction::Sign,
        "Fma" => MathFunction::Fma,
        "Mix" => MathFunction::Mix,
        "Step" => MathFunction::Step,
        "SmoothStep" => MathFunction::SmoothStep,
        "Sqrt" => MathFunction::Sqrt,
        "InverseSqrt" => MathFunction::InverseSqrt,
        "Inverse" => MathFunction::Inverse,
        "Transpose" => MathFunction::Transpose,
        "Determinant" => MathFunction::Determinant,
        "CountTrailingZeros" => MathFunction::CountTrailingZeros,
        "CountLeadingZeros" => MathFunction::CountLeadingZeros,
        "CountOneBits" => MathFunction::CountOneBits,
        "ReverseBits" => MathFunction::ReverseBits,
        "ExtractBits" => MathFunction::ExtractBits,
        "InsertBits" => MathFunction::InsertBits,
        "FindLsb" => MathFunction::FindLsb,
        "FindMsb" => MathFunction::FindMsb,
        "Pack4x8snorm" => MathFunction::Pack4x8snorm,
        "Pack4x8unorm" => MathFunction::Pack4x8unorm,
        "Pack2x16snorm" => MathFunction::Pack2x16snorm,
        "Pack2x16unorm" => MathFunction::Pack2x16unorm,
        "Pack2x16float" => MathFunction::Pack2x16float,
        "Unpack4x8snorm" => MathFunction::Unpack4x8snorm,
        "Unpack4x8unorm" => MathFunction::Unpack4x8unorm,
        "Unpack2x16snorm" => MathFunction::Unpack2x16snorm,
        "Unpack2x16unorm" => MathFunction::Unpack2x16unorm,
        "Unpack2x16float" => MathFunction::Unpack2x16float,
        _ => return None,
    })
}

/// The Debug name of a math function.
pub fn math_function_name(f: MathFunction) -> &'static str {
    match f {
        MathFunction::Abs => "Abs",
        MathFunction::Min => "Min",
        MathFunction::Max => "Max",
        MathFunction::Clamp => "Clamp",
        MathFunction::Saturate => "Saturate",
        MathFunction::Cos => "Cos",
        MathFunction::Cosh => "Cosh",
        MathFunction::Sin => "Sin",
        MathFunction::Sinh => "Sinh",
        MathFunction::Tan => "Tan",
        MathFunction::Tanh => "Tanh",
        MathFunction::Acos => "Acos",
        MathFunction::Asin => "Asin",
        MathFunction::Atan => "Atan",
        MathFunction::Atan2 => "Atan2",
        MathFunction::Asinh => "Asinh",
        MathFunction::Acosh => "Acosh",
        MathFunction::Atanh => "Atanh",
        MathFunction::Radians => "Radians",
        MathFunction::Degrees => "Degrees",
        MathFunction::Ceil => "Ceil",
        MathFunction::Floor => "Floor",
        MathFunction::Round => "Round",
        MathFunction::Fract => "Fract",
        MathFunction::Trunc => "Trunc",
        MathFunction::Modf => "Modf",
        MathFunction::Frexp => "Frexp",
        MathFunction::Ldexp => "Ldexp",
        MathFunction::Exp => "Exp",
        MathFunction::Exp2 => "Exp2",
        MathFunction::Log => "Log",
        MathFunction::Log2 => "Log2",
        MathFunction::Pow => "Pow",
        MathFunction::Dot => "Dot",
        MathFunction::Outer => "Outer",
        MathFunction::Cross => "Cross",
        MathFunction::Distance => "Distance",
        MathFunction::Length => "Length",
        MathFunction::Normalize => "Normalize",
        MathFunction::FaceForward => "FaceForward",
        MathFunction::Reflect => "Reflect",
        MathFunction::Refract => "Refract",
        MathFunction::Sign => "Sign",
        MathFunction::Fma => "Fma",
        MathFunction::Mix => "Mix",
        MathFunction::Step => "Step",
        MathFunction::SmoothStep => "SmoothStep",
        MathFunction::Sqrt => "Sqrt",
        MathFunction::InverseSqrt => "InverseSqrt",
        MathFunction::Inverse => "Inverse",
        MathFunction::Transpose => "Transpose",
        MathFunction::Determinant => "Determinant",
        MathFunction::CountTrailingZeros => "CountTrailingZeros",
        MathFunction::CountLeadingZeros => "CountLeadingZeros",
        MathFunction::CountOneBits => "CountOneBits",
        MathFunction::ReverseBits => "ReverseBits",
        MathFunction::ExtractBits => "ExtractBits",
        MathFunction::InsertBits => "InsertBits",
        MathFunction::FindLsb => "FindLsb",
        MathFunction::FindMsb => "FindMsb",
        MathFunction::Pack4x8snorm => "Pack4x8snorm",
        MathFunction::Pack4x8unorm => "Pack4x8unorm",
        MathFunction::Pack2x16snorm => "Pack2x16snorm",
        MathFunction::Pack2x16unorm => "Pack2x16unorm",
        MathFunction::Pack2x16float => "Pack2x16float",
        MathFunction::Unpack4x8snorm => "Unpack4x8snorm",
        MathFunction::Unpack4x8unorm => "Unpack4x8unorm",
        MathFunction::Unpack2x16snorm => "Unpack2x16snorm",
        MathFunction::Unpack2x16unorm => "Unpack2x16unorm",
        MathFunction::Unpack2x16float => "Unpack2x16float",
    }
}

/// `"global_invocation_id"` → `BuiltIn::GlobalInvocationId`, and so on (the
/// snake case of the Debug name, so `work_group_id`, not WGSL's
/// `workgroup_id`); `"position"` → `Position { invariant: false }`; `None`
/// for an unknown name.
pub fn builtin(name: &str) -> Option<BuiltIn> {
    Some(match name {
        "position" => BuiltIn::Position { invariant: false },
        "view_index" => BuiltIn::ViewIndex,
        "base_instance" => BuiltIn::BaseInstance,
        "base_vertex" => BuiltIn::BaseVertex,
        "clip_distance" => BuiltIn::ClipDistance,
        "cull_distance" => BuiltIn::CullDistance,
        "instance_index" => BuiltIn::InstanceIndex,
        "point_size" => BuiltIn::PointSize,
        "vertex_index" => BuiltIn::VertexIndex,
        "frag_depth" => BuiltIn::FragDepth,
        "point_coord" => BuiltIn::PointCoord,
        "front_facing" => BuiltIn::FrontFacing,
        "primitive_index" => BuiltIn::PrimitiveIndex,
        "sample_index" => BuiltIn::SampleIndex,
        "sample_mask" => BuiltIn::SampleMask,
        "global_invocation_id" => BuiltIn::GlobalInvocationId,
        "local_invocation_id" => BuiltIn::LocalInvocationId,
        "local_invocation_index" => BuiltIn::LocalInvocationIndex,
        "work_group_id" => BuiltIn::WorkGroupId,
        "work_group_size" => BuiltIn::WorkGroupSize,
        "num_work_groups" => BuiltIn::NumWorkGroups,
        "num_subgroups" => BuiltIn::NumSubgroups,
        "subgroup_id" => BuiltIn::SubgroupId,
        "subgroup_size" => BuiltIn::SubgroupSize,
        "subgroup_invocation_id" => BuiltIn::SubgroupInvocationId,
        _ => return None,
    })
}

/// The snake-case name of a builtin. `Position` names to `"position"` for
/// either value of `invariant`.
pub fn builtin_name(b: BuiltIn) -> &'static str {
    match b {
        BuiltIn::Position { .. } => "position",
        BuiltIn::ViewIndex => "view_index",
        BuiltIn::BaseInstance => "base_instance",
        BuiltIn::BaseVertex => "base_vertex",
        BuiltIn::ClipDistance => "clip_distance",
        BuiltIn::CullDistance => "cull_distance",
        BuiltIn::InstanceIndex => "instance_index",
        BuiltIn::PointSize => "point_size",
        BuiltIn::VertexIndex => "vertex_index",
        BuiltIn::FragDepth => "frag_depth",
        BuiltIn::PointCoord => "point_coord",
        BuiltIn::FrontFacing => "front_facing",
        BuiltIn::PrimitiveIndex => "primitive_index",
        BuiltIn::SampleIndex => "sample_index",
        BuiltIn::SampleMask => "sample_mask",
        BuiltIn::GlobalInvocationId => "global_invocation_id",
        BuiltIn::LocalInvocationId => "local_invocation_id",
        BuiltIn::LocalInvocationIndex => "local_invocation_index",
        BuiltIn::WorkGroupId => "work_group_id",
        BuiltIn::WorkGroupSize => "work_group_size",
        BuiltIn::NumWorkGroups => "num_work_groups",
        BuiltIn::NumSubgroups => "num_subgroups",
        BuiltIn::SubgroupId => "subgroup_id",
        BuiltIn::SubgroupSize => "subgroup_size",
        BuiltIn::SubgroupInvocationId => "subgroup_invocation_id",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const BINARY: [BinaryOperator; 18] = [
        BinaryOperator::Add,
        BinaryOperator::Subtract,
        BinaryOperator::Multiply,
        BinaryOperator::Divide,
        BinaryOperator::Modulo,
        BinaryOperator::Equal,
        BinaryOperator::NotEqual,
        BinaryOperator::Less,
        BinaryOperator::LessEqual,
        BinaryOperator::Greater,
        BinaryOperator::GreaterEqual,
        BinaryOperator::And,
        BinaryOperator::ExclusiveOr,
        BinaryOperator::InclusiveOr,
        BinaryOperator::LogicalAnd,
        BinaryOperator::LogicalOr,
        BinaryOperator::ShiftLeft,
        BinaryOperator::ShiftRight,
    ];

    const UNARY: [UnaryOperator; 3] = [UnaryOperator::Negate, UnaryOperator::LogicalNot, UnaryOperator::BitwiseNot];

    const RELATIONAL: [RelationalFunction; 4] = [
        RelationalFunction::All,
        RelationalFunction::Any,
        RelationalFunction::IsNan,
        RelationalFunction::IsInf,
    ];

    const MATH: [MathFunction; 70] = [
        MathFunction::Abs,
        MathFunction::Min,
        MathFunction::Max,
        MathFunction::Clamp,
        MathFunction::Saturate,
        MathFunction::Cos,
        MathFunction::Cosh,
        MathFunction::Sin,
        MathFunction::Sinh,
        MathFunction::Tan,
        MathFunction::Tanh,
        MathFunction::Acos,
        MathFunction::Asin,
        MathFunction::Atan,
        MathFunction::Atan2,
        MathFunction::Asinh,
        MathFunction::Acosh,
        MathFunction::Atanh,
        MathFunction::Radians,
        MathFunction::Degrees,
        MathFunction::Ceil,
        MathFunction::Floor,
        MathFunction::Round,
        MathFunction::Fract,
        MathFunction::Trunc,
        MathFunction::Modf,
        MathFunction::Frexp,
        MathFunction::Ldexp,
        MathFunction::Exp,
        MathFunction::Exp2,
        MathFunction::Log,
        MathFunction::Log2,
        MathFunction::Pow,
        MathFunction::Dot,
        MathFunction::Outer,
        MathFunction::Cross,
        MathFunction::Distance,
        MathFunction::Length,
        MathFunction::Normalize,
        MathFunction::FaceForward,
        MathFunction::Reflect,
        MathFunction::Refract,
        MathFunction::Sign,
        MathFunction::Fma,
        MathFunction::Mix,
        MathFunction::Step,
        MathFunction::SmoothStep,
        MathFunction::Sqrt,
        MathFunction::InverseSqrt,
        MathFunction::Inverse,
        MathFunction::Transpose,
        MathFunction::Determinant,
        MathFunction::CountTrailingZeros,
        MathFunction::CountLeadingZeros,
        MathFunction::CountOneBits,
        MathFunction::ReverseBits,
        MathFunction::ExtractBits,
        MathFunction::InsertBits,
        MathFunction::FindLsb,
        MathFunction::FindMsb,
        MathFunction::Pack4x8snorm,
        MathFunction::Pack4x8unorm,
        MathFunction::Pack2x16snorm,
        MathFunction::Pack2x16unorm,
        MathFunction::Pack2x16float,
        MathFunction::Unpack4x8snorm,
        MathFunction::Unpack4x8unorm,
        MathFunction::Unpack2x16snorm,
        MathFunction::Unpack2x16unorm,
        MathFunction::Unpack2x16float,
    ];

    const BUILTIN: [BuiltIn; 25] = [
        BuiltIn::Position { invariant: false },
        BuiltIn::ViewIndex,
        BuiltIn::BaseInstance,
        BuiltIn::BaseVertex,
        BuiltIn::ClipDistance,
        BuiltIn::CullDistance,
        BuiltIn::InstanceIndex,
        BuiltIn::PointSize,
        BuiltIn::VertexIndex,
        BuiltIn::FragDepth,
        BuiltIn::PointCoord,
        BuiltIn::FrontFacing,
        BuiltIn::PrimitiveIndex,
        BuiltIn::SampleIndex,
        BuiltIn::SampleMask,
        BuiltIn::GlobalInvocationId,
        BuiltIn::LocalInvocationId,
        BuiltIn::LocalInvocationIndex,
        BuiltIn::WorkGroupId,
        BuiltIn::WorkGroupSize,
        BuiltIn::NumWorkGroups,
        BuiltIn::NumSubgroups,
        BuiltIn::SubgroupId,
        BuiltIn::SubgroupSize,
        BuiltIn::SubgroupInvocationId,
    ];

    /// Snake case of a Debug name, stopping at the first space so the
    /// `Position { invariant: false }` payload does not reach the name.
    fn snake_of_debug(debug: &str) -> String {
        let mut out = String::new();
        for (i, c) in debug.chars().take_while(|c| *c != ' ').enumerate() {
            if c.is_uppercase() {
                if i > 0 {
                    out.push('_');
                }
                out.push(c.to_ascii_lowercase());
            } else {
                out.push(c);
            }
        }
        out
    }

    #[test]
    fn binary_round_trip_matches_debug() {
        for op in BINARY {
            assert_eq!(binary_operator_name(op), format!("{op:?}"), "{op:?}");
            assert_eq!(binary_operator(binary_operator_name(op)), Some(op), "{op:?}");
        }
    }

    #[test]
    fn unary_round_trip_matches_debug() {
        for op in UNARY {
            assert_eq!(unary_operator_name(op), format!("{op:?}"), "{op:?}");
            assert_eq!(unary_operator(unary_operator_name(op)), Some(op), "{op:?}");
        }
    }

    #[test]
    fn relational_round_trip_matches_debug() {
        for f in RELATIONAL {
            assert_eq!(relational_function_name(f), format!("{f:?}"), "{f:?}");
            assert_eq!(relational_function(relational_function_name(f)), Some(f), "{f:?}");
        }
    }

    #[test]
    fn math_round_trip_matches_debug() {
        for f in MATH {
            assert_eq!(math_function_name(f), format!("{f:?}"), "{f:?}");
            assert_eq!(math_function(math_function_name(f)), Some(f), "{f:?}");
        }
    }

    #[test]
    fn builtin_round_trip_matches_snake_of_debug() {
        for b in BUILTIN {
            assert_eq!(builtin_name(b), snake_of_debug(&format!("{b:?}")), "{b:?}");
            assert_eq!(builtin(builtin_name(b)), Some(b), "{b:?}");
        }
        assert_eq!(builtin_name(BuiltIn::Position { invariant: true }), "position");
        assert_eq!(builtin("position"), Some(BuiltIn::Position { invariant: false }));
    }

    #[test]
    fn unknown_names_are_none() {
        assert_eq!(binary_operator("Nope"), None);
        assert_eq!(unary_operator("Nope"), None);
        assert_eq!(relational_function("Nope"), None);
        assert_eq!(math_function("Nope"), None);
        assert_eq!(builtin("nope"), None);
        assert_eq!(builtin("workgroup_id"), None);
    }
}
