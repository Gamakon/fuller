use std::time::Instant;
fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let (model, truth, nv) = (&a[0], &a[1], a[2].parse::<usize>().unwrap());
    let vars: Vec<String> = (0..nv).map(|i| format!("x_{i}")).collect();
    let t0 = Instant::now();
    let r = fuller::srbench_equiv::srbench_equivalent(model, truth, &vars);
    println!("{:?}\t{:.2}ms", r, t0.elapsed().as_secs_f64()*1000.0);
}
