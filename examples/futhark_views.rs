//! Write every committed Futhark view's exact table (`fuller::futhark_view::write_committed`): the tables the engine
//! computes from the master and each specification, for the generator's witnesses.
//!
//! ```text
//! cargo run --release --example futhark_views -- DIR
//! python3 kingdoms/futhark/gen_tables.py --check --views DIR
//! ```

fn main() {
    let Some(dir) = std::env::args().nth(1) else {
        eprintln!("usage: futhark_views DIR");
        std::process::exit(2);
    };
    match fuller::futhark_view::write_committed(std::path::Path::new(&dir)) {
        Ok(views) => {
            for (sub, v) in views {
                println!("{dir}/{sub}: {} types, {} rows, fingerprint {:016x}", v.types_tsv.lines().count() - 1, v.functions_tsv.lines().count() - 1, v.fingerprint());
            }
        }
        Err(e) => {
            eprintln!("futhark_views: {e}");
            std::process::exit(1);
        }
    }
}
