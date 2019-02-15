extern crate bindgen;
extern crate cmake;

use std::env;
use std::path::PathBuf;
use cmake::Config;

fn main() {
    // Global configuration for FLANN build.
    let mut config = Config::new("flann");
    config
        .define("BUILD_EXAMPLES", "OFF")
        .define("BUILD_TESTS", "OFF")
        .define("BUILD_DOC", "OFF")
        .define("BUILD_PYTHON_BINDINGS", "OFF")
        .define("BUILD_MATLAB_BINDINGS", "OFF")
        .define("USE_OPENMP", "ON");

    // Handle OS-specific requirements.
    let target_os = env::var("CARGO_CFG_TARGET_OS");
    match target_os.as_ref().map(|x| &**x) {
        Ok("linux") => {
            println!("cargo:rustc-link-lib=gomp");
            println!("cargo:rustc-link-lib=stdc++");
        },
        _ => {},
    }

    // Build FLANN and add it to cargo.
    let mut dst = config.build();
    dst.push("lib");
    println!("cargo:rustc-link-search=native={}", dst.display());
    println!("cargo:rustc-link-lib=static=flann_s");

    let bindings = bindgen::Builder::default()
        .header("flann/src/cpp/flann/flann.h")
        .generate()
        .expect("Unable to generate bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
