extern crate bindgen;
extern crate cc;
extern crate cmake;

use std::env;
use std::path::PathBuf;
use cmake::Config;

fn main() {
    let mut dst = Config::new("flann")
        .define("BUILD_EXAMPLES", "OFF")
        .define("BUILD_TESTS", "OFF")
        .define("BUILD_DOC", "OFF")
        .build();
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
