extern crate bindgen;
extern crate cc;
extern crate pkg_config;

use std::env;
use std::path::PathBuf;

fn main() {
    if libaries_are_present() {
        build_code();
    }

    let bindings = bindgen::Builder::default()
        .header("wrapper.h")
        .generate()
        .expect("Unable to generate bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}

fn libaries_are_present() -> bool {
    return grab_dependencies().is_ok();
}

fn grab_dependencies() -> Result<(), pkg_config::Error> {
    pkg_config::probe_library("flann")?;
    Ok(())
}

fn build_code() {
    cc::Build::new()
        .cpp(true)
        .file("wrapper.cpp")
        .compile("wrapper.a");
}
